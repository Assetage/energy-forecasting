import logging.config
from typing import NoReturn

import marshmallow
from omegaconf import DictConfig

from .entities.predict_pipeline_params import (
    PredictingPipelineParams,
    PredictingPipelineParamsSchema,
)
from .features.build_features import prepare_dataset
from .models import create_future_df, make_prediction
from .utils import load_pkl_file, read_data, save_pred_plot

logger = logging.getLogger("ml_project/predict_pipeline")


def predict_pipeline(evaluating_pipeline_params: PredictingPipelineParams) -> NoReturn:
    logger.info("Start prediction pipeline")
    data = read_data(evaluating_pipeline_params.predict_params.input_data_path)
    logger.info(f"Dataset shape is {data.shape}")

    logger.info("Loading model...")
    model = load_pkl_file(evaluating_pipeline_params.predict_params.model_path)

    logger.info("Creating future df and concating it with an existing...")
    df_and_future = create_future_df(data, evaluating_pipeline_params.feature_params)

    logger.info("Building features...")
    data_transformed = prepare_dataset(
        df_and_future, evaluating_pipeline_params.feature_params, dropna_bool=False
    )
    df_sorted = data_transformed.sort_index()

    logger.info("Start prediction")
    df_w_predictions = make_prediction(
        df_sorted, model, evaluating_pipeline_params.feature_params
    )

    df_w_predictions.to_csv(
        evaluating_pipeline_params.predict_params.output_data_path, header=True
    )
    logger.info(
        "Prediction is done and saved to the file"
        f" {evaluating_pipeline_params.predict_params.output_data_path}"
    )

    logger.info("Saving predictions plot")
    save_pred_plot(
        df_w_predictions, evaluating_pipeline_params.predict_params.output_plot_path
    )


def predict_pipeline_start(cfg: DictConfig):
    schema = PredictingPipelineParamsSchema()
    try:
        params = schema.load(cfg)
        predict_pipeline(params)
    except marshmallow.exceptions.ValidationError as e:
        logger.error(f"Configuration validation error: {e.messages}")
        raise e
