import os
import logging.config
from typing import NoReturn

import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
import marshmallow
from .entities.predict_pipeline_params import PredictingPipelineParams, \
    PredictingPipelineParamsSchema
from .features.build_features import prepare_dataset
from .models import make_prediction, create_future_df
from .utils import read_data, load_pkl_file, save_pred_plot

logger = logging.getLogger("ml_project/predict_pipeline")


def predict_pipeline(evaluating_pipeline_params: PredictingPipelineParams) -> NoReturn:
    logger.info("Start prediction pipeline")
    data = read_data(evaluating_pipeline_params.predict_params.input_data_path)
    logger.info(f"Dataset shape is {data.shape}")

    logger.info("Loading model...")
    model = load_pkl_file(evaluating_pipeline_params.predict_params.model_path)

    logger.info("Creating future df and concating it with an existing...")
    df_and_future = create_future_df(data, evaluating_pipeline_params.feature_params)
    print("df and future", df_and_future["isFuture"].value_counts())

    logger.info("Building features...")
    data_transformed = prepare_dataset(df_and_future, evaluating_pipeline_params.feature_params, dropna_bool=False)
    df_sorted = data_transformed.sort_index()
    print("df and future", df_sorted["isFuture"].value_counts())

    logger.info("Start prediction")
    df_w_predictions = make_prediction(df_sorted, model, evaluating_pipeline_params.feature_params)

    df_w_predictions.to_csv(evaluating_pipeline_params.predict_params.output_data_path, header=True)
    logger.info(
        f"Prediction is done and saved to the file {evaluating_pipeline_params.predict_params.output_data_path}"
    )

    logger.info("Saving predictions plot")
    save_pred_plot(df_w_predictions, evaluating_pipeline_params.predict_params.output_plot_path)


def predict_pipeline_start(cfg: DictConfig):
    schema = PredictingPipelineParamsSchema()
    try:
        params = schema.load(cfg)
        predict_pipeline(params)
    except marshmallow.exceptions.ValidationError as e:
        logger.error(f"Configuration validation error: {e.messages}")
        raise e