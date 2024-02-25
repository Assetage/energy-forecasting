import os
import logging.config

import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
import marshmallow
from .entities.predict_pipeline_params import PredictingPipelineParams, \
    PredictingPipelineParamsSchema
from .features.build_features import prepare_dataset
from .models import make_prediction
from .utils import read_data, load_pkl_file

logger = logging.getLogger("ml_project/predict_pipeline")


def predict_pipeline(evaluating_pipeline_params: PredictingPipelineParams):
    logger.info("Start prediction pipeline")
    data = read_data(evaluating_pipeline_params.input_data_path)
    logger.info(f"Dataset shape is {data.shape}")

    logger.info("Loading model...")
    model = load_pkl_file(evaluating_pipeline_params.model_path)

    logger.info("Building features...")
    data_transformed = prepare_dataset(data, evaluating_pipeline_params.feature_params)
    df_sorted = data_transformed.sort_index()

    logger.info("Start prediction")
    predicts = make_prediction(
        model,
        transformed_data,
    )

    df_predicts = pd.DataFrame(predicts)

    df_predicts.to_csv(evaluating_pipeline_params.output_data_path, header=False)
    logger.info(
        f"Prediction is done and saved to the file {evaluating_pipeline_params.output_data_path}"
    )
    return df_predicts


def predict_pipeline_start(cfg: DictConfig):
    schema = PredictingPipelineParamsSchema()
    try:
        params = schema.load(cfg)
        predict_pipeline(params)
    except marshmallow.exceptions.ValidationError as e:
        logger.error(f"Configuration validation error: {e.messages}")
        raise e