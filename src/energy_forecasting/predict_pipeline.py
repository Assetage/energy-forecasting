import os
import logging.config

import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra

from .entities.predict_pipeline_params import PredictingPipelineParams, \
    PredictingPipelineParamsSchema
from .models import make_prediction
from .utils import read_data, load_pkl_file

logger = logging.getLogger("ml_project/predict_pipeline")


def predict_pipeline(evaluating_pipeline_params: PredictingPipelineParams):
    logger.info("Start prediction pipeline")
    data = read_data(evaluating_pipeline_params.input_data_path)
    logger.info(f"Dataset shape is {data.shape}")

    logger.info("Loading transformer...")
    transformer = load_pkl_file(evaluating_pipeline_params.pipeline_path)
    transformed_data = pd.DataFrame(transformer.transform(data))

    logger.info("Loading model...")
    model = load_pkl_file(evaluating_pipeline_params.model_path)

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
    if cfg is None:
        cfg = hydra.utils.get_original_cwd() + "/src/energy_forecasting/conf/predict_config.yaml"
        with open(cfg, 'r') as f:
            cfg= OmegaConf.load(f)
    schema = PredictingPipelineParamsSchema()
    params = schema.load(cfg)
    predict_pipeline(params)