import os
import logging.config

import pandas as pd
from omegaconf import DictConfig
import hydra

from src.entities.predict_pipeline_params import PredictingPipelineParams, \
    PredictingPipelineParamsSchema
from src.models import make_prediction
from src.utils import read_data, load_pkl_file

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


def predict_pipeline_start(cfg=None):
    if cfg is None:
        cfg = hydra.utils.get_original_cwd() + "/configs/predict_config.yaml"
        with open(cfg, 'r') as f:
            cfg= omegaconf.load(f)
    os.chdir(hydra.utils.to_absolute_path(".."))
    schema = PredictingPipelineParamsSchema()
    params = schema.load(cfg)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_start()