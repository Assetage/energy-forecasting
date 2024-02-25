import os
import logging.config
from typing import Dict, NoReturn

from omegaconf import DictConfig, OmegaConf
import marshmallow
import pandas as pd
import numpy as np
import hydra

from .data import time_series_split
from .entities.train_pipeline_params import TrainingPipelineParams, TrainingPipelineParamsSchema
from .features.build_features import prepare_dataset
from .models import train_model, run_cross_validation, select_model
from .utils import read_data, save_pkl_file, save_cross_val_results_to_json

logger = logging.getLogger("train_pipeline")


def train_pipeline(
        training_pipeline_params: TrainingPipelineParams,
) -> NoReturn:
    logger.info(f"Start train pipeline with params {OmegaConf.to_yaml(training_pipeline_params)}")
    model_type = training_pipeline_params.train_params['model_type']
    logger.info(f"Model is {model_type}")

    data = read_data(training_pipeline_params.path_config.input_data_path)

    logger.info("Building features...")
    data_transformed = prepare_dataset(data, training_pipeline_params.feature_params)

    logger.info("Splitting data into train and test...")
    df_sorted, tss = time_series_split(data_transformed, training_pipeline_params.splitting_params)

    logger.info("Model initialization...")
    model = select_model(training_pipeline_params.train_params)

    logger.info("Running cross validation...")
    scores = run_cross_validation(df_sorted, 
                                  tss, 
                                  training_pipeline_params.feature_params,
                                  model)
    save_cross_val_results_to_json(training_pipeline_params.path_config.cross_val_scores,
                                   training_pipeline_params.train_params["model_type"], 
                                   scores, 
                                   model.get_params())

    print("Mean RMSE accross folds is: ", np.mean(scores))
    print("RMSE Standard deviation accross folds is: ", np.std(scores))
    print("Full track of cross validation can be seen here: ", training_pipeline_params.path_config.cross_val_scores)
    
    logger.info("Starting model training...")
    model = train_model(df_sorted, 
                        model, 
                        training_pipeline_params.feature_params)
    logger.info("Model training is done.")

    save_pkl_file(model, training_pipeline_params.path_config.output_model_path +
                  training_pipeline_params.train_params["model_type"] + '.pkl')
    logger.info("Model is saved.")
    logger.info("Pipeline is finished.")


def train_pipeline_start(cfg: DictConfig):
    schema = TrainingPipelineParamsSchema()
    try:
        params = schema.load(cfg)
        train_pipeline(params)
    except marshmallow.exceptions.ValidationError as e:
        logger.error(f"Configuration validation error: {e.messages}")
        raise e