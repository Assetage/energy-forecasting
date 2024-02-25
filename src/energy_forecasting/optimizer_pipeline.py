import os
import logging.config
from typing import Dict

import pandas as pd
from sklearn.model_selection import GridSearchCV
from omegaconf import DictConfig, OmegaConf
import hydra
import marshmallow
from functools import partial
from hyperopt import fmin, tpe, Trials, hp


from .data import time_series_split
from .entities.optimizer_pipeline_params import OptimizerPipelineParams, OptimizerPipelineParamsSchema
from .features.build_features import prepare_dataset
from .models import select_model, objective, define_space
from .utils import read_data, save_optimization_results_to_json

logger = logging.getLogger("optimizer_pipeline")


def optimize_model_pipeline(
        optimizer_pipeline_params: OptimizerPipelineParams ) -> Dict[str, float]:

    logger.info(f"Start optimizer pipeline with params {OmegaConf.to_yaml(optimizer_pipeline_params)}")
    model_type = optimizer_pipeline_params.optimizer_params['model_type']
    logger.info(f"Model is {model_type}")

    data = read_data(optimizer_pipeline_params.path_config.input_data_path)

    logger.info("Building features...")
    data_transformed = prepare_dataset(data, optimizer_pipeline_params.feature_params)

    logger.info("Splitting data into train and test...")
    df_sorted, tss = time_series_split(data_transformed, optimizer_pipeline_params.splitting_params)

    logger.info("Model initialization...")
    model = select_model(optimizer_pipeline_params.optimizer_params)

    logger.info("Model optimization...")
    partial_objective = partial(objective, df_sorted, tss, optimizer_pipeline_params.feature_params, model)
    space = define_space(model_type, optimizer_pipeline_params.optimizer_params)

    trials = Trials()
    best_hyperparams = fmin(
        fn=partial_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials
    )
    save_optimization_results_to_json(optimizer_pipeline_params.path_config.opt_results,
                                      model_type,
                                      optimizer_pipeline_params.optimizer_params,
                                      best_hyperparams)
    logger.info("Best hyperparameters found:", best_hyperparams)
    logger.info("Pipeline is finished.")

def opt_pipeline_start(cfg: DictConfig):
    schema = OptimizerPipelineParamsSchema()
    try:
        params = schema.load(cfg)
        optimize_model_pipeline(params)
    except marshmallow.exceptions.ValidationError as e:
        logger.error(f"Configuration validation error: {e.messages}")
        raise e