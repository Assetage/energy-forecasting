import os
import logging.config
from typing import Dict

import pandas as pd
from sklearn.model_selection import GridSearchCV
from omegaconf import DictConfig
import hydra
import marshmallow

from .data import time_series_split
from .entities.train_pipeline_params import TrainingPipelineParams, TrainingPipelineParamsSchema
from .features.build_features import prepare_dataset
from .models import train_model, make_prediction
from .utils import read_data, save_pkl_file

logger = logging.getLogger("optimizer_pipeline")


def optimaize_model_pipeline(
        training_pipeline_params: TrainingPipelineParams ) -> Dict[str, float]:

    data = read_data(training_pipeline_params.path_config.input_data_path)
    data_transformed = prepare_dataset(data, training_pipeline_params.feature_params)

    logger.info("Start transformer building...")

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(data_transformed)
    train_df, test_df = split_train_val_data(data_transformed, training_pipeline_params.splitting_params)

    train_target = get_target(train_df, training_pipeline_params.feature_params)
    train_features = pd.DataFrame(transformer.transform(train_df))

    logger.info("Start gridsearch...")

    model_name = training_pipeline_params.opt_params.model_name
    model_params = training_pipeline_params.opt_params.model_param_space

    clf = GridSearchCV(model_name, model_params, cv=3, scoring='accuracy')

    clf.fit(train_df, train_target)

def opt_pipeline_start(cfg: DictConfig):
    schema = TrainingPipelineParamsSchema()
    try:
        params = schema.load(cfg)
        optimaize_model_pipeline(params)
    except marshmallow.exceptions.ValidationError as e:
        logger.error(f"Configuration validation error: {e.messages}")
        raise e