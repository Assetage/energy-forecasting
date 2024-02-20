import os
import logging.config
from typing import Dict

import pandas as pd
from sklearn.model_selection import GridSearchCV
from omegaconf import DictConfig
import hydra

from src.data import split_train_val_data
from src.entities.train_pipeline_params import TrainingPipelineParams, TrainingPipelineParamsSchema
from src.features.build_features import get_target, build_transformer, prepare_dataset
from src.models import train_model, make_prediction, evaluate_model
from src.utils import read_data, save_pkl_file, save_metrics_to_json

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


@hydra.main(config_path="../configs", config_name="train_config")
def opt_pipeline_start(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path(".."))
    schema = TrainingPipelineParamsSchema()
    params = schema.load(cfg)
    optimaize_model_pipeline(params)


if __name__ == "__main__":
    opt_pipeline_start()