import logging.config
from functools import partial

import marshmallow
from hyperopt import Trials, fmin, tpe
from omegaconf import DictConfig, OmegaConf

from .data import time_series_split
from .entities.optimizer_pipeline_params import (
    OptimizerPipelineParams,
    OptimizerPipelineParamsSchema,
)
from .features.build_features import prepare_dataset
from .models import define_space, objective, select_model
from .utils import read_data, save_optimization_results_to_json

logger = logging.getLogger("optimizer_pipeline")


def optimize_model_pipeline(
    optimizer_pipeline_params: OptimizerPipelineParams,
) -> None:
    """Executes the optimization pipeline for a given model using the provided
    configuration parameters.

    This function performs several steps:
    - Reads and logs the optimization parameters.
    - Loads and preprocesses the input data.
    - Splits the data into training and testing sets.
    - Initializes the model specified in the configuration.
    - Runs the hyperparameter optimization using the specified objective function
      and hyperparameter space.
    - Saves the best found hyperparameters to a specified JSON file.

    Parameters:
    - optimizer_pipeline_params (OptimizerPipelineParams): An object containing all the necessary
      parameters for the optimization pipeline, including data paths, model type, feature
      engineering parameters, and hyperparameter optimization settings.
    """
    logger.info(
        "Start optimizer pipeline with params"
        f" {OmegaConf.to_yaml(optimizer_pipeline_params)}"
    )
    model_type = optimizer_pipeline_params.optimizer_params["model_type"]
    logger.info(f"Model is {model_type}")

    data = read_data(optimizer_pipeline_params.path_config.input_data_path)

    logger.info("Building features...")
    data_transformed = prepare_dataset(data, optimizer_pipeline_params.feature_params)

    logger.info("Splitting data into train and test...")
    df_sorted, tss = time_series_split(
        data_transformed, optimizer_pipeline_params.splitting_params
    )

    logger.info("Model initialization...")
    model = select_model(optimizer_pipeline_params.optimizer_params)

    logger.info("Model optimization...")
    partial_objective = partial(
        objective, df_sorted, tss, optimizer_pipeline_params.feature_params, model
    )
    space = define_space(model_type, optimizer_pipeline_params.optimizer_params)

    trials = Trials()
    best_hyperparams = fmin(
        fn=partial_objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials
    )
    save_optimization_results_to_json(
        optimizer_pipeline_params.path_config.opt_results,
        model_type,
        optimizer_pipeline_params.optimizer_params,
        best_hyperparams,
    )
    logger.info("Best hyperparameters found:", best_hyperparams)
    logger.info("Pipeline is finished.")


def opt_pipeline_start(cfg: DictConfig):
    """Entry point for starting the model optimization pipeline.

    This function is responsible for loading and validating the optimization
    pipeline configuration using Marshmallow schemas. If the configuration is valid,
    it proceeds to run the optimization pipeline. Otherwise, it logs and raises
    a validation error.

    Parameters:
    - cfg (DictConfig): A configuration object loaded by OmegaConf, containing all
      necessary settings for the optimization pipeline.

    Raises:
    - marshmallow.exceptions.ValidationError: If the configuration fails to validate,
      indicating incorrect or missing parameters.
    """
    schema = OptimizerPipelineParamsSchema()
    try:
        params = schema.load(cfg)
        optimize_model_pipeline(params)
    except marshmallow.exceptions.ValidationError as e:
        logger.error(f"Configuration validation error: {e.messages}")
        raise e
