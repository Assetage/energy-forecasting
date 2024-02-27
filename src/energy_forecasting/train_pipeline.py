import logging.config

import marshmallow
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .data import time_series_split
from .entities.train_pipeline_params import (
    TrainingPipelineParams,
    TrainingPipelineParamsSchema,
)
from .features.build_features import prepare_dataset
from .models import run_cross_validation, select_model, train_model
from .utils import read_data, save_cross_val_results_to_json, save_pkl_file

logger = logging.getLogger("train_pipeline")


def train_pipeline(
    training_pipeline_params: TrainingPipelineParams,
) -> None:
    """Executes the training pipeline for machine learning models on time
    series data.

    This function handles the complete process of training a model, including:
    - Reading the dataset from the specified path.
    - Feature engineering based on the provided configurations.
    - Splitting the data into training and testing sets following time series considerations.
    - Selecting and initializing the model based on configuration.
    - Performing cross-validation to evaluate model performance.
    - Training the model on the entire dataset.
    - Saving the trained model for future predictions.

    Parameters:
    - training_pipeline_params (TrainingPipelineParams): An object containing all necessary
      parameters for the training pipeline, including paths for input data, model configuration,
      and output paths for the trained model and evaluation metrics.
    """
    logger.info(
        "Start train pipeline with params"
        f" {OmegaConf.to_yaml(training_pipeline_params)}"
    )
    model_type = training_pipeline_params.train_params["model_type"]
    logger.info(f"Model is {model_type}")

    data = read_data(training_pipeline_params.path_config.input_data_path)

    logger.info("Building features...")
    data_transformed = prepare_dataset(data, training_pipeline_params.feature_params)

    logger.info("Splitting data into train and test...")
    df_sorted, tss = time_series_split(
        data_transformed, training_pipeline_params.splitting_params
    )

    logger.info("Model initialization...")
    model = select_model(training_pipeline_params.train_params)

    logger.info("Running cross validation...")
    scores = run_cross_validation(
        df_sorted, tss, training_pipeline_params.feature_params, model
    )
    save_cross_val_results_to_json(
        training_pipeline_params.path_config.cross_val_scores,
        training_pipeline_params.train_params["model_type"],
        scores,
        model.get_params(),
    )

    print("Mean RMSE accross folds is: ", np.mean(scores))
    print("RMSE Standard deviation accross folds is: ", np.std(scores))
    print(
        "Full track of cross validation can be seen here: ",
        training_pipeline_params.path_config.cross_val_scores,
    )

    logger.info("Starting model training...")
    model = train_model(df_sorted, model, training_pipeline_params.feature_params)
    logger.info("Model training is done.")

    save_pkl_file(
        model,
        training_pipeline_params.path_config.output_model_path
        + training_pipeline_params.train_params["model_type"]
        + ".pkl",
    )
    logger.info("Model is saved.")
    logger.info("Pipeline is finished.")


def train_pipeline_start(cfg: DictConfig):
    """Initiates the training pipeline process.

    Validates the provided configuration using a schema. If the configuration passes validation,
    the training pipeline is executed. Validation errors are logged and raised, halting the
    process if the configuration is found to be invalid.

    Parameters:
    - cfg (DictConfig): A configuration object loaded by OmegaConf, containing all necessary
      settings for the training pipeline process.

    Raises:
    - marshmallow.exceptions.ValidationError: Indicates failure in configuration validation,
      pointing to incorrect or missing parameters.
    """
    schema = TrainingPipelineParamsSchema()
    try:
        params = schema.load(cfg)
        train_pipeline(params)
    except marshmallow.exceptions.ValidationError as e:
        logger.error(f"Configuration validation error: {e.messages}")
        raise e
