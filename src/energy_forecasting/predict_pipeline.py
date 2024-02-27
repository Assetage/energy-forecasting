import logging.config

import marshmallow
from omegaconf import DictConfig

from .entities.predict_pipeline_params import (
    PredictingPipelineParams,
    PredictingPipelineParamsSchema,
)
from .features.build_features import prepare_dataset
from .models import create_future_df, make_prediction
from .utils import load_pkl_file, read_data, save_pred_plot

logger = logging.getLogger("ml_project/predict_pipeline")


def predict_pipeline(predict_pipeline_params: PredictingPipelineParams) -> None:
    """Executes the prediction pipeline for a given dataset using a pre-trained
    model.

    Steps involved:
    - Reads the dataset specified in the prediction pipeline parameters.
    - Loads the pre-trained model from the path specified in the parameters.
    - Creates a DataFrame for future predictions and concatenates it with the existing data.
    - Prepares the dataset by building features as specified in the parameters.
    - Makes predictions on the prepared dataset using the loaded model.
    - Saves the predictions to a CSV file and a plot visualizing the predictions.

    Parameters:
    - predict_pipeline_params (PredictingPipelineParams): An object containing all the necessary
      parameters for the prediction pipeline, including paths for input data, the model, and
      output for predictions and plot.
    """
    logger.info("Start prediction pipeline")
    data = read_data(predict_pipeline_params.predict_params.input_data_path)
    logger.info(f"Dataset shape is {data.shape}")

    logger.info("Loading model...")
    model = load_pkl_file(predict_pipeline_params.predict_params.model_path)

    logger.info("Creating future df and concating it with an existing...")
    df_and_future = create_future_df(data, predict_pipeline_params.feature_params)

    logger.info("Building features...")
    data_transformed = prepare_dataset(
        df_and_future, predict_pipeline_params.feature_params, dropna_bool=False
    )
    df_sorted = data_transformed.sort_index()

    logger.info("Start prediction")
    df_w_predictions = make_prediction(
        df_sorted, model, predict_pipeline_params.feature_params
    )

    df_w_predictions.to_csv(
        predict_pipeline_params.predict_params.output_data_path, header=True
    )
    logger.info(
        "Prediction is done and saved to the file"
        f" {predict_pipeline_params.predict_params.output_data_path}"
    )

    logger.info("Saving predictions plot")
    save_pred_plot(
        df_w_predictions, predict_pipeline_params.predict_params.output_plot_path
    )


def predict_pipeline_start(cfg: DictConfig):
    """Entry point for initiating the prediction pipeline.

    Loads and validates the configuration for the prediction pipeline using a schema. If the
    configuration is valid, it proceeds to run the prediction pipeline. If there is a validation
    error, it logs and raises the exception.

    Parameters:
    - cfg (DictConfig): A configuration object loaded by OmegaConf, containing all necessary
      settings for the prediction pipeline.

    Raises:
    - marshmallow.exceptions.ValidationError: Indicates a failure in configuration validation,
      suggesting incorrect or missing parameters.
    """
    schema = PredictingPipelineParamsSchema()
    try:
        params = schema.load(cfg)
        predict_pipeline(params)
    except marshmallow.exceptions.ValidationError as e:
        logger.error(f"Configuration validation error: {e.messages}")
        raise e
