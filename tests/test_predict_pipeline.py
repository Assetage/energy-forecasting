import os

from src.energy_forecasting.entities import PredictingPipelineParams
from src.energy_forecasting.predict_pipeline import predict_pipeline


def test_predict_pipeline(predict_pipeline_params: PredictingPipelineParams,
                       output_predictions_path: str,
                       output_plot_path):

    predict_pipeline(predict_pipeline_params)

    assert os.path.exists(output_predictions_path)
    assert os.path.exists(output_plot_path)