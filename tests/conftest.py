from typing import NoReturn, List, Tuple
from datetime import datetime

import pytest
import pandas as pd
from faker import Faker

from src.energy_forecasting.entities import (
    FeatureParams,
    PredictParams,
    PredictingPipelineParams,
)

ROW_NUMS = 20000


@pytest.fixture(scope="session")
def synthetic_data_path() -> str:
    return "tests/synthetic_data.csv"


@pytest.fixture(scope="session")
def output_predictions_path() -> str:
    return "tests/test_predictions.csv"


@pytest.fixture(scope="session")
def load_model_path() -> str:
    return "tests/test_model.pkl"


@pytest.fixture(scope="session")
def output_plot_path() -> str:
    return "tests/test_predictions_plot.png"


@pytest.fixture(scope="session")
def synthetic_data(synthetic_data_path: str) -> pd.DataFrame:
    fake = Faker()
    Faker.seed(21)
    df = {
        "Datetime": [
            fake.date_time_between_dates(
                date_start=datetime(2002, 1, 1), date_end=datetime(2018, 12, 31)
            )
            for _ in range(ROW_NUMS)
        ],
        "PJMW_MW": [
            fake.pyint(min_value=2000, max_value=8000) for _ in range(ROW_NUMS)
        ],
    }
    df.to_csv(synthetic_data_path, index=False)
    return pd.DataFrame(data=df)


@pytest.fixture(scope="session")
def datetime_col() -> str:
    return "Datetime"


@pytest.fixture(scope="session")
def target_col() -> str:
    return "PJMW_MW"


@pytest.fixture(scope="session")
def feature_params(datetime_col: str, target_col: str) -> FeatureParams:
    fp = FeatureParams(datetime_col=datetime_col, target_col=target_col)
    return fp


@pytest.fixture(scope="session")
def predict_params(
    synthetic_data_path: str,
    output_predictions_path: str,
    output_plot_path: str,
    load_model_path: str,
) -> PredictParams:
    pp = PredictParams(
        input_data_path=synthetic_data_path,
        output_data_path=output_predictions_path,
        output_plot_path=output_plot_path,
        model_path=load_model_path,
    )
    return pp


@pytest.fixture(scope="package")
def predict_pipeline_params(
    predict_params: PredictParams, feature_params: FeatureParams
) -> PredictingPipelineParams:
    pred_pipeline_params = PredictingPipelineParams(
        predict_params=predict_params, feature_params=feature_params
    )
    return pred_pipeline_params
