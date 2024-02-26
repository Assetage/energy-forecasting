from typing import NoReturn, List, Tuple
from datetime import datetime

import pytest
import pandas as pd
from faker import Faker

ROW_NUMS = 200


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
def metric_path() -> str:
    return "tests/test_metrics.json"

@pytest.fixture(scope="session")
def synthetic_data() -> pd.DataFrame:
    fake = Faker()
    Faker.seed(21)
    df = {
        "Datetime": [fake.date_time_between_dates(date_start=datetime(2002,1,1), date_end=datetime(2018,12,31)) for _ in range(ROW_NUMS)],
        "PJMW_MW": [fake.pyint(min_value=2000, max_value=8000) for _ in range(ROW_NUMS)]
    }

    return pd.DataFrame(data=df)