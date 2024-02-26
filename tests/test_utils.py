import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.energy_forecasting.utils import read_data, load_pkl_file


def test_read_data(synthetic_data_path: str):
    df = read_data(synthetic_data_path)

    assert isinstance(df, pd.DataFrame)
    assert (200, 2) == df.shape


def test_load_pkl_file(load_model_path: str):
    model = load_pkl_file(load_model_path)
    assert isinstance(model, RandomForestRegressor) or isinstance(
        model, LinearRegression
    )
