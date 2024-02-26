from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

SklearnClassifierModel = Union[RandomForestRegressor, LinearRegression, MLPRegressor]


def make_prediction(
    df: pd.DataFrame, model: SklearnClassifierModel, features: dict[str, Any]
) -> np.ndarray:
    df = df.query("isFuture").copy()
    df = df.drop(columns=["isFuture", features.target_col])
    features_list = df.columns
    df.dropna(inplace=True)
    df["pred"] = model.predict(df[features_list])
    return df


def create_future_df(df_sorted: pd.DataFrame, features: dict[str, Any]) -> pd.DataFrame:
    dt_col = features.datetime_col
    max_date = pd.to_datetime(df_sorted[dt_col]).max()
    resulting_date = max_date + pd.DateOffset(years=1)
    future = pd.date_range(max_date, resulting_date, freq="1h")
    future_df = pd.DataFrame()
    future_df[dt_col] = future
    future_df["isFuture"] = True
    df_sorted["isFuture"] = False
    df_and_future = pd.concat([df_sorted, future_df])
    return df_and_future
