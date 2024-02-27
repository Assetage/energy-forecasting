from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

SklearnRegressorModel = Union[RandomForestRegressor, LinearRegression, MLPRegressor]


def make_prediction(
    df: pd.DataFrame, model: SklearnRegressorModel, features: dict[str, Any]
) -> np.ndarray:
    """Make predictions using a specified scikit-learn regression model on the
    future data points of the dataframe.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing both historical and future data points.
    - model (SklearnRegressorModel): The trained regression model for making predictions.
    - features (dict[str, Any]): A dictionary containing feature configuration, including the target column name.

    Returns:
    - np.ndarray: An array of predictions made by the model.

    The function filters the dataframe for future data points, prepares the feature matrix by dropping non-feature
    columns and rows with NA values, and then uses the provided regression model to make predictions on the future data.
    """
    df = df.query("isFuture").copy()
    df = df.drop(columns=["isFuture", features.target_col])
    features_list = df.columns
    df.dropna(inplace=True)
    df["pred"] = model.predict(df[features_list])
    return df


def create_future_df(df_sorted: pd.DataFrame, features: dict[str, Any]) -> pd.DataFrame:
    """Create a future dataframe with a specified range and frequency, marking
    these rows as future data points.

    Parameters:
    - df_sorted (pd.DataFrame): The input dataframe sorted by datetime.
    - features (dict[str, Any]): A dictionary containing feature configuration, including the datetime column name.

    Returns:
    - pd.DataFrame: The dataframe concatenated with the future data points.

    The function identifies the maximum date in the specified datetime column, creates a new dataframe with a range
    of future dates extending one year from the maximum date at an hourly frequency, and marks these new rows as
    future data points. It then concatenates this future dataframe with the original sorted dataframe.
    """
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
