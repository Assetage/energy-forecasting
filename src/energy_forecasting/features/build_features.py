import pandas as pd

from ..entities.feature_params import FeatureParams


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time series features based on the dataframe's time series index.

    Parameters:
    - df (pd.DataFrame): The input dataframe with a datetime index.

    Returns:
    - pd.DataFrame: The modified dataframe with new time series features added.

    The function adds the following features to the dataframe:
    - 'hour': The hour of the datetime index.
    - 'dayofweek': The day of the week of the datetime index.
    - 'quarter': The quarter of the datetime index.
    - 'month': The month of the datetime index.
    - 'year': The year of the datetime index.
    - 'dayofyear': The day of the year of the datetime index.
    - 'dayofmonth': The day of the month of the datetime index.
    - 'weekofyear': The week of the year of the datetime index.
    """
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    df["dayofmonth"] = df.index.day
    df["weekofyear"] = df.index.isocalendar().week
    return df


def add_lags(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    """Add lagged values of the target column to the dataframe as new features.

    Parameters:
    - df (pd.DataFrame): The input dataframe with a datetime index.
    - params (FeatureParams): Parameters including the target column name.

    Returns:
    - pd.DataFrame: The dataframe with lagged features added.

    Adds the following lagged features based on the target column specified in `params`:
    - 'lag1': The value of the target column from 364 days ago.
    - 'lag2': The value of the target column from 728 days ago.
    - 'lag3': The value of the target column from 1092 days ago.
    """
    target_map = df[params.target_col].to_dict()
    df["lag1"] = (df.index - pd.Timedelta("364 days")).map(target_map)
    df["lag2"] = (df.index - pd.Timedelta("728 days")).map(target_map)
    df["lag3"] = (df.index - pd.Timedelta("1092 days")).map(target_map)
    return df


def prepare_dataset(
    df: pd.DataFrame, params: FeatureParams, dropna_bool=True
) -> pd.DataFrame:
    """Prepare the dataset for modeling by setting the index, adding time
    series features, lagged features, and optionally dropping NA values.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - params (FeatureParams): Parameters for feature creation, including the target column and datetime column names.
    - dropna_bool (bool): If True, drop rows with NA values; default is True.

    Returns:
    - pd.DataFrame: The prepared dataframe ready for modeling.

    The function performs the following operations:
    - Sets the dataframe index to the datetime column specified in `params`.
    - Converts the index to datetime format.
    - Replaces specific values in the target column as defined (e.g., replacing 487 with 4870).
    - Adds time series and lagged features using `create_features` and `add_lags`.
    - Drops NA values if `dropna_bool` is True.
    """
    df = df.set_index(params.datetime_col)
    df.index = pd.to_datetime(df.index)
    df[params.target_col] = df[params.target_col].replace(487, 4870)
    df = create_features(df)
    df = add_lags(df, params)
    if dropna_bool:
        df.dropna(inplace=True)

    return df
