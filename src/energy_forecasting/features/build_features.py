import pandas as pd
from ..entities.feature_params import FeatureParams


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time series features based on time series index.
    """
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

def add_lags(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    target_map = df[params.target_col].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df


def prepare_dataset(df: pd.DataFrame, params: FeatureParams, dropna_bool = True) -> pd.DataFrame:

    df = df.set_index(params.datetime_col)
    df.index = pd.to_datetime(df.index)
    df[params.target_col] = df[params.target_col].replace(487, 4870)
    df = create_features(df)
    df = add_lags(df, params)
    if dropna_bool:
        df.dropna(inplace=True)

    return df