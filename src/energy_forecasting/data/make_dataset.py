import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..entities import SplittingParams


def time_series_split(
    df: pd.DataFrame, params: SplittingParams
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tss = TimeSeriesSplit(
        n_splits=params.n_splits,
        test_size=params.hours * params.days * params.years,
        gap=params.gap,
    )
    df = df.sort_index()
    return df, tss
