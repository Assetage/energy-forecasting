import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..entities import SplittingParams


def time_series_split(
    df: pd.DataFrame, params: SplittingParams
) -> tuple[pd.DataFrame, TimeSeriesSplit]:
    """Splits a time series dataframe into training and testing sets based on
    the provided splitting parameters.

    This function organizes the data frame for time series analysis by sorting it based on its index and prepares
    the TimeSeriesSplit object with the specified number of splits, test size, and gap between the training and test sets.

    Parameters:
    - df (pd.DataFrame): The input dataframe to be split, expected to have a datetime index.
    - params (SplittingParams): An object containing the parameters for the split. This includes
      `n_splits` (number of splits), `test_size` (size of the test set in terms of hours, days, and years),
      and `gap` (the gap between training and testing sets).

    Returns:
    - tuple[pd.DataFrame, TimeSeriesSplit]: A tuple containing the sorted dataframe and the TimeSeriesSplit object
      configured with the provided parameters. This object can then be used to generate train/test indices
      for cross-validation.

    Note:
    - The dataframe is expected to be sorted by its index to ensure the correct temporal order.
    - `test_size` is dynamically calculated based on the provided `hours`, `days`, and `years` in the `params`.
    """
    tss = TimeSeriesSplit(
        n_splits=params.n_splits,
        test_size=params.hours * params.days * params.years,
        gap=params.gap,
    )
    df = df.sort_index()
    return df, tss
