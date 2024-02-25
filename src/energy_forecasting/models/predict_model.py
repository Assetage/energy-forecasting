from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


SklearnClassifierModel = Union[RandomForestRegressor, LinearRegression, MLPRegressor]


def make_prediction(model: SklearnClassifierModel, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts