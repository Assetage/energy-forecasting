from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression, MLPClassifier]


def make_prediction(model: SklearnClassifierModel, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts