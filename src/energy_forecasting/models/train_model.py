from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.entities.train_params import LogRegParams, RandomForestParams, MLPParams

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression, MLPClassifier]


def train_model(features: pd.DataFrame,
                target: pd.Series,
                train_params: Union[LogRegParams, RandomForestParams, MLPParams],
                ) -> SklearnClassifierModel:

    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators,
            max_depth=train_params.max_depth,
            random_state=train_params.random_state,
        )

    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            penalty=train_params.penalty,
            tol=train_params.tol,
            random_state=train_params.random_state,
        )

    elif train_params.model_type == "MLPClassifier":
        h_layers = tuple(int(i) for i in (train_params.hidden_layer_sizes.split(',')))
        model = MLPClassifier(
            hidden_layer_sizes=h_layers,
            max_iter=train_params.max_iter,
            random_state=train_params.random_state
        )
    else:
        raise NotImplementedError()

    model.fit(features, target)
    return model