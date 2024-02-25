from typing import Union

import pandas as pd
import numpy as np
import datetime
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from ..entities.feature_params import FeatureParams
from ..entities.path_params import PathParams

from ..entities.train_params import LogRegParams, RandomForestParams, MLPParams

SklearnRegressorModel = Union[RandomForestRegressor, LinearRegression, MLPRegressor]


def select_model(train_params: Union[LogRegParams, RandomForestParams, MLPParams],
                ) -> SklearnRegressorModel:
    model_type = train_params["model_type"]
    if model_type == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=train_params["n_estimators"],
            max_depth=train_params["max_depth"],
            random_state=train_params["random_state"],
            verbose = train_params["verbose"]
        )

    elif model_type == "LinearRegression":
        model = LinearRegression(
            penalty=train_params["penalty"],
            tol=train_params["tol"],
            random_state=train_params["random_state"],
        )

    elif model_type == "MLPRegressor":
        h_layers = tuple(int(i) for i in (train_params.hidden_layer_sizes.split(',')))
        model = MLPRegressor(
            hidden_layer_sizes=h_layers,
            max_iter=train_params["max_iter"],
            random_state=train_params["random_state"]
        )
    else:
        raise NotImplementedError()
    
    return model


def train_model(df_sorted: pd.DataFrame,
                model: SklearnRegressorModel,
                params: FeatureParams,
                ) -> SklearnRegressorModel:
    TARGET = params.target_col
    FEATURES = df_sorted.drop(columns=[TARGET]).columns
    X_all = df_sorted[FEATURES]
    y_all = df_sorted[TARGET]
    model.fit(X_all, y_all)
    return model

def run_cross_validation(df_sorted: pd.DataFrame,
                         tss: TimeSeriesSplit,
                         params: FeatureParams,
                         model: SklearnRegressorModel) -> list:
    preds = []
    scores = []
    for train_idx, val_idx in tss.split(df_sorted):
        train = df_sorted.iloc[train_idx]
        test = df_sorted.iloc[val_idx]

        TARGET = params.target_col
        FEATURES = df_sorted.drop(columns=[TARGET]).columns

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        preds.append(y_pred)
        score = mean_squared_error(y_test, y_pred, squared=True)
        scores.append(score)
    
    return scores
    
    