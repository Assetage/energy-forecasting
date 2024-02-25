from typing import Union, Dict, Any

import pandas as pd
import numpy as np
import datetime
import json
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from ..entities.feature_params import FeatureParams
from ..entities.path_params import PathParams

from ..entities.train_params import LogRegParams, RandomForestParams, MLPParams
from ..entities.optimizer_params import RandomForestOptParams, MLPOptParams

SklearnRegressorModel = Union[RandomForestRegressor, LinearRegression, MLPRegressor]


def select_model(train_params: Union[LogRegParams, RandomForestParams, MLPParams],
                ) -> SklearnRegressorModel:
    model_type = train_params["model_type"]

    if model_type == "RandomForestRegressor":
        if "max_depth_from" in train_params:
            train_params["max_depth"] = train_params["max_depth_from"]
             
        model = RandomForestRegressor(
            n_estimators=train_params["n_estimators"],
            max_depth=train_params["max_depth"],
            random_state=train_params["random_state"],
            verbose = train_params["verbose"]
        )

    elif model_type == "LinearRegression":
        model = LinearRegression()

    elif model_type == "MLPRegressor":
        if "hidden_layer_sizes_from" in train_params:
            train_params["hidden_layer_sizes"] = train_params["hidden_layer_sizes_from"]
        h_layers = tuple(int(i) for i in (train_params["hidden_layer_sizes"].split(',')))
        model = MLPRegressor(
            hidden_layer_sizes=h_layers,
            max_iter=train_params["max_iter"],
            random_state=train_params["random_state"],
            verbose=train_params["verbose"]
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

def objective(df_sorted: pd.DataFrame,
              tss: TimeSeriesSplit,
              params: FeatureParams,
              base_model: SklearnRegressorModel,
              hyperparams: Dict) -> Dict[str, Any]:
    model = clone(base_model)
    model.set_params(**hyperparams)
    scores = run_cross_validation(df_sorted, tss, params, model)
    mean_score = np.mean(scores)
    
    return {'loss': -mean_score, 'status': STATUS_OK}

def define_space(model_type: str,
                 optimizer_params: Union[ RandomForestOptParams, MLPOptParams]) -> Dict[str, Any]:
    print("PARAMS",optimizer_params )
    if model_type=="MLPRegressor":
        max_layers = 3
        space = {
        'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [
            tuple(int(size) for size in np.random.randint(
                int(optimizer_params["hidden_layer_sizes_from"]),
                int(optimizer_params["hidden_layer_sizes_to"]) + 1,
                max_layers))
            for _ in range(10)
        ]),
    }
    
    elif model_type=="RandomForestRegressor":
        space = {'max_depth': hp.quniform('max_depth', optimizer_params["max_depth_from"], 
                                                       optimizer_params["max_depth_to"], 1)}
    else:
        raise NotImplementedError("Model type not supported.")
    return space

    
    