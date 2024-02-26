from typing import Any, Union

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, hp
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

from ..entities.feature_params import FeatureParams
from ..entities.optimizer_params import MLPOptParams, RandomForestOptParams
from ..entities.train_params import LogRegParams, MLPParams, RandomForestParams

SklearnRegressorModel = Union[RandomForestRegressor, LinearRegression, MLPRegressor]


def select_model(
    train_params: Union[LogRegParams, RandomForestParams, MLPParams],
) -> SklearnRegressorModel:
    model_type = train_params["model_type"]

    if model_type == "RandomForestRegressor":
        if "max_depth_from" in train_params:
            train_params["max_depth"] = train_params["max_depth_from"]

        model = RandomForestRegressor(
            n_estimators=train_params["n_estimators"],
            max_depth=train_params["max_depth"],
            random_state=train_params["random_state"],
            verbose=train_params["verbose"],
        )

    elif model_type == "LinearRegression":
        model = LinearRegression()

    elif model_type == "MLPRegressor":
        if "hidden_layer_sizes_from" in train_params:
            train_params["hidden_layer_sizes"] = train_params["hidden_layer_sizes_from"]
        h_layers = tuple(
            int(i) for i in (train_params["hidden_layer_sizes"].split(","))
        )
        model = MLPRegressor(
            hidden_layer_sizes=h_layers,
            max_iter=train_params["max_iter"],
            random_state=train_params["random_state"],
            verbose=train_params["verbose"],
        )
    else:
        raise NotImplementedError()

    return model


def train_model(
    df_sorted: pd.DataFrame,
    model: SklearnRegressorModel,
    params: FeatureParams,
) -> SklearnRegressorModel:
    target = params.target_col
    features = df_sorted.drop(columns=[target]).columns
    x_all = df_sorted[features]
    y_all = df_sorted[target]
    model.fit(x_all, y_all)
    return model


def run_cross_validation(
    df_sorted: pd.DataFrame,
    tss: TimeSeriesSplit,
    params: FeatureParams,
    model: SklearnRegressorModel,
) -> list:
    preds = []
    scores = []
    for train_idx, val_idx in tss.split(df_sorted):
        train = df_sorted.iloc[train_idx]
        test = df_sorted.iloc[val_idx]

        target = params.target_col
        features = df_sorted.drop(columns=[target]).columns

        x_train = train[features]
        y_train = train[target]

        x_test = test[features]
        y_test = test[target]

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        preds.append(y_pred)
        score = mean_squared_error(y_test, y_pred, squared=True)
        scores.append(score)

    return scores


def objective(
    df_sorted: pd.DataFrame,
    tss: TimeSeriesSplit,
    params: FeatureParams,
    base_model: SklearnRegressorModel,
    hyperparams: dict,
) -> dict[str, Any]:
    model = clone(base_model)
    model.set_params(**hyperparams)
    scores = run_cross_validation(df_sorted, tss, params, model)
    mean_score = np.mean(scores)

    return {"loss": -mean_score, "status": STATUS_OK}


def define_space(
    model_type: str, optimizer_params: Union[RandomForestOptParams, MLPOptParams]
) -> dict[str, Any]:
    if model_type == "MLPRegressor":
        max_layers = 3
        space = {
            "hidden_layer_sizes": hp.choice(
                "hidden_layer_sizes",
                [
                    tuple(
                        int(size)
                        for size in np.random.randint(
                            int(optimizer_params["hidden_layer_sizes_from"]),
                            int(optimizer_params["hidden_layer_sizes_to"]) + 1,
                            max_layers,
                        )
                    )
                    for _ in range(3)
                ],
            ),
        }

    elif model_type == "RandomForestRegressor":
        space = {
            "max_depth": hp.quniform(
                "max_depth",
                optimizer_params["max_depth_from"],
                optimizer_params["max_depth_to"],
                1,
            )
        }
    else:
        raise NotImplementedError("Model type not supported.")
    return space
