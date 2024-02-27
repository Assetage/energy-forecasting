import datetime
import json
import os
import pickle
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


def read_data(path: str) -> pd.DataFrame:
    """Read data from a CSV file into a pandas DataFrame.

    Parameters:
    - path (str): The file path to the CSV file to be read.

    Returns:
    - pd.DataFrame: The data from the CSV file as a pandas DataFrame.
    """
    data = pd.read_csv(path)
    return data


def save_cross_val_results_to_json(
    file_path: str, model_type: str, scores: dict, params_dict: dict
) -> None:
    """Save cross-validation results along with model parameters to a JSON
    file.

    Parameters:
    - file_path (str): Path to the JSON file where the results will be saved.
    - model_type (str): Type of the model used for cross-validation.
    - scores (dict): Dictionary containing the scores from cross-validation.
    - params_dict (dict): Dictionary containing the parameters of the model.

    Returns:
    - None
    """
    list_obj = []
    scores_dict = dict()
    timestamp = str(datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
    scores_dict["model_type"] = model_type
    scores_dict["timestamp"] = timestamp
    scores_dict["n_folds"] = len(scores)
    scores_dict["mean_rmse_folds"] = np.mean(scores)
    scores_dict["std_folds"] = np.std(scores)
    scores_dict["scores"] = scores
    scores_dict["model_params"] = params_dict

    if os.path.isfile(file_path) is True:
        with open(file_path) as f:
            list_obj = json.load(f)
    list_obj.append(scores_dict)

    with open(file_path, "w") as f:
        json.dump(list_obj, f, indent=4, sort_keys=True, separators=(",", ": "))


def save_optimization_results_to_json(
    file_path: str, model_type: str, params_dict: dict, tuned_params: dict
) -> None:
    """Save hyperparameter optimization results to a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file where the optimization results will be saved.
    - model_type (str): Type of the model that was optimized.
    - params_dict (dict): Dictionary containing the initial parameters used for optimization.
    - tuned_params (dict): Dictionary containing the optimized parameters.

    Returns:
    - None
    """
    list_obj = []
    opt_params_dict = dict()
    timestamp = str(datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
    opt_params_dict["model_type"] = model_type
    opt_params_dict["timestamp"] = timestamp
    opt_params_dict["model_params"] = params_dict
    opt_params_dict["tuned_params"] = tuned_params

    if os.path.isfile(file_path) is True:
        with open(file_path) as f:
            list_obj = json.load(f)
    list_obj.append(opt_params_dict)

    with open(file_path, "w") as f:
        json.dump(list_obj, f, indent=4, sort_keys=True, separators=(",", ": "))


def save_pkl_file(input_file, output_name: str) -> None:
    """Save an object to a file using pickle.

    Parameters:
    - input_file: The Python object to be saved.
    - output_name (str): The name of the file where the object will be saved.

    Returns:
    - None
    """
    with open(output_name, "wb") as f:
        pickle.dump(input_file, f)


def load_pkl_file(
    input_: str,
) -> Union[RandomForestRegressor, LinearRegression, MLPRegressor]:
    """Load a Python object from a pickle file.

    Parameters:
    - input_ (str): The path to the pickle file to be loaded.

    Returns:
    - The sklearn regression model most of the times.
    """
    with open(input_, "rb") as fin:
        res = pickle.load(fin)
    return res


def save_pred_plot(df: pd.DataFrame, output_path: str) -> None:
    """Save a plot of predictions to a file.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the predictions to be plotted.
    - output_path (str): Path to the file where the plot will be saved.

    Returns:
    - None
    """
    ax = df["pred"].plot(figsize=(10, 5), ms=1, lw=1, title="Future Predictions")
    fig = ax.get_figure()
    fig.savefig(output_path)
