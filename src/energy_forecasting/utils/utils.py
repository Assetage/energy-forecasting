import json
import pickle
from typing import NoReturn
import os

import pandas as pd
import numpy as np
import datetime


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

def save_cross_val_results_to_json(file_path: str, model_type: str, scores: dict, params_dict: dict) -> NoReturn:
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
        with open(file_path,'r') as f:
            list_obj = json.load(f)
    list_obj.append(scores_dict)

    with open(file_path,'w') as f:
        json.dump(list_obj, f, indent = 4, sort_keys = True, separators=(',',': '))

def save_optimization_results_to_json(file_path: str, model_type: str, params_dict: dict, tuned_params: dict) -> NoReturn:
    list_obj = []
    opt_params_dict = dict()
    timestamp = str(datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
    opt_params_dict["model_type"] = model_type
    opt_params_dict["timestamp"] = timestamp
    opt_params_dict["model_params"] = params_dict
    opt_params_dict["tuned_params"] = tuned_params

    if os.path.isfile(file_path) is True:
        with open(file_path,'r') as f:
            list_obj = json.load(f)
    list_obj.append(opt_params_dict)

    with open(file_path,'w') as f:
        json.dump(list_obj, f, indent = 4, sort_keys = True, separators=(',',': '))

def save_pkl_file(input_file, output_name: str) -> NoReturn:
    with open(output_name, "wb") as f:
        pickle.dump(input_file, f)

def load_pkl_file(input_: str):
    with open(input_, "rb") as fin:
        res = pickle.load(fin)
    return res

