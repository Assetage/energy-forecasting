from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class RandomForestOptParams:
    model_type: str
    n_estimator: int
    max_depth_from: int
    max_depth_to: int
    random_state: int
    verbose: int

@dataclass
class MLPOptParams:
    model_type: str
    hidden_layer_sizes_from: str
    hidden_layer_sizes_to: str
    max_iter: int
    random_state: int
    verbose: int