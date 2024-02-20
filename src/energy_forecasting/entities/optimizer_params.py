from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class OptimizerParams:
    model_name: str
    model_param_space: List[Dict]