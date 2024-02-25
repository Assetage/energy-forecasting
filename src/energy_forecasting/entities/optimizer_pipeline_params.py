from typing import Dict, Any, Union

from dataclasses import dataclass
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .optimizer_params import RandomForestOptParams, MLPOptParams
from .path_params import PathParams
from marshmallow_dataclass import class_schema


@dataclass()
class OptimizerPipelineParams:
    path_config: PathParams
    splitting_params: SplittingParams
    feature_params: FeatureParams
    optimizer_params: Dict[str, Any]


OptimizerPipelineParamsSchema = class_schema(OptimizerPipelineParams)