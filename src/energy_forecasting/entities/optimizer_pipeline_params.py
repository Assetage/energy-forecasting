from dataclasses import dataclass
from typing import Any

from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams
from .path_params import PathParams
from .split_params import SplittingParams


@dataclass()
class OptimizerPipelineParams:
    path_config: PathParams
    splitting_params: SplittingParams
    feature_params: FeatureParams
    optimizer_params: dict[str, Any]


OptimizerPipelineParamsSchema = class_schema(OptimizerPipelineParams)
