from dataclasses import dataclass

from marshmallow_dataclass import class_schema
from .feature_params import FeatureParams
from .predict_params import PredictParams

@dataclass()
class PredictingPipelineParams:
    predict_params: PredictParams
    feature_params: FeatureParams



PredictingPipelineParamsSchema = class_schema(PredictingPipelineParams)