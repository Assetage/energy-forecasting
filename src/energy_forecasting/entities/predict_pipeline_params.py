from dataclasses import dataclass

from marshmallow_dataclass import class_schema
from .feature_params import FeatureParams

@dataclass()
class PredictingPipelineParams:
    input_data_path: str
    output_data_path: str
    pipeline_path: str
    model_path: str
    feature_params: FeatureParams



PredictingPipelineParamsSchema = class_schema(PredictingPipelineParams)