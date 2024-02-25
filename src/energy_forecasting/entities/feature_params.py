from dataclasses import dataclass

@dataclass()
class FeatureParams:
    datetime_col: str
    target_col: str