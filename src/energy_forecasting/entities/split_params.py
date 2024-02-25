from dataclasses import dataclass, field


@dataclass()
class SplittingParams:
    n_splits: int = field(default=5)
    hours: int = field(default=24)
    days: int = field(default=365)
    years: int = field(default=1)
    gap: int = field(default=24)