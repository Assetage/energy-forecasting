from dataclasses import dataclass, field


@dataclass()
class LogRegParams:
    model_type: str = field(default="LinearRegression")
    penalty: str = field(default="l2")
    tol: float = field(default=1e-4)
    random_state: int = field(default=21)


@dataclass()
class RandomForestParams:
    model_type: str = field(default="RandomForestRegressor")
    n_estimators: int = field(default=50)
    max_depth: int = field(default=5)
    random_state: int = field(default=21)
    verbose: int = field(default=1)


@dataclass()
class MLPParams:
    model_type: str = field(default="MLPRegressor")
    hidden_layer_sizes: str = field(default="128")
    max_iter: int = field(default=300)
    random_state: int = field(default=21)