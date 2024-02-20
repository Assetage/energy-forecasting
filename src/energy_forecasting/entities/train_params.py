from dataclasses import dataclass, field


@dataclass()
class LogRegParams:
    model_type: str = field(default="LogisticRegression")
    penalty: str = field(default="l2")
    tol: float = field(default=1e-4)
    random_state: int = field(default=21)


@dataclass()
class RandomForestParams:
    model_type: str = field(default="RandomForestClassifier")
    n_estimators: int = field(default=50)
    max_depth: int = field(default=5)
    random_state: int = field(default=21)


@dataclass()
class MLPParams:
    model_type: str = field(default="MLPClassifier")
    hidden_layer_sizes: str = field(default="128")
    max_iter: int = field(default=300)
    random_state: int = field(default=21)