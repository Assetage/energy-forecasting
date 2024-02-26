from dataclasses import dataclass


@dataclass
class PredictParams:
    input_data_path: str
    output_data_path: str
    output_plot_path: str
    model_path: str
