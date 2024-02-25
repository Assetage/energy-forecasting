#!/usr/bin/env python3
# __main__.py

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import os

from .predict_pipeline import predict_pipeline_start
from .train_pipeline import train_pipeline_start
from .model_params_optimizer import opt_pipeline_start


@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(__file__), "conf"), config_name="config")
def main(cfg: DictConfig):
    pipeline_name = cfg._group_.name
    cfg = cfg._group_
    del cfg.name
    if pipeline_name=="predict":
        predict_pipeline_start(cfg)
    elif pipeline_name=="train":
        train_pipeline_start(cfg)
    elif pipeline_name=="optimize":
        opt_pipeline_start(cfg)
    else:
        raise KeyError(
            f"Unknown pipeline, must equal to predict / train / optimize, got instead `{cfg.pipeline}`."
        )

if __name__ == "__main__":
    main()  # type: ignore