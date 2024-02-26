#!/usr/bin/env python3
# __main__.py

import os

import hydra
from omegaconf import DictConfig

from .optimizer_pipeline import opt_pipeline_start
from .predict_pipeline import predict_pipeline_start
from .train_pipeline import train_pipeline_start


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.path.dirname(__file__), "conf"),
    config_name="config",
)
def main(cfg: DictConfig):
    pipeline_name = cfg._group_.name
    cfg = cfg._group_
    del cfg.name
    if pipeline_name == "predict":
        del cfg.optimizer_params
        del cfg.train_params
        predict_pipeline_start(cfg)
    elif pipeline_name == "train":
        del cfg.optimizer_params
        train_pipeline_start(cfg)
    elif pipeline_name == "optimize":
        del cfg.train_params
        opt_pipeline_start(cfg)
    else:
        raise KeyError(
            "Unknown pipeline, must equal to predict / train / optimize, got instead"
            f" `{cfg.pipeline}`."
        )


if __name__ == "__main__":
    main()  # type: ignore
