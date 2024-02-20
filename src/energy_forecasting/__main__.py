#!/usr/bin/env python3
# __main__.py

import hydra
from omegaconf import DictConfig

from .predict_pipeline import predict_pipeline_start
from .train_pipeline import train_pipeline_start
from .model_params_optimizer import opt_pipeline_start


@hydra.main(version_base=None, config_path=".../configs", config_name="config")
def main(cfg: DictConfig):
    """Function that run the pipeline given a configuration."""
    cfg = hydra.utils.instantiate(cfg)

    pipeline_args = dict(
        pipeline=cfg.pipeline,
        config=cfg
    )
    if cfg.pipeline=="predict":
        predict_pipeline_start()
    elif cfg.pipeline=="train":
        train_pipeline_start()
    elif cfg.pipeline=="optimize":
        opt_pipeline_start()
    else:
        raise KeyError(
            f"Unknown pipeline, must equal to predict / train / optimize, got instead `{cfg.pipeline}`."
        )

if __name__ == "__main__":
    main()  # type: ignore