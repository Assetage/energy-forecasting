#!/usr/bin/env python3
# __main__.py
"""Main entry point for the application to execute different pipelines
including prediction, training, and optimization based on the provided
configuration.

This script uses Hydra for configuration management, allowing for flexible
and dynamic command-line interfaces. Configuration files are organized in a
directory structure and specify parameters for each pipeline. Depending on
the command-line arguments, this script determines which pipeline to run and
configures it accordingly.

Usage:
To run this script, use the command line to specify the pipeline and its
configuration. For example:
`python __main__.py predict --config-name my_prediction_config`
`python __main__.py train --config-name my_training_config`
`python __main__.py optimize --config-name my_optimization_config`

The configurations are loaded from the specified `config_path` and `config_name`,
and the appropriate pipeline function is called with the loaded configuration.

Dependencies:
- hydra-core: For dynamic configuration management.
- omegaconf: For configuration data structure management.
"""

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
    """Main function to run specified pipeline based on the configuration.

    Parameters:
    - cfg (DictConfig): The configuration object provided by Hydra based on the command-line arguments
                        and the configuration files.

    The function determines the pipeline to execute (predict, train, optimize) based on the
    configuration, adjusts the configuration object accordingly, and starts the selected pipeline.

    Raises:
    - KeyError: If the specified pipeline name is not recognized.
    """
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
    main()
