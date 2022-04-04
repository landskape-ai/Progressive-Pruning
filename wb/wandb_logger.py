"""
Utilities for Weights & Biases logging. 
"""

from pathlib import Path
from typing import Union

import PIL
from matplotlib.pyplot import Figure
from PIL.Image import Image
from torch import Tensor

__all__ = ["WandBLogger"]


class WandBLogger:
    """
    The `WandBLogger` provides an easy integration with
    Weights & Biases logging. Each monitored metric is automatically
    logged to a dedicated Weights & Biases project dashboard.

    .. note::
    The wandb log files are placed by default in "./wandb/" unless specified.
    """

    def __init__(
        self,
        project_name: str = "APP",
        run_name: str = "Prune1",
        save_code: bool = True,
        config: object = None,
        dir: Union[str, Path] = None,
        model: object = None,
        params: dict = None,
    ) -> None:
        """
        Creates an instance of the `WandBLogger`.
        :param project_name: Name of the W&B project.
        :param run_name: Name of the W&B run.
        :param save_code: Saves the main training script to W&B.
        :param dir: Path to the local log directory for W&B logs to be saved at.
        :param config: Syncs hyper-parameters and config values used to W&B.
        :param params: All arguments for wandb.init() function call.
        Visit https://docs.wandb.ai/ref/python/init to learn about all
        wand.init() parameters.
        """

        self.project_name = project_name
        self.run_name = run_name
        self.save_code = save_code
        self.dir = dir
        self.config = config
        self.model = model
        self.params = params

        self._import_wandb()
        self._args_parse()
        self._before_job()

    def _import_wandb(self):
        try:
            import wandb

            assert hasattr(wandb, "__version__")
        except (ImportError, AssertionError):
            raise ImportError('Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def _args_parse(self):
        self.init_kwargs = {
            "project": self.project_name,
            "name": self.run_name,
            "save_code": self.save_code,
            "dir": self.dir,
            "config": self.config,
        }
        if self.params:
            self.init_kwargs.update(self.params)

    def _before_job(self):
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()
        if self.model is not None:
            self.wandb.watch(self.model)

    def log_metrics(
        self,
        log_dict: dict = None,
        img: Union[Image, Figure, str, Path] = None,
        curve: object = None,
    ) -> None:
        for key, value in log_dict.items():

            if isinstance(value, (int, float, Tensor)):
                self.wandb.log({key: value})
            else:
                if "ARE" in key:
                    curr_val = value
                else:
                    curr_val = value[-1]

                if isinstance(curr_val, (int, float, Tensor)):
                    if "train" in key.lower():
                        key = "Train/" + key
                        self.wandb.log({key: curr_val})
                    if "val" in key.lower():
                        key = "Val/" + key
                        self.wandb.log({key: curr_val})
                    if "test" in key.lower():
                        key = "Test/" + key
                        self.wandb.log({key: curr_val})
                    self.wandb.log({key: curr_val})

                else:
                    return

            if img is not None:
                if isinstance(img, (Image, Figure)):
                    self.wandb.log({"Media/Training Curve": self.wandb.Image(img)})

                if isinstance(img, (str, Path)):
                    img_pil = PIL.Image.open(img)
                    self.wandb.log({"Media/Training Curve": self.wandb.Image(img_pil)})

            if curve is not None:
                if isinstance(curve, (object)):
                    self.wandb.log({"Training Curves": curve})
