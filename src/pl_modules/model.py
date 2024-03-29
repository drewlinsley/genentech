from typing import Any, Dict, List, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torch.optim import Optimizer

import numpy as np
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients, GuidedGradCam
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from src.common.utils import iterate_elements_in_batches, render_images

from src.pl_modules import resnets
from src.pl_modules import losses

# from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay


class MyModel(pl.LightningModule):
    def __init__(
            self,
            cfg: DictConfig,
            name,
            num_classes,
            final_nl,
            loss,
            self_supervised=False,
            num_samples=False,
            batch_size=False,
            task="binary",
            *args,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.name = name
        self.self_supervised = self_supervised
        self.validation_step_outputs = []
        self.test_step_outputs = []
        # self.automatic_optimization = False
        self.num_classes = num_classes
        self.loss = getattr(losses, loss)  # Add this to the config
        # if final_nl:
        #     self.final_nl = getattr(F, final_nl)
        # else:
        #     self.final_nl = lambda x, dim: x
        self.final_nl = lambda x: torch.argmax(x, 1).float()
        if self.name == "resnet18":
            self.net = resnets.resnet18(pretrained=True, num_classes=num_classes)
        elif self.name == "simclr_resnet18":
            self.net = resnets.simclr_resnet18(
                pretrained=False,
                num_classes=num_classes,
                num_samples=num_samples,
                batch_size=batch_size)
        elif self.name == "simclr_resnet18_transfer":
            self.net = resnets.simclr_resnet18_transfer(
                pretrained=False,
                num_classes=num_classes,
                num_samples=num_samples,
                batch_size=batch_size)
        else:
            raise NotImplementedError("Could not find network {}.".format(self.net))

        metric = torchmetrics.Accuracy(task)
        self.train_accuracy = metric.clone().cuda()
        self.val_accuracy = metric.clone().cuda()
        self.test_accuracy = metric.clone().cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def step(self, x, y) -> Dict[str, torch.Tensor]:
        if self.self_supervised:
            z1, z2 = self.net.shared_step(x)
            logits = z1
            loss = self.loss(z1, z2)
        else:
            logits = self(x)
            if logits.shape[-1] > 1:
                loss = self.loss(logits, y)
            else:
                logits = logits.ravel()
                loss = self.loss(logits, y)
        return {"logits": logits, "loss": loss, "y": y, "x": x}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        out = self.step(x, y)
        # opt = self.optimizers()
        # opt.zero_grad()
        # self.manual_backward(out["loss"])
        # opt.step()
        return out

    def training_step_end(self, out):
        self.train_accuracy(self.final_nl(out["logits"]), out["y"])
        self.log_dict(
            {
                "train_acc": self.train_accuracy,
                "train_loss": out["loss"].mean(),
            },
            on_step=True,
            on_epoch=False
        )
        return out["loss"].mean()

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        out = self.step(x, y)
        self.validation_step_outputs.append(out)
        return out
    
    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        out = self.step(x, y)
        self.test_step_outputs.append(out)
        return out

    def test_step_end(self, out):
        self.test_accuracy(self.final_nl(out["logits"]), out["y"])
        self.log_dict(
            {
                "test_acc": self.test_accuracy,
                "test_loss": out["loss"].mean(),
            },
        )
        return {
            "image": out["x"],
            "y_true": out["y"],
            "logits": out["logits"],
            "val_loss": out["loss"].mean(),
        }

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        if hasattr(self.cfg.optim, "exclude_bn_bias") and \
                self.cfg.optim.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.cfg.optim.optimizer.weight_decay)
            print("Warning: Excluding-biases-from-weight-decay is not properly implemented yet.")
            params = self.parameters()
        else:
            params = self.parameters()

        opt = hydra.utils.instantiate(
            self.cfg.optim.optimizer, params=params, weight_decay=self.cfg.optim.optimizer.weight_decay
        )
        
        if not self.cfg.optim.use_lr_scheduler:
            return opt

        # Handle schedulers if requested
        if 0:  # Need to fix this. self.cfg.optim.lr_scheduler.warmup_steps:
            # Right now this is specific to SimCLR
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    opt,
                    linear_warmup_decay(
                        self.cfg.optim.lr_scheduler.warmup_steps,
                        self.cfg.optim.lr_scheduler.total_steps,
                        cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            lr_scheduler = self.cfg.optim.lr_scheduler
        scheduler = hydra.utils.instantiate(lr_scheduler, optimizer=opt)
        return [opt], [scheduler]

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]
