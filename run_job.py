config_path = 'conf/'
conf = "kaggle_t1_train_t2_test_resnet18.yaml"
load_ckpt = False

import logging
import os
import shutil
import pathlib
from pathlib import Path
from typing import List

import hydra
import omegaconf
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pytorch_lightning as pl
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning import Callback, seed_everything
#from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar
)
from pytorch_lightning.loggers import WandbLogger

from src.common.utils import load_envs, weights_update
import torch
torch.backends.cudnn.benchmark = True
# # Set the cwd to the project root
# os.chdir(Path(__file__).parent.parent)

# Load environment variables
load_envs()


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info(f"Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.train:
        hydra.utils.log.info(f"Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.train.model_checkpoints:
        hydra.utils.log.info(f"Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
            )
        )

    callbacks.append(
        TQDMProgressBar(
            refresh_rate=cfg.logging.progress_bar_refresh_rate
            )
        )
    return callbacks

# Hydra run directory
with initialize(version_base=None, config_path=config_path):
    cfg = compose(conf)
    print(cfg)
hydra_dir = os.getcwd()

# Instantiate datamodule
hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
datamodule: pl.LightningDataModule = hydra.utils.instantiate(
    cfg.data.datamodule, cfg=cfg, _recursive_=False
)


# Instantiate model
hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
model: pl.LightningModule = hydra.utils.instantiate(cfg.model, cfg=cfg, _recursive_=False)

# Instantiate the callbacks
# callbacks: List[Callback] = build_callbacks(cfg=cfg)

if load_ckpt:
    print("Loading checkpoints from ", load_ckpt)
    model = weights_update(
        model=model,
        checkpoint=torch.load(cfg.train.ckpt))
else:
    print("Beginning with random weights.")

# if cfg.train.eval_only:
#     trainer = pl.Trainer(
#         logger=False,
#         default_root_dir=hydra_dir,  # Path('./experiments/train'),
#         val_check_interval=cfg.logging.val_check_interval,
#         log_every_n_steps=10,
#         **cfg.train,
#     )

#     hydra.utils.log.info(f"EVAL ONLY SELECTED. Starting testing!")
#     trainer.test(model=model, datamodule=datamodule)
#     sys.exit()

# Logger instantiation/configuration
wandb_logger = None
if "wandb" in cfg.logging:
    hydra.utils.log.info(f"Instantiating <WandbLogger>")
    wandb_config = cfg.logging.wandb
    wandb_logger = WandbLogger(
        name="{}_{}_{}".format(
            cfg.core.experiment,
            cfg.data.datamodule.dataset_name,
            cfg.model.name),
        project=wandb_config.project,
        entity=wandb_config.entity,
        tags=cfg.core.tags,
        log_model=True,
    )
    hydra.utils.log.info(f"W&B is now watching <{wandb_config.watch.log}>!")
    wandb_logger.watch(
        model, log=wandb_config.watch.log, log_freq=wandb_config.watch.log_freq
    )

hydra.utils.log.info(f"Instantiating the Trainer")

# The Lightning core, the Trainer
# checkpoint_callback = ModelCheckpoint(monitor="val_loss")
trainer = pl.Trainer(
    default_root_dir=hydra_dir,
    logger=wandb_logger,
    callbacks=None,
    # callbacks=[checkpoint_callback],
    val_check_interval=cfg.logging.val_check_interval,
    log_every_n_steps=10,
    gpus=0,
    # accelerator=None,  # 'dp', "ddp" if args.gpus > 1 else None,
    **cfg.train,
)

# num_samples = len(datamodule.train_dataset)m
num_classes = cfg.model.num_classes
batch_size = datamodule.batch_size["train"]

hydra.utils.log.info("Starting training with {} classes and batches of {} images".format(
    num_classes,
    batch_size))

trainer.fit(model=model, datamodule=datamodule)

hydra.utils.log.info(f"Starting testing!")
trainer.test(model=model, datamodule=datamodule)

shutil.copytree(".hydra", Path(wandb_logger.experiment.dir) / "hydra")

# Logger closing to release resources/avoid multi-run conflicts
if wandb_logger is not None:
    wandb_logger.experiment.finish()
