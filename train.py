import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import models as Model
import hydra
from hydra.utils import to_absolute_path
import os
from data.cifar10 import CIFAR10Data


@hydra.main(config_path="configs", config_name="simple")
def main(cfg):
    cfg.data_dir = to_absolute_path(cfg.data_dir)
    data = CIFAR10Data(**cfg.data)

    model = getattr(Model, cfg.model.name)(cfg.model.args, cfg.task)
    callbacks = []
    callbacks.append(ModelCheckpoint(**cfg.task.checkpoint))
    name = f"exp"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)      
    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=callbacks,
                         logger=logger)


    trainer.fit(model, data)
    # check if bin 0-20 has changed
    
if __name__ == "__main__":
    main()