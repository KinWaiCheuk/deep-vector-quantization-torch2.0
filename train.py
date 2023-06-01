import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import models as Model
import hydra
from hydra.utils import to_absolute_path
import os
import data as Dataset

@hydra.main(config_path="configs", config_name="simple")
def main(cfg):
    cfg.data_dir = to_absolute_path(cfg.data_dir)
    data = getattr(Dataset, cfg.dataset.name)(**cfg.dataset.args)

    model = getattr(Model, cfg.model.name)(cfg.model.args, cfg.task)
    callbacks = []
    callbacks.append(ModelCheckpoint(**cfg.task.checkpoint))
    name = f"CorrectVQshape{cfg.dataset.name}-{cfg.model.name}"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)      
    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=callbacks,
                         logger=logger)


    trainer.fit(model, data)
    # check if bin 0-20 has changed
    
if __name__ == "__main__":
    main()