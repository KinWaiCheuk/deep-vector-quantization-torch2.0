import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.vqvae import VQVAE
import hydra
from hydra.utils import to_absolute_path
from data.cifar10 import CIFAR10Data

import os

"""
These ramps/decays follow DALL-E Appendix A.2 Training https://arxiv.org/abs/2102.12092
"""
class DecayTemperature(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The relaxation temperature τ is annealed from 1 to 1/16 over the first 150,000 updates.
        t = cos_anneal(0, 150000, 1.0, 1.0/16, trainer.global_step)
        pl_module.quantizer.temperature = t

class RampBeta(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The KL weight β is increased from 0 to 6.6 over the first 5000 updates
        # "We divide the overall loss by 256 × 256 × 3, so that the weight of the KL term
        # becomes β/192, where β is the KL weight."
        # TODO: OpenAI uses 6.6/192 but kinda tricky to do the conversion here... about 5e-4 works for this repo so far... :\
        t = cos_anneal(0, 5000, 0.0, 5e-4, trainer.global_step)
        pl_module.quantizer.kld_scale = t

class DecayLR(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        t = cos_anneal(0, 1200000, 3e-4, 1.25e-6, trainer.global_step)
        for g in pl_module.optimizer.param_groups:
            g['lr'] = t


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    cfg.data_dir = to_absolute_path(cfg.data_dir)
    data = CIFAR10Data(**cfg.data)

    model = VQVAE(cfg.model.args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor="val_recon_loss",
                                      filename="{epoch:02d}",
                                      save_top_k=2,
                                      mode="min",
                                      auto_insert_metric_name=False,
                                      save_last=True))
    if cfg.model.args.vq_flavor == 'gumbel':
        callbacks.extend([DecayTemperature(), RampBeta()])  
    name = f"exp"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)      
    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=callbacks,
                         logger=logger)


    trainer.fit(model, data)
    # check if bin 0-20 has changed
    
if __name__ == "__main__":
    main()