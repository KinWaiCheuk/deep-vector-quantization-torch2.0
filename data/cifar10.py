from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl

class CIFAR10Data(pl.LightningDataModule):
    """ returns cifar-10 examples in floats in range [0,1] """

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4, padding_mode='reflect'),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        dataset = CIFAR10(**self.hparams.train.dataset, transform=transform)
        dataloader = DataLoader(dataset,**self.hparams.train.dataloader)
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        dataset = CIFAR10(**self.hparams.val.dataset, transform=transform)
        dataloader = DataLoader(dataset,**self.hparams.val.dataloader)
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

