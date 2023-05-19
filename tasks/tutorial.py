import pytorch_lightning as pl
from torch.optim import Adam
from torch.nn import functional as F

class Tutorial(pl.LightningModule):
    def __init__(self, task_args):
        super().__init__()
        self.save_hyperparameters()
                
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat, commitment_loss, codebook_loss, perplexity = self(x)
        recon_loss = F.mse_loss(x_hat, x)
        loss =  recon_loss + commitment_loss + codebook_loss
        
        self.log('Train/recon_loss', recon_loss, prog_bar=True)
        self.log('Train/commitment_loss', commitment_loss, prog_bar=True)
        self.log('Train/codebook_loss', codebook_loss, prog_bar=True)
        self.log('Train/total_loss', codebook_loss, prog_bar=True)
    
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat, commitment_loss, codebook_loss, perplexity = self(x)
        recon_loss = F.mse_loss(x_hat, x)
        loss =  recon_loss + commitment_loss + codebook_loss
        
        self.log('Val/recon_loss', recon_loss)
        self.log('Val/commitment_loss', commitment_loss)
        self.log('Val/codebook_loss', codebook_loss)
        self.log('Val/total_loss', codebook_loss)


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.task_args.lr)
        return [optimizer], []
