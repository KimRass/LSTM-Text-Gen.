# References:
    # https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/05_autoregressive/01_lstm/lstm.ipynb

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
import pytorch_lightning as pl


class LSTMLM(pl.LightningModule):
    def __init__(self, vocab_size, lr=None, embed_dim=100, hidden_dim=128, n_layers=1):
        super().__init__()

        self.lr = lr

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, vocab_size)

        self.val_losses = list()

    def forward(self, x):
        x = self.embed(x)
        x, (_, _) = self.lstm(x)
        x = self.proj(x)
        return x

    def get_loss(self, in_token_id, out_token_id):
        b, l = in_token_id.shape
        pred_token_id = self(in_token_id)
        loss = F.cross_entropy(pred_token_id.view(b * l, -1), out_token_id.view(b * l))
        return loss

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.lr)
        return [optim]

    def training_step(self, batch, batch_idx):
        in_token_id, out_token_id = batch
        loss = self.get_loss(in_token_id=in_token_id, out_token_id=out_token_id)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        in_token_id, out_token_id = batch
        loss = self.get_loss(in_token_id=in_token_id, out_token_id=out_token_id)
        self.log("val_loss", loss)
        self.val_losses.append(loss)
        return {"val_loss": loss}

    def on_validation_epch_end(self):
        avg_val_loss = torch.stack(self.val_losses, dim=0).mean()
        self.log("val_loss", avg_val_loss)
        self.val_losses.clear()
