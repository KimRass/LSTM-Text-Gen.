# References:
    # https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/05_autoregressive/01_lstm/lstm.ipynb

import torch
from torch import nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, n_layers=1):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, vocab_size)

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


if __name__ == "__main__":
    vocab_size = 1000
    model = LSTM(vocab_size=vocab_size)
    x = torch.randint(0, 1000, size=(4, 512))

    out = model(x)
    out.shape
