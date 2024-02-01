import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import os
from pathlib import Path

from utils import get_device, set_seed
from data import EpicuriousDS
from model import LSTM

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", type=str, required=True)
    # parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=888, required=False)
    parser.add_argument("--n_epochs", type=int, default=30, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--max_len", type=int, default=200, required=False)
    parser.add_argument("--lr", type=float, default=0.0001, required=False)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


@torch.no_grad()
def validate(val_dl, model, device):
    model.eval()

    cum_loss = 0
    for in_token_id, out_token_id in val_dl:
        in_token_id = in_token_id.to(device)
        out_token_id = out_token_id.to(device)

        loss = model.get_loss(in_token_id=in_token_id, out_token_id=out_token_id)
        cum_loss += loss.item()

    model.train()
    return cum_loss / len(val_dl)


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = get_device()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    ds = EpicuriousDS(json_path=args.JSON_PATH, tokenizer=tokenizer, max_len=args.MAX_LEN)
    train_val_ds, _ = random_split(ds, lengths=(0.9, 0.1))
    train_ds, val_ds = random_split(train_val_ds, lengths=(0.8, 0.2))
    train_dl = DataLoader(
        train_ds, batch_size=args.BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True,
    )

    model = LSTM(vocab_size=tokenizer.vocab_size).to(DEVICE)
    optim = AdamW(model.parameters(), lr=args.LR)

    for epoch in range(1, args.N_EPOCHS + 1):
        cum_loss = 0
        for in_token_id, out_token_id in tqdm(train_dl, leave=False):
            in_token_id = in_token_id.to(DEVICE)
            out_token_id = out_token_id.to(DEVICE)

            loss = model.get_loss(in_token_id=in_token_id, out_token_id=out_token_id)
            optim.zero_grad()
            loss.backward()
            optim.step()

            cum_loss += loss.item()
        train_loss = cum_loss / len(train_dl)
        val_loss = validate(val_dl=val_dl, model=model, device=DEVICE)

        log = f"""[ {epoch}/{args.N_EPOCHS} ]"""
        log += f"[ Train loss: {train_loss:.4f} ]"
        log += f"[ Val loss: {val_loss:.4f} ]"
        print(log)

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.G.state_dict(), str(Path(save_dir)/f"epoch_{epoch}.pth"))


if __name__ == "__main__":
    main()
