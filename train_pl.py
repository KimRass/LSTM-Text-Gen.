from transformers import AutoTokenizer
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import get_device, set_seed
from data_pl import EpicuriousLDM
from model_pl import LSTMLM

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

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


def main():
    args = get_args()
    set_seed(args.SEED)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dm = EpicuriousLDM(
        json_path=args.JSON_PATH,
        tokenizer=tokenizer,
        max_len=args.MAX_LEN,
        batch_size=args.BATCH_SIZE,
        n_cpus=args.N_CPUS,
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    model = LSTMLM(vocab_size=tokenizer.vocab_size, lr=args.LR)

    ckpt_callback = ModelCheckpoint(dirpath=args.SAVE_DIR, save_top_k=2, monitor="val_loss")
    trainer = pl.Trainer(
        max_epochs=args.N_EPOCHS, callbacks=[ckpt_callback],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
