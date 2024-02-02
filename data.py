# Source: https://www.kaggle.com/datasets/hugodarwood/epirecipes

import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from transformers import AutoTokenizer


class EpicuriousDS(Dataset):
    def __init__(self, prefs, direcs, tokenizer, max_len):
        super().__init__()

        self.prefs = prefs
        self.direcs = direcs
        self.texts = [pref + direc for pref, direc in zip(prefs, direcs)]

        tokenized = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_len + 1,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        self.ls_token_ids = tokenized["input_ids"]

    def __len__(self):
        return len(self.ls_token_ids)

    def __getitem__(self, idx):
        token_ids = self.ls_token_ids[idx]
        in_token_id = torch.as_tensor(token_ids[: -1])
        out_token_id = torch.as_tensor(token_ids[1:])
        return in_token_id, out_token_id


class EpicuriousLDM(pl.LightningDataModule):
    def __init__(self, json_path, tokenizer, max_len, batch_size, n_cpus, seed):
        super().__init__()

        self.json_path = json_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        self.seed = seed

        self.prepare_data_per_node = False

    def prepare_data(self):
        pass

    def setup(self, stage=""):
        with open(self.json_path) as f:
            data = json.load(f)

        prefs = list()
        direcs = list()
        for sample in data:
            if "title" in sample and "directions" in sample:
                if sample["directions"]:
                    prefs.append(f"""The recipe for "{sample["title"].strip()}":\n""")
                    direcs.append(" " + f"\n ".join(sample["directions"]))

        (
            train_val_prefs,
            self.test_prefs,
            train_val_direcs,
            test_direcs,
        ) = train_test_split(prefs, direcs, test_size=0.1, random_state=self.seed)
        self.test_ds = EpicuriousDS(
            prefs=self.test_prefs, direcs=test_direcs, tokenizer=self.tokenizer, max_len=self.max_len,
        )

        if stage == "fit":
            (
                train_prefs,
                val_prefs,
                train_direcs,
                val_direcs,
            ) = train_test_split(train_val_prefs, train_val_direcs, test_size=0.2, random_state=self.seed)

            self.train_ds = EpicuriousDS(
                prefs=train_prefs, direcs=train_direcs, tokenizer=self.tokenizer, max_len=self.max_len,
            )
            self.val_ds = EpicuriousDS(
                prefs=val_prefs, direcs=val_direcs, tokenizer=self.tokenizer, max_len=self.max_len,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            num_workers=self.n_cpus,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            num_workers=self.n_cpus,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
            num_workers=self.n_cpus,
        )


if __name__ == "__main__":        
    json_path = "/Users/jongbeomkim/Documents/datasets/archive/full_format_recipes.json"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
