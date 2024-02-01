# Source: https://www.kaggle.com/datasets/hugodarwood/epirecipes

import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from transformers import AutoTokenizer


class EpicuriousDS(Dataset):
    def __init__(self, json_path, tokenizer, max_len):
        super().__init__()

        with open(json_path) as f:
            data = json.load(f)

        self.prefs = list()
        self.direcs = list()
        self.texts = list()
        for sample in data:
            if "title" in sample and "directions" in sample:
                if sample["directions"]:
                    pref = f"""The recipe for "{sample["title"].strip()}":\n\n"""
                    direc = " " + f"\n ".join(sample["directions"])
                    text = pref + direc
                    self.prefs.append(pref)
                    self.direcs.append(direc)
                    self.texts.append(text)

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
    def __init__(self, json_path, tokenizer, max_len, batch_size, n_cpus):
        super().__init__()

        self.json_path = json_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_cpus = n_cpus

        self.prepare_data_per_node = False

    def prepare_data(self):
        pass

    def setup(self, stage):
        ds = EpicuriousDS(json_path=self.json_path, tokenizer=self.tokenizer, max_len=self.max_len)

        test_size = int(len(ds) * 0.1) // self.batch_size * self.batch_size
        train_val_size = len(ds) - test_size
        train_val_ds, self.test_ds = random_split(ds, lengths=(train_val_size, test_size))

        if stage == "fit":
            self.train_ds, self.val_ds = random_split(train_val_ds, lengths=(0.8, 0.2))

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
            drop_last=False,
            num_workers=self.n_cpus,
        )


if __name__ == "__main__":        
    json_path = "/Users/jongbeomkim/Documents/datasets/archive/full_format_recipes.json"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    ds = EpicuriousDS(json_path=json_path, tokenizer=tokenizer, max_len=100)
    ds.prefs[: 10]
    in_token_id, out_token_id = ds[0]

    prefs = ds.prefs
    prefs[: 5]
    
    tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_len + 1,
        add_special_tokens=True,
        return_attention_mask=False,
    )
    sample
    torch.as_tensor([43523])
    torch.equal(torch.as_tensor([43523]), sample)
    tokenizer.all_special_ids
    tokenizer.convert_ids_to_tokens(50256)