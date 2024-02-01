# Source: https://www.kaggle.com/datasets/hugodarwood/epirecipes

import json
import string
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from pprint import pprint


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