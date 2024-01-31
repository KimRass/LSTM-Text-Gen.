# Source: https://www.kaggle.com/datasets/hugodarwood/epirecipes

import json
import string
import re
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from pprint import pprint

# Epicurious
WIDTH = 70
# MAX_LEN = 200
Q = 80

# def pad_punctuation(text):
#     text = re.sub(pattern=f"([{string.punctuation}])", repl=r" \1 ", string=text)
#     text = re.sub(pattern=" +", repl=" ", string=text)
#     return text
# print(pad_punctuation(texts[30]))
# print(texts[30])


with open("/Users/jongbeomkim/Documents/datasets/archive/full_format_recipes.json") as f:
    data = json.load(f)
len(data)
data[0].keys()
pprint(data[0], width=WIDTH)

texts = list()
for sample in data:
    if "title" in sample and "directions" in sample:
        if sample["directions"]:
            pref = f"""The recipe for "{sample["title"].strip()}":\n\n"""
            dir_text = " " + f"\n ".join(sample["directions"])
            text = pref + dir_text
            # break
            texts.append(text)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

tokenized = tokenizer(texts, add_special_tokens=False)
lens = [len(token_ids) for token_ids in tokenized["input_ids"]]
MAX_LEN = int(np.percentile(lens, Q))

tokenized = tokenizer(
    texts, padding="max_length", truncation=True, max_length=MAX_LEN, add_special_tokens=False,
)
tokenized["input_ids"][0]
