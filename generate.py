import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import os

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


class TextGenerator(object):
    def __init__(self, tokenizer, model_params_path, max_len):
        self.tokenizer = tokenizer
        # self.model = model
        self.max_len = max_len

        self.model = LSTM(vocab_size=self.tokenizer.vocab_size)
        state_dict = torch.load()

    def tokenize(self, x):
        return self.tokenizer(
            x, padding=False, add_special_tokens=False, return_attention_mask=False,
        )


    def token_id_to_str(self, token_id):
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(token_id[0]),
        )

    def generate(self, prompt):
        tokenized = self.tokenize(prompt)
        in_token_id = torch.as_tensor(tokenized.input_ids)[None, ...]
        while True:
            pred = self.model(in_token_id)
            prob = F.softmax(pred[0, -1, :], dim=0)
            sample = torch.multinomial(prob, num_samples=1)
            if torch.equal(torch.as_tensor([self.tokenizer.pad_token_id]), sample):
                break
            in_token_id = torch.cat([in_token_id, sample[None, ...]], dim=1)
        text = self.token_id_to_str(in_token_id)
        return text


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = get_device()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.all_special_tokens

    ds = EpicuriousDS(json_path=args.JSON_PATH, tokenizer=tokenizer, max_len=args.MAX_LEN)
    _, test_ds = random_split(ds, lengths=(0.9, 0.1))
    test_prefs = test_ds.prefs
    prompt = test_prefs[0]
    
    max_len = 200
    text_gen = TextGenerator(tokenizer=tokenizer, max_len=max_len)
    text_gen.generate(prompt)
