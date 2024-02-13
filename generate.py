# References:
    # https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/05_autoregressive/01_lstm/lstm.ipynb

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os
import random

from utils import get_device, set_seed
from data import EpicuriousLDM
from model import LSTMLM

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_args(to_upperse=True):
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--model_params", type=str, required=True)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=888, required=False)
    parser.add_argument("--max_len", type=int, default=200, required=False)

    args = parser.parse_args()

    if to_upperse:
        args_dict = vars(args)
        new_args_dict = dict()
        for k, v in args_dict.items():
            new_args_dict[k.upper()] = v
        args = argparse.Namespace(**new_args_dict)    
    return args


class TextGenerator(object):
    def __init__(self, model_params, tokenizer, max_len, device):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

        self.model = LSTMLM.load_from_checkpoint(
            model_params, vocab_size=tokenizer.vocab_size, map_location=device,
        )

        self.pad_id_tensor = torch.as_tensor([self.tokenizer.pad_token_id]).to(self.device)

    def tokenize(self, x):
        return self.tokenizer(
            x, padding=False, add_special_tokens=False, return_attention_mask=False,
        )

    def sample(self, prob, temp):
        prob **= 1 / temp
        prob /= torch.sum(prob, dim=0)
        sample = torch.multinomial(prob, num_samples=1)
        return sample

    def token_id_to_str(self, token_id):
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(token_id[0]),
        )

    def generate(self, prompt, temp=1):
        tokenized = self.tokenize(prompt)
        in_token_id = torch.as_tensor(tokenized.input_ids)[None, ...]
        cur_len = in_token_id.size(1)
        while True:
            in_token_id = in_token_id.to(self.device)
            pred = self.model(in_token_id)
            prob = F.softmax(pred[0, -1, :], dim=0)
            sample = self.sample(prob=prob, temp=temp)

            if torch.equal(self.pad_id_tensor, sample) or cur_len >= self.max_len:
                break

            in_token_id = torch.cat([in_token_id, sample[None, ...]], dim=1)
            cur_len += 1
        text = self.token_id_to_str(in_token_id)
        return text


def main():
    args = get_args()
    set_seed(args.SEED)
    DEVICE = get_device()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dm = EpicuriousLDM(
        json_path=args.JSON_PATH,
        tokenizer=tokenizer,
        max_len=args.MAX_LEN,
        batch_size=1,
        n_cpus=args.N_CPUS,
        seed=args.SEED,
    )
    dm.setup()

    text_gen = TextGenerator(
        model_params=args.MODEL_PARAMS,
        tokenizer=tokenizer,
        max_len=args.MAX_LEN,
        device=DEVICE,
    )

    test_prefs = dm.test_prefs[: args.N_SAMPLES]
    random.shuffle(test_prefs)
    gen_texts = list()
    for prompt in tqdm(test_prefs, leave=False):
        gen_text = text_gen.generate(prompt)
        gen_texts.append(gen_text)
        # break

    with open(args.SAVE_PATH, mode="w") as f:
        for row in gen_texts:
            f.write(row + "\n\n\n")


if __name__ == "__main__":
    main()
