#!/bin/sh

source ../venv/cv/bin/activate
source set_pythonpath.sh

python3 train.py\
    --json_path="/Users/jongbeomkim/Documents/datasets/archive/full_format_recipes.json"\
    --save_dir="/Users/jongbeomkim/Documents/datasets/lstm"\
    --n_cpus=3\
    --resume_from="/Users/jongbeomkim/Documents/datasets/lstm/lstm-epicurious-epoch=02-train_loss=3.822-val_loss=3.727.ckpt"
