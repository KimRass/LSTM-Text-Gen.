#!/bin/sh

source ../../venv/cv/bin/activate
source set_pythonpath.sh

python3 ../generate.py\
    --json_path="/Users/jongbeomkim/Documents/datasets/archive/full_format_recipes.json"\
    --model_params="/Users/jongbeomkim/Documents/lstm/lstm-epicurious-epoch=49-train_loss=1.899-val_loss=1.848.ckpt"\
    --n_sample=10\
    --n_cpus=4\
    --save_path="/Users/jongbeomkim/Desktop/workspace/LSTM-Text-Gen./gen_texts.txt"\
