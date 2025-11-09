#!/bin/bash

# Example training script with custom parameters

python melopy/train.py \
  --data_dir data/train \
  --seq_length 512 \
  --batch_size 32 \
  --num_epochs 3 \
  --resume checkpoint_quicksave.pt \
  --debug 100