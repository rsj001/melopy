#!/bin/bash

# Example training script with custom parameters

python melopy/train.py \
  --data_dir data/train \
  --checkpoint_dir checkpoints \
  --seq_length 512 \
  --batch_size 32 \
  --num_epochs 50 \
  --resume checkpoint_quicksave.pt
