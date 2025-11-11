#!/bin/bash

# only chunk_stride data_dir_with_weights seq_length piano_channels matters in preprocessing

python melopy/train.py \
  --preprocess_dataset_as favs.pth \
  --data_dir_with_weights data/train/favs:1.0 \
  --seq_length 512 \
  --chunk_stride 256 