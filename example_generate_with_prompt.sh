#!/bin/bash

# Example: Generate MIDI with a prompt

# Option 1: Use a MIDI file as prompt with custom length
python melopy/generate.py \
  --checkpoint checkpoints/checkpoint_quicksave.pt \
  --prompt_midi data/train/sample_c_major.mid \
  --prompt_length 14 \
  --output results/continuation.mid \
  --max_length 512 \
  --temperature 1.2 \
  --top_k 50 \
  --top_p 0.9

# Option 2: Use entire MIDI file as prompt (no prompt_length)
# python melopy/generate.py \
#   --checkpoint checkpoints/best_model.pt \
#   --prompt_midi data/train/sample_melody.mid \
#   --output results/continuation.mid \
#   --temperature 0.9

# Option 3: Generate from scratch (no prompt)
# python melopy/generate.py \
#   --checkpoint checkpoints/best_model.pt \
#   --output results/from_scratch.mid \
#   --temperature 1.0