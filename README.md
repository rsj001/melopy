# Melopy - MIDI Generation with Transformer

A PyTorch-based Decoder-only GPT model for autoregressive generation of piano MIDI sequences.

## Features

- **Granular Tokenization**: Converts MIDI events into discrete tokens (NOTE_ON, NOTE_OFF, TIME_SHIFT, VELOCITY)
- **Piano-Only**: Filters for piano-only tracks with configurable pitch range (default: C2-C7)
- **Decoder-Only**: GPT-style architecture with causal masking

## Project Structure

```
.
├── melopy/
│   ├── tokenizer.py      # MIDI tokenization and vocabulary
│   ├── dataset.py        # Dataset loader and preprocessing
│   ├── model.py          # Decoder-only GPT architecture
│   ├── train.py          # Training loop and utilities
│   ├── visualizer.py     # Visualizing training with tensorboard
│   └── generate.py       # Generation and sampling
├── data/                # Place your MIDI files here
└── checkpoints/         # Saved model checkpoints
```

## Setup

1. Add your MIDI files to the `data/train/` directory
2. Install dependencies:
   - PyTorch
   - tensorboard
   - mido
   - pretty_midi
   - tqdm
   - pyfluidsynth (with libs)
   - matplotlib
3. Pre-process data with example_preprocess.sh
4. Train with example_train.sh
5. Generate with example_generate_with_prompt.sh
