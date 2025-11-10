import os
import torch
from torch.utils.data import Dataset
from typing import List, Optional
from tokenizer import MIDITokenizer
from tqdm.auto import tqdm


class MIDIDataset(Dataset):
    """
    Dataset for MIDI sequences.
    Handles chunking of long sequences and padding.
    """
    
    def __init__(
        self,
        midi_files: List[str],
        tokenizer: MIDITokenizer,
        seq_length: int = 512,
        stride: Optional[int] = None,
        piano_channels: Optional[List[int]] = None
    ):
        """
        Args:
            midi_files: List of paths to MIDI files
            tokenizer: MIDITokenizer instance
            seq_length: Maximum sequence length for training
            stride: Stride for sequence chunking (default: seq_length, no overlap)
            piano_channels: MIDI channels to extract (None = [0], for piano-only)
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length
        # self.stride > seq_length makes no sense in this case, I suppose
        
        # Default to channel 0 (piano) for piano-only dataset
        # clarify tracks / channels here:
        # A "piano channel" in MIDI is not a specific technical term, but rather a user-defined designation, usually MIDI channel 1.
        if piano_channels is None:
            piano_channels = [0]
        
        # Tokenize all MIDI files
        self.sequences = []
        print(f"Processing {len(midi_files)} MIDI files (channels: {piano_channels})...")
        
        pbar = tqdm(midi_files, desc=f'Preprocessing')
        for midi_file_idx, midi_file in enumerate(pbar):
            try:
                tokens = tokenizer.encode_midi(midi_file, piano_channels=piano_channels)
                
                # Chunk the sequence
                for i in range(0, len(tokens) - seq_length, self.stride):
                    chunk = tokens[i:i + seq_length + 1]  # +1 for target
                    if len(chunk) == seq_length + 1:
                        self.sequences.append(chunk)
            # TODO : efficiency evaluation

            except Exception as e:
                tqdm.write(f"Error processing {midi_file}: {e}")
                continue
        
        print(f"Created {len(self.sequences)} sequences of length {seq_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns input and target sequences.
        Target is input shifted by 1 position.
        """
        sequence = self.sequences[idx]
        
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }


def get_midi_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Get all MIDI files from a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of MIDI file paths
    """
    midi_files = []
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.mid', '.midi')):
                    midi_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if file.endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(directory, file))
    
    return midi_files
