import os
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any
from tokenizer import MIDITokenizer
from tqdm.auto import tqdm
import multiprocessing as mp
# This is a static method.
def worker(args):
    midi_file, tokenizer, stride, seq_length, piano_channels, pad_token = args
    return_val = []
    try:
        tokens = tokenizer.encode_midi(midi_file).ids
        tokens.insert(0, tokenizer.bos_token)
        tokens.append(tokenizer.eos_token)
        Len = len(tokens)
        sl = seq_length + 1
        for i in range(0, Len - sl + 1, stride):
            return_val.append(tokens[i:i+sl])
                
    except Exception as e:
        return [], f"Error processing {midi_file}: {e}"
    return return_val, None


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
        piano_channels: Optional[List[int]] = None,
        num_workers: int = 16,
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
        self.num_workers = num_workers
        self.piano_channels = piano_channels

        self.pad_token = tokenizer.pad_token

        # self.stride > seq_length makes no sense in this case, I suppose
        
        # Default to channel 0 (piano) for piano-only dataset
        # clarify tracks / channels here:
        # A "piano channel" in MIDI is not a specific technical term, but rather a user-defined designation, usually MIDI channel 1.

        if piano_channels is None:
            piano_channels = [0, 1, 2, 3, 4, 5]
        
        # Tokenize all MIDI files
        self.sequences = []
        print(f"Processing {len(midi_files)} MIDI files (channels: {piano_channels})...")
        
        tasks = [
            (file, tokenizer, self.stride, seq_length, piano_channels, self.pad_token) 
            for file in midi_files
        ]

        print(f"Use {num_workers} workers.")
        mp.set_start_method('spawn', force=True)
        with mp.Pool(self.num_workers) as pool:
            for return_val, message in tqdm(
                pool.imap_unordered(worker, tasks),
                total=len(midi_files),
                desc="Preprocessing",
                dynamic_ncols = True
            ):
                if message is None:
                    self.sequences.extend(return_val)
                else:
                    tqdm.write(message)

        self.sequences = torch.tensor(self.sequences, dtype=torch.uint8)
        
        # print(f"Created {len(self.sequences)} sequences of length {seq_length}, processing data augmentation twice...")
        # DATA AUGMENTATION BEGIN
        # self.sequences = torch.cat([self.sequences, self.augment_pitch(self.sequences), self.augment_pitch(self.sequences)])
        # DATA AUGMENTATION END
        
        print(f"Created {len(self.sequences)} sequences of length {seq_length} in total.")
    
    # ===============================================================
    # Save / Load Methods Begin
    # ===============================================================

    def save(self, path: str, extra_meta: Optional[Dict[str, Any]] = None):
        """
        Save preprocessed dataset to disk.

        Args:
            path: Path to save file (.pt)
            extra_meta: Optional dict for extra metadata
        """
        meta = {
            "seq_length": self.seq_length,
            "stride": self.stride,
        }
        if extra_meta:
            meta.update(extra_meta)

        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save({
            "sequences": self.sequences,
            "meta": meta
        }, path)

        print(f"Dataset saved to {path} ({len(self.sequences)} sequences)")

    @classmethod
    def load(cls, path: str, tokenizer: MIDITokenizer):
        """
        Load a preprocessed dataset from disk.
        Does NOT reprocess MIDI files.

        Args:
            path: Path to .pt file
            tokenizer: tokenizer
        """
        data = torch.load(path, map_location="cpu")

        # Create empty instance without calling __init__
        obj = cls.__new__(cls)
        obj.tokenizer = tokenizer
        obj.seq_length = data["meta"].get("seq_length", 512)
        obj.stride = data["meta"].get("stride", obj.seq_length)
        obj.sequences = data["sequences"]

        # obj.midi_files is unused

        print(f"Loaded dataset from {path} ({len(obj.sequences)} sequences)")
        return obj

    # ===============================================================
    # Save / Load Methods End
    # ===============================================================

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Returns input and target sequences.
        Target is input shifted by 1 position.
        """
        sequence = self.sequences[idx].long()
        
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }
    # def augment_pitch(self, sequences, pitch_index=0, special_ids={0,1,2}):
    #     pitch_min = self.tokenizer.token_to_id["note"][f"NOTE_{self.tokenizer.min_pitch}"]
    #     pitch_max = self.tokenizer.token_to_id["note"][f"NOTE_{self.tokenizer.max_pitch}"]

    #     seq_pitch = sequences[..., pitch_index].to(torch.int16)

    #     valid_mask = torch.ones_like(seq_pitch, dtype=torch.bool)
    #     for sid in special_ids:
    #         valid_mask &= (seq_pitch != sid)
        
    #     any_token = valid_mask.any(dim=1)
    #     sequences = sequences[any_token]
    #     seq_pitch = seq_pitch[any_token]
    #     valid_mask = valid_mask[any_token]

    #     seq_pitch_min = torch.where(valid_mask, seq_pitch, pitch_max).min(dim=1).values
    #     seq_pitch_max = torch.where(valid_mask, seq_pitch, pitch_min).max(dim=1).values

    #     # 每个 sequence 可偏移范围
    #     down_range = torch.clamp(seq_pitch_min - pitch_min, min=0, max = 12)
    #     up_range   = torch.clamp(pitch_max - seq_pitch_max, min=0, max = 12)

    #     # 随机 float ∈ [0,1)，再映射到对应整数偏移范围
    #     rand_float = torch.rand(sequences.size(0), device=sequences.device)
    #     offsets = (rand_float * (up_range + down_range + 1).to(torch.float32) - down_range.to(torch.float32)).floor().to(torch.int16)

    #     keep = offsets != 0
    #     offsets = offsets[keep][:, None]
    #     valid_mask = valid_mask[keep]
    #     sequences = sequences[keep]
    #     seq_pitch = sequences[..., pitch_index].to(torch.int16)
    #     sequences[..., pitch_index] = torch.where(valid_mask, seq_pitch + offsets, seq_pitch).to(torch.uint8)

    #     # print(sequences)

    #     return sequences

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
