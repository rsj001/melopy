"""
Training script for MIDI GPT model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import argparse

import random

from tokenizer import MIDITokenizer
from dataset import MIDIDataset, get_midi_files
from model import MIDITransformer


class Trainer:
    """Trainer class for MIDI GPT model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 100  # Assuming 100 epochs max
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass
            output = self.model(input_ids, target_ids)
            logits, loss = output
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                output = self.model(input_ids, target_ids)
                logits, loss = output
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, filename='checkpoint.pt'):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename='checkpoint.pt'):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            print(f"Checkpoint {path} not found")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
        return True
    
    def train(self, num_epochs: int, save_every: int = 5):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Training on {self.device}")
        print(f"Model has {self.model.get_num_params():,} parameters")
        
        start_epoch = self.epoch # 1-indexed, 真受不了 0

        for epoch in range(start_epoch + 1, start_epoch + num_epochs + 1):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                if val_loss is not None:
                    print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model.pt") # save with special name!
                        print(f"New best validation loss: {val_loss:.4f}")
            
            # Save checkpoint periodically
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            self.save_checkpoint(f'checkpoint_quicksave.pt')
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pt')
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train MIDI GPT model')
    parser.add_argument('--data_dir', type=str, default='data/train', help='Directory containing MIDI files')
    parser.add_argument('--val_data_dir', type=str, default='data/val', help='Directory containing MIDI files (val)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for checkpoints')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')

    parser.add_argument('--save_every', type=int, default=5, help='Checkpoints saving frequency')

    parser.add_argument(
        '--debug',
        type=int,
        nargs='?',
        const=25,
        default=None,
        help='Trim the training set'
    )

    parser.add_argument(
        '--resume',
        type=str,
        nargs='?', # 可选参数
        const='final_model.pt', # 如果只写 --resume，不指定值，就用默认路径
        default=None, # 如果不写 --resume，就是 None，啊啊啊烦死了
        help='Resume from checkpoint'
    )
    
    parser.add_argument('--piano_channels', type=str, default='0', help='Comma-separated list of MIDI channels for piano (default: 0)')

    args = parser.parse_args()
    
    # Parse piano channels
    piano_channels = [int(c.strip()) for c in args.piano_channels.split(',')]
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = MIDITokenizer(
        min_pitch=36,  # C2
        max_pitch=84,  # C7
        num_velocity_bins=32,
        max_time_shift=100,
        time_shift_resolution=10
    )
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Save tokenizer config
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tokenizer_config = {
        'min_pitch': tokenizer.min_pitch,
        'max_pitch': tokenizer.max_pitch,
        'num_velocity_bins': tokenizer.num_velocity_bins,
        'max_time_shift': tokenizer.max_time_shift,
        'time_shift_resolution': tokenizer.time_shift_resolution,
        'vocab_size': tokenizer.vocab_size
    }
    with open(os.path.join(args.checkpoint_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Load MIDI files
    print(f"Loading MIDI files from {args.data_dir}...")
    midi_files = get_midi_files(args.data_dir)

    print(f"Loading MIDI files from {args.val_data_dir}...")
    val_midi_files = get_midi_files(args.val_data_dir)


    if args.debug:
        if len(midi_files) > args.debug:
            print(f"Dataset trimmed from {len(midi_files)} to {args.debug}")
            debug_list = random.sample(midi_files, args.debug)
            midi_files = debug_list
    
    if len(midi_files) == 0:
        print(f"No MIDI files found in {args.data_dir}")
        print("Please add MIDI files to the data directory and try again.")
        return
    
    if len(val_midi_files) == 0:
        print(f"No MIDI files found in {args.val_data_dir}")
        print("Please add MIDI files to the data directory and try again. (val)")
        return
    
    print(f"Found {len(midi_files)} MIDI files, {len(val_midi_files)} MIDI files (val)")
    
    # Create dataset
    dataset = MIDIDataset(
        midi_files=midi_files,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        piano_channels=piano_channels
    )

    val_dataset = MIDIDataset(
        midi_files=val_midi_files,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        piano_channels=piano_channels
    )
    
    if len(dataset) == 0:
        print("No sequences created from MIDI files. Check your data.")
        return
    if len(val_dataset) == 0:
        print("No sequences created from MIDI files. Check your data. (val)")
        return
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
        # num_workers=2,       # Use small >0 to speed up 然并卵
        # pin_memory=True,     # Help copy data to GPU faster 然并卵
        # persistent_workers=True  # Keeps workers alive between epochs 然并卵
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, # Note the const
        shuffle=False,       # 验证集不打乱
        num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = MIDITransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_model * 4,
        max_seq_length=args.seq_length,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=args.num_epochs, save_every=args.save_every)


if __name__ == '__main__':
    main()
