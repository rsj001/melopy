"""
Training script for MIDI GPT model.
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
import os
import json
from tqdm.auto import tqdm
import argparse
import math

from typing import Optional, List
import random
import datetime

from tokenizer import MIDITokenizer
from dataset import MIDIDataset, get_midi_files
from model import MIDITransformer # , UncertaintyLossWrapper
from visualizer import TrainVisualizer

class Trainer:
    """Trainer class for MIDI GPT model."""
    
    def __init__(
        self,
        model: MIDITransformer,
        train_loader: DataLoader,
        categories: List[str],
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints',
        vis: Optional[TrainVisualizer] = None,
        save_every: int = 5
    ):
        self.save_every = save_every
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.vis = vis
        # self.num_tasks = 4 # HARDCODED
        self.num_epochs = num_epochs
        self.categories = categories
        # HARDCODED
        self.loss_weights = torch.tensor([1.5, 1.1, 0.6, 1.0, 1.0, 0.3, 0.3] ,device=device)
        # self.uncertainty = UncertaintyLossWrapper(self.num_tasks, device)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW([{
            'params': model.parameters(),
            'lr': learning_rate,
            'weight_decay': weight_decay,
            'betas':(0.9, 0.98),
        },
        # {
        #     'params': self.uncertainty.parameters(),
        #     'lr': learning_rate,
        #     'weight_decay': 0.0,
        #     'betas':(0.9, 0.95),
        # }
        ])
        
        # Learning rate scheduler, based on num_epochs, init on first time run
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = int(0.06 * num_training_steps)
        
        # TODO TODO not an efficient way to do this
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
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
            logits, losses = output
            
            loss = self.loss_weights @ losses
            # loss = self.uncertainty(losses)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(self.uncertainty.parameters(), max_norm=0.1)
            
            self.optimizer.step()
            self.scheduler.step()

            
            with torch.no_grad():
                # Update metrics
                total_loss += loss.item()
                self.global_step += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
                })

                # --- TensorBoard metrics ---
                if self.vis is not None:
                    if self.global_step % self.vis.log_interval == 0:
                        accuracies = []
                        for idx, _logits in enumerate(logits):
                            target_for_type = target_ids[..., idx]
                            pred_ids = torch.argmax(_logits, dim=-1)
                            correct = (pred_ids == target_for_type)[target_for_type != 0].float() # NOTE THIS IS HARDCODED 
                            accuracies.append(correct.mean().item())

                        self.vis.log_loss(loss.item(), self.global_step)
                        self.vis.log_lr(self.optimizer, self.global_step)

                        # self.vis.log_grad_norm(self.uncertainty, self.global_step, name = 'uncertainty_grad_norm')
                        self.vis.log_grad_norm(self.model, self.global_step)
                        self.vis.log_loss(total_loss / (batch_idx + 1), self.global_step, prefix="train", name="avg_loss")

                        for idx, category in enumerate(self.categories):
                            self.vis.log_loss(accuracies[idx], self.global_step, prefix="train_acc_token", name=category)
                            self.vis.log_loss(losses[idx], self.global_step, prefix="train_loss_token", name=category)

        
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
                logits, losses = output
                total_loss += self.loss_weights @ losses
                # total_loss += self.uncertainty(losses)
        
        avg_loss = total_loss / len(self.val_loader)

        if self.vis is not None:
            self.vis.log_scalar("loss", avg_loss, self.global_step, prefix="val")

        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""

        def build_config_from_attrs(obj, attr_names):
            config = {}
            for name in attr_names:
                if not hasattr(obj, name):
                    raise AttributeError(f"Object has no attribute '{name}'")
                config[name] = getattr(obj, name)
            return config
        
        # HARDCODE
        model_config = build_config_from_attrs(self.model, ["vocab_sizes", "d_model", "num_layers", "num_heads", "max_seq_length", "dropout", "pad_token_id"])
        with open(os.path.join(self.checkpoint_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=4)
            
        checkpoint = {
            # 'uncertainty_state_dict': self.uncertainty.state_dict(),
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
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        
        
        path = os.path.join(self.checkpoint_dir, filename)
        config_path = os.path.join(self.checkpoint_dir, "model_config.json")
        if not os.path.exists(path):
            print(f"Checkpoint {path} not found")
            return False
        
        # migration, checkpoint found but no json, use 
        if not os.path.exists(config_path):
            print(f"Checkpoint config {config_path} not found. Use parameters from console.")
        else:
            model_config = json.load(open(config_path))
            # 重新实例化 model
            # TODO config SAFE?
            self.model = MIDITransformer(**model_config).to(self.device)
            if self.vis is not None:
                self.vis.preload_model = self.model
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.uncertainty.load_state_dict(checkpoint['uncertainty_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}, step {self.global_step}")
        return True
    
    def train(self):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Training on {self.device}")
        print(f"Model has {self.model.get_num_params():,} parameters")
        
        num_epochs = self.num_epochs
        
        start_epoch = self.epoch # 1-indexed, 真受不了 0

        for epoch in range(start_epoch + 1, num_epochs + 1): # 总数，而非叠加
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                if val_loss is not None:
                    print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}", end="")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model.pt") # save with special name!
                        print(f" New best!")
                    else:
                        print("")
            
            # Save checkpoint periodically
            if epoch % self.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            self.save_checkpoint(f'checkpoint_quicksave.pt')

            if self.vis is not None:
                self.vis.generate_and_log_midi(self.global_step)

        # Save final checkpoint
        self.save_checkpoint('final_model.pt')
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train MIDI GPT model')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        '--data_dir_with_weights',
        type=str,
        nargs='+',
        help='Directory containing MIDI files with weights (e.g., data/train:1.0)'
    )
    group.add_argument(
        '--load_pth_with_weights',
        type=str,
        nargs='+',
        help='Path(s) to preprocessed .pth dataset files with weights(e.g., data/train.pth:1.0)'
    )

    parser.add_argument('--val_data_dir', type=str, default='data/val', help='Directory containing MIDI files (val)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for checkpoints')

    args, _ = parser.parse_known_args()
    default_log_dir = os.path.join(args.checkpoint_dir, "logs")
    parser.add_argument('--log_dir', type=str, default=default_log_dir, help='Directory for logs')
    parser.add_argument('--result_dir', type=str, default="results/auto", help='Directory for generated files')

    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    # 到底要不要持久化呢
    # 还是算了吧，
    
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.12, help='Dropout')
    
    parser.add_argument('--save_every', type=int, default=5, help='Checkpoints saving frequency')
    parser.add_argument('--chunk_stride', type=int, default=256, help='Stride for sequence chunking')
    parser.add_argument('--log_interval', type=int, default=500, help='Log frenqeuency (in steps)')

    parser.add_argument(
        '--debug',
        type=int,
        nargs='?',
        const=25,
        default=None,
        help='Trim the training set for debugging'
    )

    parser.add_argument(
        '--preprocess_dataset_as',
        type=str,
        nargs='?',
        const="saved_tensor.pth",
        default=None,
        help='Save whole dataset as .pth tensor and exit'
    )

    parser.add_argument(
        '--resume',
        type=str,
        nargs='?', # 可选参数
        const='final_model.pt', # 如果只写 --resume，不指定值，就用默认路径
        default=None, # 如果不写 --resume，就是 None，啊啊啊烦死了
        help='Resume from checkpoint'
    )

    parser.add_argument('--piano_channels', type=str, default='0,1,2,3,4,5', help='Comma-separated list of MIDI channels for piano (default: 0)')
    parser.add_argument('--cpu_num_workers', type=int, default=16, help='The number of CPU workers for pre-processing data')

    # only chunk_stride data_dir_with_weights seq_length piano_channels matters in preprocessing
    # TODO: 把训练逻辑和预处理分开
    args = parser.parse_args()
    
    # Parse piano channels
    piano_channels = [int(c.strip()) for c in args.piano_channels.split(',')]
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = MIDITokenizer()
    
    # go default, check tokenizer.py for params
    # 我并不需要怎么重建这个 tokenizer

    print(f"Vocabulary size: {tokenizer.vocab_sizes}")
    print(f"Total Size: {len(tokenizer.vocab_sizes)}")
    # Save tokenizer config
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # tokenizer_config = {
    #     'min_pitch': tokenizer.min_pitch,
    #     'max_pitch': tokenizer.max_pitch,
    #     'velocity_bins': tokenizer.velocity_bins,
    #     'duration_bins': tokenizer.duration_bins,
    #     'time_shift_bins': tokenizer.time_shift_bins,
    #     'version': tokenizer.version
    # }
    # with open(os.path.join(args.checkpoint_dir, 'tokenizer_config.json'), 'w') as f:
    #     json.dump(tokenizer_config, f, indent=2)
    
    if args.load_pth_with_weights is not None:
        if args.preprocess_dataset_as is not None:
            parser.error("--load_pth_with_weights conflicted with --preprocess_dataset_as")
            return
        dataset_list = []
        for item in args.load_pth_with_weights:
            path, weight = item.split(':')
            weight = float(weight)
            print(f"Loading MIDI pth from {path}... ", end='')
            loaded_data = MIDIDataset.load(path, tokenizer)

            tot_num = len(loaded_data)
            sample_num = max(0, int(tot_num * weight))
            sample_num = min(len(loaded_data), sample_num)
            ids = random.sample(range(len(loaded_data)), sample_num)
            print(f"Random sampling {weight} * {tot_num} -> {sample_num} sequences... ")
            dataset_list.append(Subset(loaded_data, ids))

        dataset = ConcatDataset(dataset_list)
        dataset_len = len(dataset)
        if dataset_len == 0:
            print("No sequences created from MIDI files. Check your data.")
            return
        print(f"{dataset_len} sequences loaded in total.")
    else:
        midi_files = []
        for item in args.data_dir_with_weights:
            path, weight = item.split(':')
            weight = float(weight)
            print(f"Loading MIDI files from {path}... ", end='')
            cur_midi_files = get_midi_files(path)
            tot_num = len(cur_midi_files)
            sample_num = max(0, int(weight * tot_num))
            sample_num = min(sample_num, tot_num)
            print(f"Random sampling {weight} * {tot_num} -> {sample_num} files... ")
            cur_midi_files = random.sample(cur_midi_files, sample_num)
            midi_files += cur_midi_files
        if args.debug:
            if len(midi_files) > args.debug:
                print(f"Debug Option: Training set trimmed from {len(midi_files)} to {args.debug} files.")
                debug_list = random.sample(midi_files, args.debug)
                midi_files = debug_list
        if len(midi_files) == 0:
            print(f"No MIDI files found in {args.data_dir_with_weights}. Check the format.")
            print("Please add MIDI files to the data directory and try again.")
            return
        print(f"Found {len(midi_files)} MIDI files")
        
        # Create dataset
        dataset = MIDIDataset(
            midi_files=midi_files,
            tokenizer=tokenizer,
            seq_length=args.seq_length,
            piano_channels=piano_channels,
            stride=args.chunk_stride,
            num_workers=args.cpu_num_workers
        )
        if len(dataset) == 0:
            print("No sequences created from MIDI files. Check your data.")
            return
        if args.preprocess_dataset_as is not None:
            dataset.save(args.preprocess_dataset_as, {
                "data_dir_with_weights": args.data_dir_with_weights
            })
            print("Program terminated.")
            return
    
    
    print(f"Loading MIDI files from {args.val_data_dir}...")
    val_midi_files = get_midi_files(args.val_data_dir)
    if len(val_midi_files) == 0:
        print(f"No MIDI files found in {args.val_data_dir}")
        print("Please add MIDI files to the data directory and try again. (val)")
        return
    print(f"Found {len(val_midi_files)} MIDI files (val)")
    val_dataset = MIDIDataset(
        midi_files=val_midi_files,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        piano_channels=piano_channels,
        stride=args.chunk_stride,
        num_workers=args.cpu_num_workers
    )
    if len(val_dataset) == 0:
        print("No sequences created from MIDI files. Check your data. (val)")
        return
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,       # Use small >0 to speed up 然并卵
        pin_memory=True,     # Help copy data to GPU faster 然并卵
        persistent_workers=True  # Keeps workers alive between epochs 然并卵
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16, # 为了统一标准，这应该是常数
        shuffle=False,       # 验证集不打乱
        num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = MIDITransformer(
        vocab_sizes=tokenizer.vocab_sizes,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_length=args.seq_length,
        dropout=args.dropout,
        pad_token_id=0
        # 严格意义上这不是 token
    )

    os.makedirs(args.result_dir, exist_ok=True)
    generation_args = {
        # Checkpoint options are useless here
        "output_dir": args.result_dir, # this is processed in visualizer, not generator
        "output": "This will be forced to change in visualizer.mid",
        "prompt_midi": None,
        "prompt_length": None,
        "max_length": 512,
        "temperature": [1.2, 1.0, 0.8, 1.0, 1.0, 0.8, 0.8],
        "top_k": [20] * 7,
        "top_p": [0.9] * 7,
        "seed": None,
        "piano_channels": '0, 1, 2, 3, 4, 5'
    } # 这是给Visualizer的Generation准备的
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        num_epochs=args.num_epochs, # 还用于重建 lambda scheduler
        save_every=args.save_every,
        train_loader=train_loader,
        categories=tokenizer.categories,
        val_loader=val_loader,
        learning_rate=args.lr, # learning_rate 会被 resume 覆盖
        checkpoint_dir=args.checkpoint_dir,
        vis=TrainVisualizer(log_dir = args.log_dir, log_interval=args.log_interval, generation_args=generation_args, preload_model=model)
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()
    if trainer.vis is not None:
        trainer.vis.close()


if __name__ == '__main__':
    main()
