"""
Utility functions for training, checkpointing, and reproducibility.
"""

import torch
import torch.backends.cudnn as cudnn
import random
import os
import json
import numpy as np
from typing import Dict, Any
import yaml


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    cudnn.benchmark = False
    cudnn.deterministic = True


def save_checkpoint(state, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, map_location="cpu"):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=map_location)
    print(f"Checkpoint loaded from {path}")
    return checkpoint


class WarmupLinearDecay:
    """Learning rate scheduler with warmup and linear decay."""
    
    def __init__(self, total_steps, warmup_steps):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        """Get learning rate multiplier for given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return step / max(1, self.warmup_steps)
        else:
            # Linear decay
            return max(0.0, (self.total_steps - step) / max(1, self.total_steps - self.warmup_steps))


class DictConfig:
    """Simple configuration object that allows dot notation access."""
    
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, DictConfig(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self):
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return DictConfig(config_dict)


def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    if hasattr(args, 'epochs') and args.epochs is not None:
        config.train.epochs = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if hasattr(args, 'lr') and args.lr is not None:
        config.train.lr = args.lr
    if hasattr(args, 'limit') and args.limit is not None:
        config.data.limit = args.limit
    return config


def get_device(config):
    """Get appropriate device for training."""
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif config.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS")
    elif config.device in ["auto", "cuda", "mps"]:
        # Auto-detect best available device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon MPS (auto-detected)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA (auto-detected): {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("Using CPU (auto-detected)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def compute_metrics(logits, labels):
    """
    Compute masked token accuracy.
    
    Args:
        logits: (B, T, V) tensor
        labels: (B, T) tensor with -100 for non-masked positions
        
    Returns:
        Dictionary with metrics
    """
    # Get predictions
    predictions = logits.argmax(dim=-1)  # (B, T)
    
    # Only consider masked positions (labels != -100)
    mask = labels != -100
    
    if mask.sum() == 0:
        return {"masked_token_acc": 0.0, "masked_tokens": 0}
    
    # Calculate accuracy on masked positions only
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return {
        "masked_token_acc": accuracy.item(),
        "masked_tokens": mask.sum().item()
    }


def log_metrics(step, epoch, metrics, lr=None, log_file=None):
    """Log training metrics."""
    log_str = f"Step {step}, Epoch {epoch}"
    for key, value in metrics.items():
        if isinstance(value, float):
            log_str += f", {key}: {value:.4f}"
        else:
            log_str += f", {key}: {value}"
    
    if lr is not None:
        log_str += f", lr: {lr:.2e}"
    
    print(log_str)
    
    # Write to log file if provided
    if log_file:
        log_entry = {
            "step": step,
            "epoch": epoch,
            "lr": lr,
            **metrics
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


def create_output_dir(config):
    """Create output directory for logs and checkpoints."""
    output_dir = os.path.join(config.logging.output_dir, config.logging.run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    log_dir = os.path.join(config.logging.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    return output_dir


def format_time(seconds):
    """Format seconds into readable time string."""
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}"