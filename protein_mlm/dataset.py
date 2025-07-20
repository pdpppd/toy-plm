"""
PyTorch Dataset and DataLoader utilities for protein sequences.
"""

import torch
from torch.utils.data import Dataset

from .vocab import tokenize, PAD_ID


class ProteinSeqDataset(Dataset):
    """
    Dataset for protein sequences with tokenization.
    """
    
    def __init__(self, sequences, max_len):
        """
        Args:
            sequences: List of protein sequence strings
            max_len: Maximum sequence length (including CLS/SEP tokens)
        """
        self.sequences = sequences
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        token_ids = tokenize(seq, self.max_len)
        return torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch):
    """
    Collate function for DataLoader with dynamic padding.
    
    Args:
        batch: List of tokenized sequences (1D tensors)
        
    Returns:
        Padded batch tensor of shape (B, T)
    """
    # Find maximum length in this batch
    max_len = max(len(seq) for seq in batch)
    
    # Pad sequences to max length
    padded_sequences = []
    for seq in batch:
        pad_len = max_len - len(seq)
        if pad_len > 0:
            # Pad with PAD_ID tokens
            padded_seq = torch.cat([
                seq, 
                torch.full((pad_len,), PAD_ID, dtype=torch.long)
            ])
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    # Stack into batch tensor
    return torch.stack(padded_sequences)  # (B, T)


def create_dataloaders(train_sequences, val_sequences, config):
    """
    Create train and validation DataLoaders.
    
    Args:
        train_sequences: List of training sequences
        val_sequences: List of validation sequences  
        config: Configuration object with data and train parameters
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = ProteinSeqDataset(train_sequences, config.data.max_len)
    val_dataset = ProteinSeqDataset(val_sequences, config.data.max_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return train_loader, val_loader