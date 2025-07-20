"""
FASTA file loading and sequence preprocessing utilities.
"""

import random


def read_fasta(path):
    """
    Simple FASTA parser without external dependencies.
    
    Args:
        path: Path to FASTA file
        
    Returns:
        List of sequences (strings)
    """
    seqs = []
    buf = []
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                    buf = []
            else:
                buf.append(line)
        if buf:
            seqs.append("".join(buf))
    
    return seqs


# Valid canonical amino acids
VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq, to_upper=True):
    """
    Clean protein sequence by removing invalid characters.
    
    Args:
        seq: Raw protein sequence string
        to_upper: Convert to uppercase
        
    Returns:
        Cleaned sequence string
    """
    if to_upper:
        seq = seq.upper()
    
    # Remove invalid chars (simple filter)
    seq = "".join(c for c in seq if c in VALID_AAS)
    return seq


def load_and_split(path, train_frac, min_len, limit=None, seed=42):
    """
    Load FASTA file, clean sequences, and split into train/validation.
    
    Args:
        path: Path to FASTA file
        train_frac: Fraction for training (0-1)
        min_len: Minimum sequence length to keep
        limit: Maximum number of sequences to use (for debugging)
        seed: Random seed for reproducible splitting
        
    Returns:
        Tuple of (train_sequences, val_sequences)
    """
    # Load and preprocess
    raw = read_fasta(path)
    cleaned = [clean_sequence(s) for s in raw]
    filtered = [s for s in cleaned if len(s) >= min_len]
    
    print(f"Loaded {len(raw)} sequences, {len(filtered)} after filtering (min_len={min_len})")
    
    if limit:
        filtered = filtered[:limit]
        print(f"Limited to {len(filtered)} sequences for debugging")
    
    # Shuffle and split
    random.Random(seed).shuffle(filtered)
    n_train = int(len(filtered) * train_frac)
    
    train_seqs = filtered[:n_train]
    val_seqs = filtered[n_train:]
    
    print(f"Split: {len(train_seqs)} train, {len(val_seqs)} validation")
    
    return train_seqs, val_seqs


def get_sequence_stats(sequences):
    """Get basic statistics about sequence lengths."""
    if not sequences:
        return {}
    
    lengths = [len(s) for s in sequences]
    return {
        "count": len(sequences),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "mean_len": sum(lengths) / len(lengths),
        "total_residues": sum(lengths),
    }