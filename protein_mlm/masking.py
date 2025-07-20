"""
BERT-style dynamic masking for protein sequences.
"""

import torch
import random
from .vocab import PAD_ID, CLS_ID, SEP_ID, MASK_ID, AMINO_ACIDS, TOK2IDX


def mask_batch(batch_ids, mask_prob=0.15, mask_ratio=0.80, random_ratio=0.10):
    """
    Apply BERT-style masking to a batch of tokenized sequences.
    
    For each position to be masked:
    - 80% of the time: replace with [MASK] token
    - 10% of the time: replace with random amino acid
    - 10% of the time: keep unchanged
    
    Args:
        batch_ids: (B, T) LongTensor of token IDs
        mask_prob: Probability of masking each position
        mask_ratio: Fraction of masked tokens to replace with [MASK]
        random_ratio: Fraction of masked tokens to replace with random token
        
    Returns:
        Tuple of (masked_input_ids, labels) where labels are -100 for non-masked positions
    """
    B, T = batch_ids.shape
    labels = batch_ids.clone()
    
    # Special tokens that should never be masked
    special_tokens = {PAD_ID, CLS_ID, SEP_ID}
    
    # Get amino acid token IDs for random replacement
    aa_token_ids = [TOK2IDX[aa] for aa in AMINO_ACIDS]
    
    for b in range(B):
        tokens = batch_ids[b].clone()
        
        # Find candidate positions (exclude special tokens)
        candidate_positions = [
            i for i, token_id in enumerate(tokens.tolist()) 
            if token_id not in special_tokens
        ]
        
        if not candidate_positions:
            # No valid positions to mask
            labels[b, :] = -100
            continue
        
        # Determine how many positions to mask
        n_to_mask = max(1, int(len(candidate_positions) * mask_prob))
        mask_positions = random.sample(candidate_positions, n_to_mask)
        
        # Apply masking strategy to each selected position
        for pos in mask_positions:
            r = random.random()
            if r < mask_ratio:
                # Replace with [MASK] token
                tokens[pos] = MASK_ID
            elif r < mask_ratio + random_ratio:
                # Replace with random amino acid
                tokens[pos] = random.choice(aa_token_ids)
            else:
                # Keep unchanged (but still predict it)
                pass
        
        # Update batch with masked tokens
        batch_ids[b] = tokens
        
        # Set labels to -100 for non-masked positions
        for i in range(T):
            if i not in mask_positions:
                labels[b, i] = -100
    
    return batch_ids, labels


def get_masking_stats(labels):
    """
    Get statistics about masking for debugging.
    
    Args:
        labels: Tensor with -100 for non-masked positions
        
    Returns:
        Dictionary with masking statistics
    """
    total_tokens = labels.numel()
    masked_tokens = (labels != -100).sum().item()
    
    return {
        "total_tokens": total_tokens,
        "masked_tokens": masked_tokens,
        "masking_ratio": masked_tokens / total_tokens if total_tokens > 0 else 0.0,
    }