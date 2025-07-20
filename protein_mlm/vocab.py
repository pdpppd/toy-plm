"""
Vocabulary management for protein sequences.
Defines amino acid tokens and special tokens for BERT-style masking.
"""

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]

IDX2TOK = SPECIALS + AMINO_ACIDS
TOK2IDX = {t: i for i, t in enumerate(IDX2TOK)}

PAD_ID = TOK2IDX["[PAD]"]
CLS_ID = TOK2IDX["[CLS]"]
SEP_ID = TOK2IDX["[SEP]"]
MASK_ID = TOK2IDX["[MASK]"]
UNK_ID = TOK2IDX["[UNK]"]

VOCAB_SIZE = len(IDX2TOK)


def tokenize(seq: str, max_len: int):
    """
    Tokenize a protein sequence with CLS and SEP tokens.
    
    Args:
        seq: Raw protein sequence string
        max_len: Maximum sequence length including CLS/SEP tokens
        
    Returns:
        List of token IDs
    """
    # Reserve space for CLS + SEP
    core = [TOK2IDX.get(c, UNK_ID) for c in seq][:max_len - 2]
    return [CLS_ID] + core + [SEP_ID]


def detokenize(indices):
    """
    Convert token indices back to amino acid sequence (excluding specials).
    
    Args:
        indices: List or tensor of token indices
        
    Returns:
        String representation of the sequence
    """
    return "".join(IDX2TOK[i] for i in indices if i >= len(SPECIALS))


def get_vocab_info():
    """Return vocabulary information for debugging."""
    return {
        "vocab_size": VOCAB_SIZE,
        "amino_acids": AMINO_ACIDS,
        "specials": SPECIALS,
        "special_ids": {
            "PAD": PAD_ID,
            "CLS": CLS_ID,
            "SEP": SEP_ID,
            "MASK": MASK_ID,
            "UNK": UNK_ID,
        }
    }