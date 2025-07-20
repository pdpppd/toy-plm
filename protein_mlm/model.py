"""
Transformer encoder model for masked language modeling on protein sequences.
"""

import torch
import torch.nn as nn
import math

from .vocab import VOCAB_SIZE, PAD_ID


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class TransformerMLM(nn.Module):
    """
    Transformer encoder for masked language modeling.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.d_model = config.model.d_model
        self.vocab_size = VOCAB_SIZE
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=PAD_ID)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, config.model.max_len)
        
        # Input dropout
        self.dropout = nn.Dropout(config.model.dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.model.n_heads,
            dim_feedforward=config.model.d_ff,
            dropout=config.model.dropout,
            activation='gelu',
            batch_first=True,  # Important: expect (batch, seq, feature)
            norm_first=True,   # Pre-layer norm (like GPT)
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.model.n_layers,
        )
        
        # Output layer norm
        self.layer_norm = nn.LayerNorm(self.d_model, eps=config.model.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(self.d_model, self.vocab_size)
        
        # Weight tying (tie input embedding and output projection weights)
        if config.model.weight_tie:
            self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def _create_padding_mask(self, input_ids):
        """Create attention mask for padding tokens."""
        # True for positions that should be masked (padding)
        return (input_ids == PAD_ID)
    
    def forward(self, input_ids):
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len) tensor of token IDs
            
        Returns:
            logits: (batch_size, seq_len, vocab_size) tensor
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        embeddings = self.token_embedding(input_ids)  # (B, T, D)
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)
        
        # Input dropout
        embeddings = self.dropout(embeddings)
        
        # Create padding mask
        padding_mask = self._create_padding_mask(input_ids)  # (B, T)
        
        # Transformer encoder
        hidden_states = self.transformer(
            embeddings,
            src_key_padding_mask=padding_mask
        )  # (B, T, D)
        
        # Layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)  # (B, T, V)
        
        return logits


def count_params(model):
    """Count the number of parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def create_model(config):
    """Create and return the model."""
    model = TransformerMLM(config)
    param_count = count_params(model)
    print(f"Created model with {param_count['trainable']:,} trainable parameters")
    return model