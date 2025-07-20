# Claude Code Memory - Toy Protein MLM Project

## Project Overview
A minimal BERT-style Transformer implementation for protein sequence masked language modeling, optimized for Apple Silicon (M1/M2 Macs).

## Key Implementation Details

### Architecture
- **Model**: 8-layer Transformer encoder (512d, 8 heads, ~25M parameters) 
- **Vocabulary**: 25 tokens (20 canonical amino acids + PAD/CLS/SEP/MASK/UNK)
- **Sequences**: Up to 1024 residues (including CLS/SEP tokens)
- **Masking**: BERT-style 15% masking (80% [MASK], 10% random AA, 10% unchanged)
- **Training**: AdamW optimizer with linear warmup + decay scheduling

### Apple Silicon Optimizations
- **Device Detection**: Auto-detects MPS backend on M1/M2 Macs
- **Mixed Precision**: Only enabled on CUDA (MPS doesn't support it)
- **Modern APIs**: Uses `torch.amp` instead of deprecated `torch.cuda.amp`
- **Performance**: Significant speedup on Apple Silicon vs CPU

### File Structure
```
protein_mlm/
├── vocab.py          # 25-token vocabulary with tokenization
├── fasta.py          # FASTA loading and sequence preprocessing  
├── masking.py        # Dynamic BERT-style masking (15% rate)
├── dataset.py        # PyTorch Dataset with dynamic padding
├── model.py          # Transformer encoder with positional encoding
├── train.py          # Training loop with AMP and checkpointing
├── evaluate.py       # Model evaluation and sample predictions
├── inference.py      # Interactive masked residue prediction
└── utils.py          # Device detection, checkpointing, metrics
```

### Configuration (configs/default.yaml)
```yaml
device: "auto"        # Auto-detects MPS/CUDA/CPU
data:
  max_len: 1024       # Including CLS/SEP tokens (increased)
  train_split: 0.95   # 95% train, 5% validation
model:
  d_model: 512        # Model dimension (increased)
  n_layers: 8         # Transformer layers (increased)
  n_heads: 8          # Attention heads (increased)
  d_ff: 2048          # Feed-forward dimension (increased)
train:
  batch_size: 8       # Reduced for larger model
  lr: 0.0001          # Reduced learning rate
  warmup_steps: 1000  # Increased warmup
  epochs: 10
  amp: true           # Mixed precision (CUDA only)
eval:
  batch_size: 16      # Reduced for memory
logging:
  run_name: "large_mlm"  # Updated name
```

### Common Usage Patterns

**Training:**
```bash
# Full training
python -m protein_mlm.train --config configs/default.yaml

# Debug overfit (respects --epochs parameter)
python -m protein_mlm.train --config configs/default.yaml --debug_overfit --epochs 3

# Custom parameters
python -m protein_mlm.train --config configs/default.yaml --lr 0.0001 --batch_size 16
```

**Evaluation:**
```bash
python -m protein_mlm.evaluate --checkpoint checkpoints/large_mlm/best_model.pt
```

**Inference:**
```bash
# Interactive mode
python -m protein_mlm.inference --checkpoint checkpoints/large_mlm/best_model.pt --interactive

# Single prediction
python -m protein_mlm.inference --checkpoint checkpoints/large_mlm/best_model.pt \
  --sequence "MKTAYIAK[MASK]RQISFVKSHFS"
```

### Recent Fixes Applied

1. **Epochs Parameter Bug**: Debug mode was overriding `--epochs` argument
   - Fixed: Only sets default 20 epochs if `--epochs` not specified
   
2. **Apple Silicon Optimization**: Added MPS backend support
   - Device auto-detection: MPS > CUDA > CPU priority
   - Updated deprecated `torch.cuda.amp` to `torch.amp`
   - MPS-compatible mixed precision handling

### Technical Notes

- **YAML Parsing**: Scientific notation (3e-4) causes type errors, use decimal (0.0003)
- **Mixed Precision**: Only works on CUDA, disabled for MPS/CPU
- **Memory**: Dynamic padding reduces memory usage vs fixed-length sequences  
- **Reproducibility**: Seed setting includes cudnn.deterministic = True
- **Checkpointing**: Saves model, optimizer, scheduler, and scaler states

### Performance Expectations

**Debug Mode (10 sequences, 1 epoch, large model):**
- Loss: ~3.28 → ~3.24 (small overfitting dataset)
- Training time: ~20 seconds on Apple Silicon MPS
- Masked token accuracy: ~3-4% (small dataset limitation)
- Model size: **25.2M parameters** (vs 3.2M previously)

**Full Training (68k sequences, 10 epochs, large model):**
- Expected loss: ~4.0 → ~1.5 (better than smaller model)
- Expected perplexity: ~5-10 (improved)
- Expected accuracy: ~50-70% (higher capacity)
- Sequence coverage: Handles up to 1024 residues (~70% of dataset fully)

### Lint/Typecheck Commands
```bash
# No specific linting setup - basic Python/PyTorch project
python -m py_compile protein_mlm/*.py  # Basic syntax check
```

This implementation provides a solid foundation for protein language modeling research and can be easily extended for larger models or different tasks.