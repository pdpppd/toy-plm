# Toy Protein Masked Language Model (MLM)

A minimal BERT-style Transformer implementation for protein sequence masked language modeling.

## Features

- 🧬 **Protein-focused**: 20 canonical amino acids + special tokens
- 🎭 **BERT-style masking**: Dynamic 15% masking with 80/10/10 strategy
- 🔥 **Modern PyTorch**: Transformer encoder with mixed precision training
- 📊 **Comprehensive**: Training, evaluation, and inference utilities
- ⚡ **Efficient**: ~3M parameters, trains on CPU/GPU

## Quick Start

### Requirements

```bash
pip install torch pyyaml tqdm
```

### Training

```bash
# Full training
python -m protein_mlm.train --config configs/default.yaml

# Debug mode (overfits on 10 sequences)
python -m protein_mlm.train --config configs/default.yaml --debug_overfit

# Custom parameters
python -m protein_mlm.train --config configs/default.yaml --epochs 5 --lr 0.0001
```

### Evaluation

```bash
python -m protein_mlm.evaluate --checkpoint checkpoints/large_mlm/best_model.pt
```

### Inference

```bash
# Interactive mode
python -m protein_mlm.inference --checkpoint checkpoints/large_mlm/best_model.pt --interactive

# Single prediction
python -m protein_mlm.inference --checkpoint checkpoints/large_mlm/best_model.pt \
  --sequence "MKTAYIAK[MASK]RQISFVKSHFS"
```

## Architecture

- **Model**: 8-layer Transformer encoder (512d, 8 heads, ~25M parameters)
- **Vocab**: 25 tokens (20 AAs + PAD/CLS/SEP/MASK/UNK)  
- **Sequences**: Up to 1024 residues (including CLS/SEP tokens)
- **Training**: AdamW optimizer with warmup + linear decay
- **Masking**: 15% tokens masked using BERT strategy

## Project Structure

```
toy-plm/
├── protein_mlm/          # Core implementation
│   ├── vocab.py          # Vocabulary and tokenization  
│   ├── fasta.py          # FASTA loading and preprocessing
│   ├── masking.py        # BERT-style dynamic masking
│   ├── dataset.py        # PyTorch Dataset and DataLoader
│   ├── model.py          # Transformer encoder architecture
│   ├── train.py          # Training script and loop
│   ├── evaluate.py       # Model evaluation utilities
│   └── inference.py      # Masked residue prediction
├── configs/              # YAML configurations
├── scripts/              # Shell scripts for training/eval
├── data/                 # FASTA sequences
├── checkpoints/          # Saved models
└── logs/                 # Training logs
```

## Configuration

Edit `configs/default.yaml` to customize:

- **Model size**: `d_model`, `n_layers`, `n_heads`
- **Training**: `batch_size`, `lr`, `epochs`
- **Data**: `max_len`, `train_split`, `limit`
- **Masking**: `prob`, `mask_ratio`, `random_ratio`

## Example Results

After training, you should see:

- **Loss**: ~4.0 → ~2.0 (depending on data)
- **Perplexity**: ~8-15 on validation
- **Masked accuracy**: ~40-60%

Sample predictions:
```
Position 8: true=K top5=[K:0.34, R:0.18, N:0.07, Q:0.06, E:0.05]
```

## Implementation Notes

- **Apple Silicon optimized**: Auto-detects and uses MPS backend on M1/M2 Macs
- **Device auto-detection**: `device: auto` automatically picks best available (MPS > CUDA > CPU)
- Supports mixed precision training on CUDA (`amp: true`)
- Automatic checkpoint saving and resuming
- Dynamic padding for variable sequence lengths
- Gradient clipping and warmup scheduling
- Modern PyTorch APIs (torch.amp instead of deprecated cuda.amp)

## Extensions

Easy to extend with:
- Larger models (increase `d_model`, `n_layers`)
- Different masking strategies (span masking, etc.)
- Multi-task training (secondary structure, etc.)
- Different tokenization schemes

## License

MIT License - feel free to use for research and education!
