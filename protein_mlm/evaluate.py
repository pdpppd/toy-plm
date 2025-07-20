"""
Evaluation script for the protein masked language model.
"""

import argparse
import math
import torch
from tqdm import tqdm

from .utils import load_checkpoint, get_device, DictConfig, compute_metrics
from .fasta import load_and_split
from .dataset import create_dataloaders
from .model import create_model
from .masking import mask_batch
from .vocab import detokenize, IDX2TOK


def evaluate_model(model, val_loader, device, config):
    """Evaluate model and return detailed metrics."""
    model.eval()
    
    total_loss = 0.0
    total_acc = 0.0
    total_masked_tokens = 0
    num_batches = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = batch.to(device)
            
            # Apply masking
            masked_inputs, labels = mask_batch(
                batch,
                mask_prob=config.masking.prob,
                mask_ratio=config.masking.mask_ratio,
                random_ratio=config.masking.random_ratio
            )
            
            # Forward pass
            device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'
            with torch.amp.autocast(device_type, enabled=config.train.amp and device.type == 'cuda'):
                logits = model(masked_inputs)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
            
            # Compute metrics
            metrics = compute_metrics(logits, labels)
            total_loss += loss.item()
            total_acc += metrics["masked_token_acc"]
            total_masked_tokens += metrics["masked_tokens"]
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    perplexity = math.exp(avg_loss)
    
    return {
        "loss": avg_loss,
        "masked_acc": avg_acc,
        "perplexity": perplexity,
        "masked_tokens": total_masked_tokens
    }


def sample_predictions(model, val_loader, device, config, num_samples=5):
    """Sample some predictions to inspect model behavior."""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if len(samples) >= num_samples:
                break
                
            batch = batch.to(device)
            
            # Apply masking
            masked_inputs, labels = mask_batch(
                batch,
                mask_prob=config.masking.prob,
                mask_ratio=config.masking.mask_ratio,
                random_ratio=config.masking.random_ratio
            )
            
            # Forward pass
            device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'
            with torch.amp.autocast(device_type, enabled=config.train.amp and device.type == 'cuda'):
                logits = model(masked_inputs)
                probs = torch.softmax(logits, dim=-1)
            
            # Process first sequence in batch
            seq_tokens = batch[0].cpu()  # Original tokens
            seq_masked = masked_inputs[0].cpu()  # Masked tokens
            seq_labels = labels[0].cpu()  # Labels
            seq_probs = probs[0].cpu()  # Prediction probabilities
            
            # Find masked positions
            masked_positions = torch.where(seq_labels != -100)[0]
            
            if len(masked_positions) > 0:
                sample = {
                    "original_seq": detokenize(seq_tokens),
                    "predictions": []
                }
                
                for pos in masked_positions[:3]:  # Show first 3 masked positions
                    pos = pos.item()
                    true_token = seq_tokens[pos].item()
                    true_aa = IDX2TOK[true_token] if true_token < len(IDX2TOK) else "UNK"
                    
                    # Get top-k predictions
                    topk_probs, topk_indices = seq_probs[pos].topk(5)
                    topk_tokens = [(IDX2TOK[idx.item()], prob.item()) 
                                 for idx, prob in zip(topk_indices, topk_probs)]
                    
                    sample["predictions"].append({
                        "position": pos,
                        "true": true_aa,
                        "topk": topk_tokens
                    })
                
                samples.append(sample)
    
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--limit", type=int, help="Limit number of sequences for quick evaluation")
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint)
    config = DictConfig(checkpoint["config"])
    
    if args.limit:
        config.data.limit = args.limit
    
    # Get device
    device = get_device(config)
    
    # Load data
    print("Loading data...")
    train_seqs, val_seqs = load_and_split(
        path=config.data.fasta_path,
        train_frac=config.data.train_split,
        min_len=config.data.min_len,
        limit=config.data.limit,
        seed=config.seed
    )
    
    # Create validation loader
    _, val_loader = create_dataloaders(train_seqs, val_seqs, config)
    
    # Create model and load weights
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    
    # Evaluate
    print(f"\nLoaded checkpoint from epoch {checkpoint['epoch']}")
    print("Evaluating model...")
    
    metrics = evaluate_model(model, val_loader, device, config)
    
    print(f"\nEvaluation Results:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Masked Token Accuracy: {metrics['masked_acc']:.4f}")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
    print(f"  Total Masked Tokens: {metrics['masked_tokens']}")
    
    # Sample predictions
    print("\nSample Predictions:")
    samples = sample_predictions(model, val_loader, device, config)
    
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"  Original: {sample['original_seq']}")
        for pred in sample['predictions']:
            print(f"  Pos {pred['position']:2d} true={pred['true']} "
                  f"top5={pred['topk']}")


if __name__ == "__main__":
    main()