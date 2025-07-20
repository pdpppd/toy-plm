"""
Training script for the protein masked language model.
"""

import argparse
import os
import time
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .utils import (
    set_seed, load_config, update_config_with_args, get_device,
    save_checkpoint, compute_metrics, log_metrics, create_output_dir,
    format_time, WarmupLinearDecay
)
from .fasta import load_and_split, get_sequence_stats
from .dataset import create_dataloaders
from .model import create_model
from .masking import mask_batch, get_masking_stats


def train_epoch(model, train_loader, optimizer, scheduler, scaler, loss_fn, device, config, epoch, global_step):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_masked_tokens = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        batch = batch.to(device)  # (B, T)
        
        # Apply dynamic masking
        masked_inputs, labels = mask_batch(
            batch,
            mask_prob=config.masking.prob,
            mask_ratio=config.masking.mask_ratio,
            random_ratio=config.masking.random_ratio
        )
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'
        with torch.amp.autocast(device_type, enabled=config.train.amp and device.type == 'cuda'):
            logits = model(masked_inputs)  # (B, T, V)
            
            # Compute loss (only on masked positions)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),  # (B*T, V)
                labels.view(-1)  # (B*T,)
            )
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(logits, labels)
            total_loss += loss.item()
            total_acc += metrics["masked_token_acc"]
            total_masked_tokens += metrics["masked_tokens"]
            num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{metrics['masked_token_acc']:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # Log periodically
        if global_step % config.train.log_interval == 0:
            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            current_lr = scheduler.get_last_lr()[0]
            
            log_metrics(
                step=global_step,
                epoch=epoch,
                metrics={
                    "train_loss": avg_loss,
                    "train_acc": avg_acc,
                    "masked_tokens": total_masked_tokens,
                },
                lr=current_lr,
                log_file=os.path.join(config.logging.log_dir, f"{config.logging.run_name}.log")
            )
        
        global_step += 1
    
    return global_step, total_loss / num_batches, total_acc / num_batches


def evaluate(model, val_loader, loss_fn, device, config):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_masked_tokens = 0
    num_batches = 0
    
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
        "val_loss": avg_loss,
        "val_acc": avg_acc,
        "val_ppl": perplexity,
        "val_masked_tokens": total_masked_tokens
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--limit", type=int, help="Limit number of sequences for debugging")
    parser.add_argument("--debug_overfit", action="store_true", help="Debug mode with overfitting")
    args = parser.parse_args()
    
    # Load and update config
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    # Debug overfit mode
    if args.debug_overfit:
        config.data.limit = 10
        if not hasattr(args, 'epochs') or args.epochs is None:
            config.train.epochs = 20  # Only set default if not specified by user
        config.train.batch_size = 4
        print("Debug overfit mode enabled")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Get device
    device = get_device(config)
    
    # Create output directory
    output_dir = create_output_dir(config)
    
    # Load and split data
    print("Loading data...")
    train_seqs, val_seqs = load_and_split(
        path=config.data.fasta_path,
        train_frac=config.data.train_split,
        min_len=config.data.min_len,
        limit=config.data.limit,
        seed=config.seed
    )
    
    # Print data statistics
    train_stats = get_sequence_stats(train_seqs)
    val_stats = get_sequence_stats(val_seqs)
    print(f"Train: {train_stats}")
    print(f"Val: {val_stats}")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(train_seqs, val_seqs, config)
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay
    )
    
    # Calculate total steps
    total_steps = len(train_loader) * config.train.epochs
    lr_scheduler_fn = WarmupLinearDecay(total_steps, config.train.warmup_steps)
    scheduler = LambdaLR(optimizer, lr_scheduler_fn)
    
    # Mixed precision scaler (compatible with MPS/CUDA/CPU)
    if device.type == "cuda":
        scaler = torch.amp.GradScaler('cuda', enabled=config.train.amp)
    else:
        # MPS and CPU don't support amp scaling
        scaler = torch.amp.GradScaler('cpu', enabled=False)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Training loop
    print(f"Starting training for {config.train.epochs} epochs...")
    start_time = time.time()
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(1, config.train.epochs + 1):
        # Train
        global_step, train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn,
            device, config, epoch, global_step
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, loss_fn, device, config)
        
        # Log epoch results
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val: loss={val_metrics['val_loss']:.4f}, acc={val_metrics['val_acc']:.4f}, "
              f"ppl={val_metrics['val_ppl']:.2f}")
        print(f"  Time: {format_time(epoch_time)}")
        
        # Save checkpoint
        if epoch % config.train.save_every == 0:
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
            
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "config": config.to_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_metrics": val_metrics,
            }
            
            checkpoint_path = os.path.join(output_dir, f"epoch_{epoch}.pt")
            save_checkpoint(checkpoint, checkpoint_path)
            
            if is_best:
                best_path = os.path.join(output_dir, "best_model.pt")
                save_checkpoint(checkpoint, best_path)
        
        start_time = time.time()
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {output_dir}")


if __name__ == "__main__":
    main()