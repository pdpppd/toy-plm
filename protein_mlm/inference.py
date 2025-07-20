"""
Inference utilities for masked residue prediction.
"""

import torch
import re

from .utils import load_checkpoint, get_device, DictConfig
from .model import create_model
from .vocab import tokenize, IDX2TOK, TOK2IDX, MASK_ID


def load_model_for_inference(checkpoint_path):
    """Load model from checkpoint for inference."""
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    config = DictConfig(checkpoint["config"])
    
    # Get device
    device = get_device(config)
    
    # Create model and load weights
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    
    return model, config, device


def predict_masked(model, sequence, device, config, mask_positions=None, topk=5):
    """
    Predict masked residues in a protein sequence.
    
    Args:
        model: Trained model
        sequence: Protein sequence string (can contain [MASK] tokens)
        device: Device to run inference on
        config: Model configuration
        mask_positions: List of positions to mask (0-indexed in original sequence)
        topk: Number of top predictions to return
        
    Returns:
        List of predictions for each masked position
    """
    original_seq = sequence
    
    # If mask_positions provided, create masked sequence
    if mask_positions is not None:
        seq_list = list(sequence)
        for pos in mask_positions:
            if pos < len(seq_list):
                seq_list[pos] = "[MASK]"
        sequence = "".join(seq_list)
    
    # Find [MASK] tokens in sequence
    mask_pattern = re.compile(r'\[MASK\]')
    mask_matches = list(mask_pattern.finditer(sequence))
    
    if not mask_matches:
        return []  # No masks found
    
    # Replace [MASK] with actual MASK tokens for tokenization
    clean_seq = sequence.replace('[MASK]', 'M')  # Temporary replacement
    
    # Tokenize
    token_ids = tokenize(clean_seq, config.data.max_len)
    
    # Find positions of mask tokens and replace with MASK_ID
    masked_positions = []
    token_list = token_ids.copy()
    
    # Map mask positions from string to token positions
    char_to_token_map = {}
    token_idx = 1  # Skip CLS token
    for char_idx, char in enumerate(clean_seq):
        if token_idx < len(token_list) - 1:  # Skip SEP token
            char_to_token_map[char_idx] = token_idx
            token_idx += 1
    
    for match in mask_matches:
        start_pos = match.start()
        if start_pos in char_to_token_map:
            token_pos = char_to_token_map[start_pos]
            token_list[token_pos] = MASK_ID
            masked_positions.append(token_pos)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor([token_list], dtype=torch.long).to(device)
    
    # Run inference
    with torch.no_grad():
        device_type = device.type if device.type in ['cuda', 'cpu'] else 'cpu'
        with torch.amp.autocast(device_type, enabled=config.train.amp and device.type == 'cuda'):
            logits = model(input_tensor)  # (1, T, V)
            probs = torch.softmax(logits, dim=-1)
    
    # Extract predictions for masked positions
    predictions = []
    for i, token_pos in enumerate(masked_positions):
        if mask_positions is not None and i < len(mask_positions):
            original_pos = mask_positions[i]
            true_residue = original_seq[original_pos] if original_pos < len(original_seq) else "?"
        else:
            original_pos = token_pos
            true_residue = "?"
        
        # Get top-k predictions for this position
        pos_probs = probs[0, token_pos]  # (V,)
        topk_probs, topk_indices = pos_probs.topk(topk)
        
        topk_predictions = []
        for idx, prob in zip(topk_indices, topk_probs):
            token = IDX2TOK[idx.item()]
            if not token.startswith('['):  # Skip special tokens in output
                topk_predictions.append((token, prob.item()))
        
        predictions.append({
            "position": original_pos,
            "true": true_residue,
            "topk": topk_predictions
        })
    
    return predictions


def predict_random_positions(model, sequence, device, config, num_positions=3, topk=5):
    """
    Randomly mask positions in a sequence and predict them.
    
    Args:
        model: Trained model
        sequence: Protein sequence string
        device: Device to run inference on
        config: Model configuration
        num_positions: Number of positions to randomly mask
        topk: Number of top predictions to return
        
    Returns:
        List of predictions for each masked position
    """
    import random
    
    # Choose random positions to mask
    valid_positions = list(range(len(sequence)))
    if len(valid_positions) == 0:
        return []
    
    num_to_mask = min(num_positions, len(valid_positions))
    mask_positions = random.sample(valid_positions, num_to_mask)
    
    return predict_masked(model, sequence, device, config, mask_positions, topk)


def interactive_prediction(checkpoint_path):
    """Interactive mode for making predictions."""
    print("Loading model...")
    model, config, device = load_model_for_inference(checkpoint_path)
    print("Model loaded. Ready for predictions!")
    
    print("\nUsage:")
    print("  Enter a protein sequence with [MASK] tokens")
    print("  Example: MKTAYIAK[MASK]RQISFVKSHFS")
    print("  Type 'quit' to exit\n")
    
    while True:
        sequence = input("Enter sequence: ").strip()
        
        if sequence.lower() in ['quit', 'exit', 'q']:
            break
        
        if not sequence:
            continue
        
        try:
            predictions = predict_masked(model, sequence, device, config)
            
            if not predictions:
                print("No [MASK] tokens found in sequence.")
                continue
            
            print("\nPredictions:")
            for pred in predictions:
                print(f"  Position {pred['position']:2d}: true={pred['true']} "
                      f"top5={pred['topk']}")
            print()
        
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("Goodbye!")


def main():
    """Command line interface for inference."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--sequence", help="Protein sequence to predict")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_prediction(args.checkpoint)
    elif args.sequence:
        model, config, device = load_model_for_inference(args.checkpoint)
        predictions = predict_masked(model, args.sequence, device, config, topk=args.topk)
        
        if predictions:
            print("Predictions:")
            for pred in predictions:
                print(f"  Position {pred['position']:2d}: true={pred['true']} "
                      f"top{args.topk}={pred['topk']}")
        else:
            print("No [MASK] tokens found in sequence.")
    else:
        print("Please provide --sequence or use --interactive mode")


if __name__ == "__main__":
    main()