import torch
import torch.nn.functional as F
import argparse
import json
import os

from tokenizer import MIDITokenizer
from model import MIDITransformer


def top_k_top_p_filtering(logits, top_k=0, top_p=0.9, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    
    Args:
        logits: logits distribution shape (vocab_size,)
        top_k: keep only top k tokens with highest probability (top-k filtering)
        top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering)
    """
    assert logits.dim() == 1
    
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    return logits


@torch.no_grad()
def generate(
    model: MIDITransformer,
    tokenizer: MIDITokenizer,
    prompt = None,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generate a sequence autoregressively.
    
    Args:
        model: Trained MIDITransformer model
        tokenizer: MIDITokenizer instance
        prompt: Initial token sequence (optional)
        max_length: Maximum length to generate
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Nucleus sampling parameter
        device: Device to run on
        
    Returns:
        Generated token sequence
    """
    model.eval()
    model.to(device)
    
    # Initialize with BOS token if no prompt
    if prompt is None:
        generated = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
    else:
        generated = prompt.to(device)
        if generated.dim() == 2:
            generated = generated.unsqueeze(0)
        # batch dim = 1
    
    vocab_size = list(tokenizer.vocab_size.values())
    for _ in range(max_length):
        # Get model predictions
        # Only use the last max_seq_length tokens as input
        input_seq = generated[:, -model.max_seq_length:]
        logits = model(input_seq)

        offset = 0
        next_token_full = torch.tensor([], dtype=torch.long, device=input_seq.device)
        for idx, siz in enumerate(vocab_size):
            cur_logit = logits[..., offset: offset + siz]
            offset += siz

            # Get logits for the last position
            next_token_logits = cur_logit[0, -1, :] / temperature
        
            # Apply top-k and top-p filtering
            _top_k = min(siz//2, top_k)
            if(siz <= 10):
                _top_k = siz
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=_top_k, top_p=top_p)
        
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            next_token_full = torch.cat([next_token_full, next_token])

        generated = torch.cat([generated, next_token_full.unsqueeze(0).unsqueeze(0)], dim=1)        
        if next_token_full[0] == tokenizer.eos_token_id[0]: # special token
            print("\nGenerated:", next_token_full)
            print("\nEOS:", tokenizer.eos_token_id)
            break        
    
    return generated[0].cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description='Generate MIDI using trained model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='generated.mid', help='Output MIDI file path')
    parser.add_argument('--prompt_midi', type=str, default=None, help='Optional MIDI file to use as prompt/seed')
    parser.add_argument('--prompt_length', type=int, default=None, help='Number of tokens to use from prompt (default: all)')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling parameter')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    # this is what model will hear before regressive generation
    parser.add_argument('--piano_channels', type=str, default='0, 1, 2, 3, 4, 5', help='Comma-separated MIDI channels for prompt (default: 0)')
    
    args = parser.parse_args()
    
    # Parse piano channels
    piano_channels = [int(c.strip()) for c in args.piano_channels.split(',')]
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer config
    checkpoint_dir = os.path.dirname(args.checkpoint)
    tokenizer_config_path = os.path.join(checkpoint_dir, 'tokenizer_config.json')
    
    if not os.path.exists(tokenizer_config_path):
        print(f"Tokenizer config not found at {tokenizer_config_path}")
        return
    
    with open(tokenizer_config_path, 'r') as f:
        tokenizer_config = json.load(f)
    
    # Initialize tokenizer
    # 这很诡异，你知道吗
    tokenizer = MIDITokenizer()
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found at {args.checkpoint}")
        return
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Initialize model
    # 这个参数要加载吗
    model = MIDITransformer(
        vocab_size=list(tokenizer.vocab_size.values()),
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=512 * 4,
        max_seq_length=512,
        dropout=0.1,
        pad_token_id=0
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Load prompt if provided
    prompt = None
    if args.prompt_midi is not None:
        if not os.path.exists(args.prompt_midi):
            print(f"Prompt MIDI file not found: {args.prompt_midi}")
            return
        
        print(f"\nLoading prompt from: {args.prompt_midi}")
        prompt_tokens = tokenizer.encode_midi(args.prompt_midi, piano_channels=piano_channels)
        
        # Use specified length or all tokens
        if args.prompt_length is not None:
            prompt_tokens = prompt_tokens[:args.prompt_length]
            print(f"Using first {len(prompt_tokens)} tokens as prompt")
        else:
            print(f"Using all {len(prompt_tokens)} tokens as prompt")
        
        # Show prompt preview
        print("\nPrompt sequence preview:")
        for i, tid in enumerate(prompt_tokens[:10]):
            token_name = []
            for idx, key in enumerate(tokenizer.vocab_size):
                token_name.append(tokenizer.id_to_token[key][tid[idx]])
            
            print(f"  {i}: {token_name}")
        if len(prompt_tokens) > 10:
            print(f"  ... ({len(prompt_tokens) - 10} more tokens)")
        
        prompt = torch.tensor(prompt_tokens, dtype=torch.long)
    
    # Generate
    print(f"\nGenerating with temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    if prompt is not None:
        print(f"Starting from {len(prompt)} prompt tokens")
    else:
        print("Starting from scratch (no prompt)")
    
    generated_tokens = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )
    
    print(f"Generated {len(generated_tokens)} tokens")
    
    # Decode to MIDI
    print(f"Saving to {args.output}")
    tokenizer.decode_to_midi(generated_tokens, args.output)
    print("Done!")


if __name__ == '__main__':
    main()
