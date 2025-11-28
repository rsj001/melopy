import torch
import torch.nn.functional as F
import argparse
from types import SimpleNamespace
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
    temperature: list[float] = [1.1,1.1,1.1,1.1],
    top_k: list[int] = [10, 5, 10, 20],
    top_p: list[float] = [0.6, 0.8, 0.8, 0.6],
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
    assert len(temperature) == len(top_k) == len(top_p) == len(tokenizer.categories)
    
    # Initialize with BOS token if no prompt
    if prompt is None:
        generated = torch.tensor([[tokenizer.bos_token]], dtype=torch.long, device=device)
    else:
        generated = prompt.to(device)
        if generated.dim() == 2:
            generated = generated.unsqueeze(0)
        # batch dim = 1
    
    tensor_eos_token = torch.tensor(tokenizer.eos_token, device=device)
    for _ in range(max_length):
        # Get model predictions
        # Only use the last max_seq_length tokens as input
        input_seq = generated[:, -model.max_seq_length:]
        logits_full = model(input_seq)

        next_token_full = torch.tensor([], dtype=torch.long, device=device)

        for idx, logits in enumerate(logits_full):
            # Get logits for the last position
            next_token_logits = logits[0, -1, :] / temperature[idx]
        
            # Apply top-k and top-p filtering
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k[idx], top_p=top_p[idx])
        
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_full = torch.cat([next_token_full, next_token])

        if (tensor_eos_token == next_token_full).sum() > 0: # at least one eos HARDCODE
            print("\nCUR TOKEN:", next_token_full)
            print("\nSTD EOS:", tokenizer.eos_token)
            break
        generated = torch.cat([generated, next_token_full.unsqueeze(0).unsqueeze(0)], dim=1)        
    
    return generated[0][1:].cpu().tolist()


def GenerationWorkflow(use_parser: bool = True, user_args: dict = {}, preload_model: MIDITransformer | None = None, preload_prompt: torch.Tensor | None = None):
    if use_parser:
        parser = argparse.ArgumentParser(description='Generate MIDI using trained model')
        
        num_tasks = 7 # HARDCODED for Octuple
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v3', help='Directory of checkpoints')
        parser.add_argument('--checkpoint_name', type=str, default='best_model.pt', help='Filename of model checkpoint')
        parser.add_argument('--output', type=str, default='generated.mid', help='Output MIDI file path')
        parser.add_argument('--prompt_midi', type=str, default=None, help='Optional MIDI file to use as prompt/seed')
        parser.add_argument('--prompt_length', type=int, default=None, help='Number of tokens to use from prompt (default: all)')
        parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length to generate')
        parser.add_argument('--temperature', nargs=num_tasks, type=float, default=[1.4,1.1,1.1,1.2], help='Sampling temperature')
        parser.add_argument('--top_k', nargs=num_tasks, type=int, default=[10, 5, 10, 20], help='Top-k sampling parameter for 4 Heads')
        parser.add_argument('--top_p', nargs=num_tasks, type=float, default=[0.6, 0.8, 0.8, 0.6], help='Nucleus sampling parameter for 4 Heads')
        
        parser.add_argument('--seed', type=int, default=None, help='Random seed')
        # this is what model will hear before regressive generation
        parser.add_argument('--piano_channels', type=str, default='0, 1, 2, 3, 4, 5', help='Comma-separated MIDI channels for prompt (default: 0)')
        
        args = parser.parse_args()
    else:
        args = user_args
        default = {
            "checkpoint_dir": "checkpoints",
            "checkpoint_name": "best_model.pt",
            "output": "generated.mid",
            "prompt_midi": None,
            "prompt_length": None,
            "max_length": 512,
            "temperature": [1.0,1.0,1.0,1.0],
            "top_k": [10, 5, 10, 20],
            "top_p": [0.6, 0.8, 0.8, 0.6],
            "seed": None,
            "piano_channels": '0, 1, 2, 3, 4, 5'
        }
        for key, val in default.items():
            if key not in args:
                args[key] = val
        args = SimpleNamespace(**args)
    
    # Parse piano channels
    piano_channels = [int(c.strip()) for c in args.piano_channels.split(',')]
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer config
    # tokenizer_config_path = os.path.join(args.checkpoint_dir, 'tokenizer_config.json')
    
    # if not os.path.exists(tokenizer_config_path):
    #     print(f"Tokenizer config not found at {tokenizer_config_path}")
    #     return
    # with open(tokenizer_config_path, 'r') as f:
    #     tokenizer_config = json.load(f)
    # Initialize tokenizer
    # 这很诡异，你知道吗
    tokenizer = MIDITokenizer()

    if preload_model is not None:
        model = preload_model
    else:
        # Load checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            return
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        config_path = os.path.join(args.checkpoint_dir, "model_config.json")
        if not os.path.exists(config_path):
            print(f"Checkpoint config {config_path} not found. Cannot load model.")
            return 

        model_config = json.load(open(config_path))
        # 重新实例化 model
        # TODO config SAFE?
        model = MIDITransformer(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Load prompt if provided
    if preload_prompt:
        prompt = preload_prompt
    else:
        prompt = None
        if args.prompt_midi is not None:
            if not os.path.exists(args.prompt_midi):
                print(f"Prompt MIDI file not found: {args.prompt_midi}")
                return
            
            print(f"\nLoading prompt from: {args.prompt_midi}")
            prompt_tokens = tokenizer.encode_midi(args.prompt_midi).ids
            prompt_tokens.insert(0, tokenizer.bos_token)
            # Use specified length or all tokens
            if args.prompt_length is not None:
                prompt_tokens = prompt_tokens[:args.prompt_length]
                print(f"Using first {len(prompt_tokens)} tokens as prompt")
            else:
                print(f"Using all {len(prompt_tokens)} tokens as prompt")
            
            for i in prompt_tokens:
                print(i)
            # Show prompt preview
            # print("\nPrompt sequence preview:")
            # for i, tid in enumerate(prompt_tokens[:10]):
            #     token_name = []
            #     for idx, key in enumerate(tokenizer.vocab_sizes):
            #         token_name.append(tokenizer.id_to_token[idx][tid[idx]])
                
            #     print(f"  {i}: {token_name}")
            # if len(prompt_tokens) > 10:
            #     print(f"  ... ({len(prompt_tokens) - 10} more tokens)")
            
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
    
    # print("\nSequence preview:")
    # for i, tid in enumerate(generated_tokens):
    #     token_name = []
    #     for idx, key in enumerate(tokenizer.vocab_sizes):
    #         token_name.append(tokenizer.id_to_token[idx][tid[idx]])
    #     print(f"  {i}: {token_name}")
                
                
    print(f"Generated {len(generated_tokens)} tokens, saving to {args.output}. ")
    print(generated_tokens)
    tokenizer.decode_to_midi(generated_tokens, args.output)

if __name__ == '__main__':
    GenerationWorkflow()
