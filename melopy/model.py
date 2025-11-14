import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module): # hardcode for token ranges, remember to update if tokenizer changes
    def __init__(self, vocab_size, d_model, padding_idx):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=padding_idx) # TOKEN_233 is MASK
        self.type_embed = nn.Embedding(5, d_model, padding_idx=0)  # pad, pitch on/off, time, velocity

    def forward(self, token_ids):
        """
        token_ids: [batch, seq_len]
        """
        type_ids = torch.zeros_like(token_ids)
        # NOTE: THIS IS HARDCODED RANGES!!
        mask1 = (token_ids >= 3) & (token_ids <= 51)      # Note On
        mask2 = (token_ids >= 52) & (token_ids <= 100)    # Note Off
        mask3 = (token_ids >= 101) & (token_ids <= 200)   # Time
        mask4 = (token_ids >= 201) & (token_ids <= 232)   # Velocity
        type_ids[mask1] = 1
        type_ids[mask2] = 2
        type_ids[mask3] = 3
        type_ids[mask4] = 4
        return self.token_embed(token_ids) + self.type_embed(type_ids)

class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding."""

    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.base = base

        position = torch.arange(max_seq_len, dtype=torch.float32)
        dim_half = torch.arange(0, dim // 2, dtype=torch.float32)
        freqs = 1.0 / (base ** (dim_half / (dim // 2)))
        angles = torch.einsum('p,d->pd', position, freqs)

        sin, cos = torch.sin(angles), torch.cos(angles)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, heads, seq_len, head_dim)
        """
        seq_len = x.size(-2)
        sin, cos = self.sin[:seq_len, :], self.cos[:seq_len, :]
        sin, cos = sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x_out = torch.zeros_like(x)
        x_out[..., ::2] = x1 * cos - x2 * sin
        x_out[..., 1::2] = x1 * sin + x2 * cos
        return x_out

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE and causal masking."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or None
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # === Apply RoPE ===
        q = self.rope(q)
        k = self.rope(k)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer decoder block with causal self-attention."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout, max_seq_len=max_seq_len)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class MIDITransformer(nn.Module):
    """
    Decoder-only GPT model for MIDI generation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        
        # Token embedding
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx=pad_token_id)
        # self.token_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_token_id) # MASK TOKEN
        
        # Positional encoding
        # self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, max_seq_len=max_seq_length)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        # Create causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_length, max_seq_length)).view(1, 1, max_seq_length, max_seq_length)
        )
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
        
        # Initialize positional embeddings
        # nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len)
            targets: (batch_size, seq_len) - optional, for computing loss
            
        Returns:
            logits or (logits, loss)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings + positional embeddings
        x = self.token_embedding(input_ids)
        # x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Get causal mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        
        # Apply transformer blocks
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, mask)
        
        # Final layer norm and output projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=self.pad_token_id
            )
        
        return (logits, loss) if loss is not None else logits
    
    def get_num_params(self):
        """Get number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
