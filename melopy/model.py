import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

# 一个带正则化系数 ln x 的线性组合
# 仍然存在模型刻意降低 loss 大的参数的风险
class UncertaintyLossWrapper(nn.Module):
    def __init__(self, num_tasks, device):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, device=device))
        
    def forward(self, losses):
        return sum(torch.exp(-self.log_vars) * losses + self.log_vars)
    

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
        sin, cos = self.get_buffer("sin")[:seq_len, :], self.get_buffer("cos")[:seq_len, :]
        sin, cos = sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x_out = torch.zeros_like(x)
        x_out[..., ::2] = x1 * cos - x2 * sin
        x_out[..., 1::2] = x1 * sin + x2 * cos
        return x_out

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE and causal masking."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, max_seq_len: int):
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

        # RoPE
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
    
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer decoder block with causal self-attention."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, max_seq_len: int):
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
    
class FiveFusionLinearPooling(nn.Module):
    def __init__(self, vector_dim):
        super().__init__()
        self.num_vec = 5 # HARDCODED FIVE
        self.vector_dim = vector_dim
        self.total_dim = self.num_vec * vector_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(self.total_dim, 2 * vector_dim),
            nn.ReLU(),
            nn.Linear(2 * vector_dim, vector_dim)
        )
    def forward(self, vectors: tuple[torch.Tensor, ...] | list[torch.Tensor]):
        # vectors: [v1, v2, v3, v4, v5], 每个形状为 [batch_size, vector_dim]
        concatenated = torch.cat(vectors, dim=-1)  # [batch_size, 5 * vector_dim]
        fused = self.fusion_net(concatenated)  # [batch_size, output_dim]
        return fused

class MIDITransformer(nn.Module):
    """
    Decoder-only GPT model for MIDI generation.
    """
    
    def __init__(
        self,
        vocab_size: List[int],
        # 为了更好地coding，这里vocab_size改为dict类型，表示不同类别的token数量
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
        # 这里是对每一个 token_dim 的 pad
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        
        # Token embedding
        self.embeds = nn.ModuleList([
            nn.Embedding(siz, d_model) for siz in self.vocab_size
        ])
        self.vocab_size_full = sum(self.vocab_size)
        self.linear_pooling = FiveFusionLinearPooling(d_model)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, max_seq_len=max_seq_length)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size_full, bias=False)
        
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
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len, token_dim)
            targets: (batch_size, seq_len, token_dim) - optional, for computing loss
            
        Returns:
            logits or (logits, loss)
        """
        batch_size, seq_len, token_dim = input_ids.shape
        
        # Token embeddings 
        
        # A TOO SIMPLE POOLING 可能导致什么问题呢？
        # x = sum(self.embeds[i](input_ids[..., i]) for i in range(token_dim)) / token_dim
        # Linear pooling
        x = self.linear_pooling([self.embeds[i](input_ids[..., i]) for i in range(token_dim)])
        
        
        # PE (deprecated, in favor of RoPE)
        x = x + self.pos_embedding[:, :seq_len, :]

        x = self.dropout(x)
        
        # Get causal mask
        mask = self.get_buffer("causal_mask")[:, :, :seq_len, :seq_len]
        
        # Apply transformer blocks
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, mask)
        
        # Final layer norm and output projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        if targets is not None:
            logits_T = logits.view(-1, self.vocab_size_full)
            targets_T = targets.view(-1, token_dim)
            offset = 0
            losses = []
            for i, v in enumerate(self.vocab_size):
                lg = logits_T[:, offset:offset+v]
                lb = targets_T[:, i]
                offset += v
                if (lb != self.pad_token_id).sum() == 0:
                    loss = torch.tensor(0., device=lg.device, dtype=lg.dtype)
                else:
                    loss = F.cross_entropy(lg, lb, ignore_index=self.pad_token_id)
                losses.append(loss)
            return (logits, torch.stack(losses))
        else:
            return logits
    
    def get_num_params(self):
        """Get number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
