import torch
import torch.nn as nn
import torch.nn.functional as F

from x_transformers import Decoder, AutoregressiveWrapper
from typing import List

# 一个带正则化系数 ln x 的线性组合
# 仍然存在模型刻意降低 loss 大的参数的风险
class UncertaintyLossWrapper(nn.Module):
    def __init__(self, num_tasks, device):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, device=device))
        
    def forward(self, losses):
        return sum(torch.exp(-self.log_vars) * losses + self.log_vars)

class FusionLinearPooling(nn.Module):
    def __init__(self, vector_dim, num_vec):
        super().__init__()
        self.num_vec = num_vec
        self.vector_dim = vector_dim
        self.total_dim = self.num_vec * vector_dim
        self.fusion_net = nn.Linear(self.total_dim, vector_dim)
        # Just Linear. Trust me.
    def forward(self, vectors: tuple[torch.Tensor, ...] | list[torch.Tensor]):
        concatenated = torch.cat(vectors, dim=-1)
        fused = self.fusion_net(concatenated)
        return fused

class MIDITransformer(nn.Module):
    """
    Decoder-only GPT model for MIDI generation.
    """
    
    def __init__(
        self,
        vocab_size: List[int],
        # 为了更好地coding，这里vocab_size改为list类型，表示不同类别的token数量
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_length: int = 512,
        dropout: float = 0.12,
        pad_token_id: int = 0
        # NOTE !! 这里是对每一个 token_dim 的 pad
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.vocab_size_full = sum(self.vocab_size)
        self.num_token_type = len(self.vocab_size)
        
        # Token embedding
        self.embeds = nn.ModuleList([
            nn.Embedding(siz, d_model) for siz in self.vocab_size
        ])
        
        self.linear_pooling = FusionLinearPooling(d_model, self.num_token_type)
        self.dropout_layer = nn.Dropout(dropout)
        
        # attention with causal mask
        # attention is all you need
        self.attn_layer = Decoder(
            dim=d_model,
            depth=num_layers,
            heads=num_heads,
            attn_num_mem_kv = 16,
            use_scalenorm = True,
            ff_glu = True,
            attn_flash = True,
            rotary_pos_emb = True,
            rotary_xpos = True,
            layer_dropout = dropout,
            attn_dropout = dropout,
            ff_dropout = dropout,
        )
        
        self.lm_head = nn.ModuleList([
            nn.Linear(d_model, size, bias=False)
            for size in self.vocab_size
        ])
        
        # self.autoregressive = AutoregressiveWrapper(
        #     self.backbone,
        #     mask_prob = 0.15,  # in paper, they use 15%, same as BERT
        #     pad_value = pad_token_id,
        #     ignore_index = pad_token_id
        # ).cuda()
    
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
        
        # Token embeddings - Linear pooling
        x = self.linear_pooling([self.embeds[i](input_ids[..., i]) for i in range(token_dim)])
        x = self.dropout_layer(x)
        
        # 随机掩码
        self.mask_prob = 0.15

        rand = torch.randn(batch_size, seq_len, device = x.device)
        rand[:, 0] = -torch.finfo(rand.dtype).max
        num_mask = min(int(seq_len * self.mask_prob), seq_len - 1)
        indices = rand.topk(num_mask, dim = -1).indices
        mask = ~torch.zeros_like(rand).scatter(1, indices, 1.).bool()
        
        x = self.attn_layer(x, self_attn_kv_mask = mask)
        logits = [head(x) for head in self.lm_head]
        
        
        if targets is None:
            return logits
        
        targets_T = targets.view(-1, token_dim)
        losses = [F.cross_entropy(lg.view(-1, self.vocab_size[i]), targets_T[:, i], ignore_index=self.pad_token_id) for i, lg in enumerate(logits)]
        return (logits, torch.stack(losses))
        
    def get_num_params(self):
        """Get number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
