import torch
import torch.nn as nn
import torch.nn.functional as F

from x_transformers import Decoder, TransformerWrapper, AutoregressiveWrapper

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
        max_seq_length: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.model = TransformerWrapper(
            num_tokens = vocab_size,
            max_seq_len = max_seq_length,
            emb_dropout = dropout,
            attn_layers = Decoder(
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
        )
    
    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len)
            targets: (batch_size, seq_len) - optional, for computing loss
            
        Returns:
            logits or (logits, loss)
        """
        batch_size, seq_len = input_ids.shape
        
        
        self.mask_prob = 0.14
        self_attn_kv_mask = None
        if self.mask_prob > 0. and targets is not None:
            rand = torch.randn(input_ids.shape, device = input_ids.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max
            num_mask = min(int(seq_len * self.mask_prob), seq_len - 1)
            indices = rand.topk(num_mask, dim = -1).indices
            mask = ~torch.zeros_like(input_ids).scatter(1, indices, 1.).bool()
            self_attn_kv_mask = mask
        logits = self.model(input_ids, self_attn_kv_mask = self_attn_kv_mask)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=self.pad_token_id)
        return (logits, loss)
        
    def get_num_params(self):
        """Get number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
