"""
Educational skeleton for the decoder-only Transformer.
Implement the TODOs in each forward() to learn the key pieces.
Reference: ../transformer.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Any
from dataclasses import dataclass

# helper
ACT2FN = {
    'relu': F.relu,
    'gelu': F.gelu,
    'silu': F.silu,
    'swish': F.silu,
}

@dataclass
class AttentionConfig:
    D: int = 768
    layer_idx: Optional[int] = None
    head_dim: int = 64
    causal: bool = True
    device: str = "cuda"

class Attention(nn.Module):  # BSD -> BSD
    def __init__(self,
                 config: AttentionConfig):
        super().__init__()
        self.D = config.D
        self.head_dim = config.head_dim
        assert self.D % self.head_dim == 0
        self.nheads = self.D // self.head_dim
        self.Wq = nn.Linear(self.D, self.D)
        self.Wk = nn.Linear(self.D, self.D)
        self.Wv = nn.Linear(self.D, self.D)
        self.causal = config.causal
        self.Wo = nn.Linear(self.D, self.D)
        self.device = config.device
        self.layer_idx = config.layer_idx

    def forward(self, x: torch.Tensor, kv_cache: Optional[Any] = None) -> torch.Tensor:
        # TODO: Implement multi-head self-attention.
        # Input x: [B, S, D]. Output: [B, S, D].
        # 1) Compute Q, K, V with self.Wq, self.Wk, self.Wv.
        # 2) Reshape to multi-head: [B, S, D] -> [B, nh, S, hd].
        # 3) If kv_cache is not None and layer_idx is set, call kv_cache.update(layer_idx, K, V)
        #    and read K, V from cache (slice by kv_cache.current_length).
        # 4) logits = (Q @ K^T) / scale, scale = sqrt(head_dim). Shapes: logits [B, nh, S, S].
        # 5) If causal: mask future positions (triu, diagonal=1) with -inf, then softmax.
        # 6) preout = attention_weights @ V, then concat heads and apply self.Wo.
        B, S, D = x.shape
        return torch.zeros(B, S, D, dtype=x.dtype, device=x.device)

@dataclass
class MLPConfig:
    D: int
    hidden_multiplier: int = 4
    act: str = 'swish'
    device: Optional[torch.device] = None

class MLP(nn.Module):
    """Token-wise: each position independently. D -> D."""

    def __init__(self,
                 config: MLPConfig):
        super().__init__()
        self.D = config.D
        self.device = config.device if config.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.up_proj = nn.Linear(self.D, self.D * config.hidden_multiplier)
        self.down_proj = nn.Linear(self.D * config.hidden_multiplier, self.D)
        self.act = ACT2FN[config.act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: One line: up_proj -> act -> down_proj. Input/output [B, S, D].
        return torch.zeros_like(x)

@dataclass
class LNConfig:
    D: int
    eps: float = 1e-9
    device: Optional[torch.device] = None

class LN(nn.Module):
    def __init__(self,
                 config: LNConfig):
        super().__init__()
        self.D = config.D
        self.eps = config.eps
        self.device = config.device if config.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_scale = nn.Parameter(torch.zeros(self.D))
        self.std_scale = nn.Parameter(torch.ones(self.D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: LayerNorm. x: [B, S, D].
        # Compute mean and std over last dim (use keepdim=True). Normalize (x - mean) / std,
        # then scale and shift with self.std_scale and self.mean_scale.
        return torch.zeros_like(x)

@dataclass
class TransformerLayerConfig:
    D: int
    device: Optional[torch.device] = None

class TransformerLayer(nn.Module):
    def __init__(self,
                 config: TransformerLayerConfig):
        super().__init__()
        self.D = config.D
        self.device = config.device if config.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        attn_config = AttentionConfig(D=self.D, device=self.device)
        mlp_config = MLPConfig(D=self.D, device=self.device)
        ln_config = LNConfig(D=self.D, device=self.device)

        self.attn = Attention(attn_config)
        self.mlp = MLP(mlp_config)
        self.ln1 = LN(ln_config)
        self.ln2 = LN(ln_config)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Any] = None) -> torch.Tensor:
        # TODO: Pre-norm + residual. Order: ln1 -> attn -> add to x; ln2 -> mlp -> add to x.
        return torch.zeros_like(x)

@dataclass
class PositionalEmbeddingConfig:
    max_seq_len: int
    D: int
    device: Optional[torch.device] = None

class PositionalEmbedding(nn.Module):
    def __init__(self,
                 config: PositionalEmbeddingConfig):
        super().__init__()
        self.device = config.device if config.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_embedding = nn.Parameter(torch.randn(config.max_seq_len, config.D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Add positional embedding to x. x: [B, S, D]. Return x + self.pos_embedding[:S].
        return torch.zeros_like(x)

@dataclass
class EmbeddingLayerConfig:
    vocab_size: int
    D: int
    device: Optional[torch.device] = None

class EmbeddingLayer(nn.Module):
    """Lookup table (embedding[input_ids]), not matmul."""

    def __init__(self,
                 config: EmbeddingLayerConfig):
        super().__init__()
        self.device = config.device if config.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Parameter(torch.randn(config.vocab_size, config.D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Lookup. Return self.embedding[x]. x is token ids, output [B, S, D].
        B, S = x.shape
        D = self.embedding.shape[1]
        return torch.zeros(B, S, D, dtype=self.embedding.dtype, device=x.device)

@dataclass
class UnembeddingLayerConfig:
    vocab_size: int
    D: int
    device: Optional[torch.device] = None

class UnembeddingLayer(nn.Module):
    def __init__(self,
                 config: UnembeddingLayerConfig):
        super().__init__()
        self.device = config.device if config.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.V = config.vocab_size
        self.unembedding = nn.Linear(config.D, self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Linear from hidden dim to vocab size. x: [B, S, D] -> logits [B, S, V].
        return self.unembedding(x)  # already one line; leave as-is or stub if you want to implement manually

@dataclass
class TransformerConfig:
    vocab_size: int
    depth: int = 8
    hidden_dim: int = 512
    max_seq_len: int = 16384
    device: Optional[torch.device] = None
    mtp: bool = False

class Transformer(nn.Module):
    def __init__(self,
                 config: TransformerConfig):
        super().__init__()
        self.depth = config.depth
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size

        emb_config = EmbeddingLayerConfig(vocab_size=config.vocab_size, D=config.hidden_dim, device=config.device)
        pos_emb_config = PositionalEmbeddingConfig(max_seq_len=config.max_seq_len, D=config.hidden_dim, device=config.device)
        unemb_config = UnembeddingLayerConfig(vocab_size=config.vocab_size, D=config.hidden_dim, device=config.device)

        self.emb = EmbeddingLayer(emb_config)
        self.pos_emb = PositionalEmbedding(pos_emb_config)
        self.unemb = UnembeddingLayer(unemb_config)
        self.mtp = config.mtp

        layer_config = TransformerLayerConfig(D=config.hidden_dim, device=config.device)
        self.layers = nn.ModuleList([TransformerLayer(layer_config) for _ in range(config.depth)])
        for i, layer in enumerate(self.layers):
            layer.attn.layer_idx = i

        if config.mtp:
            self.mtp_heads = nn.ModuleList([nn.Linear(self.hidden_dim, self.vocab_size) for _ in range(4)])

        self.device = config.device

    def forward(self, x: torch.Tensor, kv_cache: Optional[Any] = None) -> torch.Tensor:
        # TODO: Full forward. emb -> pos_emb (use kv_cache.current_length offset when kv_cache is set) -> layers -> unemb.
        # When kv_cache is not None: pos_emb only for new tokens: pos_embedding[pos_offset : pos_offset + x.size(1)].
        B, S = x.shape
        V = self.vocab_size
        return torch.zeros(B, S, V, dtype=next(self.parameters()).dtype, device=x.device)
