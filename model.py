"""
model.py - Transformer architecture for German -> English neural machine
translation, implementing the original "Attention Is All You Need" design
with a few configuration switches used by the ablation experiments.

Autograder-facing public surface (signatures preserved):
    scaled_dot_product_attention(Q, K, V, mask) -> (out, weights)
    MultiHeadAttention.forward(query, key, value, mask) -> Tensor
    PositionalEncoding.forward(x) -> Tensor
    make_src_mask(src, pad_idx) -> BoolTensor
    make_tgt_mask(tgt, pad_idx) -> BoolTensor
    Transformer.encode(src, src_mask) -> Tensor
    Transformer.decode(memory, src_mask, tgt, tgt_mask) -> Tensor
    Transformer().infer(src_sentence: str) -> str
"""

from __future__ import annotations

import copy
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Scaled dot-product attention (module-level, autograder hook)
# ============================================================================

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scaling: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    The `use_scaling` flag is for the §2.2 ablation (sqrt(d_k) on/off).

    Args:
        Q, K: (..., seq_q, d_k) and (..., seq_k, d_k)
        V:    (..., seq_k, d_v)
        mask: bool, broadcastable to (..., seq_q, seq_k); True = mask out.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scaling:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn


# ============================================================================
# Mask helpers (module-level, autograder hooks)
# ============================================================================

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """[B, S] -> [B, 1, 1, S] bool. True at padding positions."""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """[B, T] -> [B, 1, T, T] bool. True at pad OR future positions."""
    B, T = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)        # [B,1,1,T]
    causal = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=tgt.device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)                                  # [1,1,T,T]
    return pad_mask | causal


# ============================================================================
# Multi-Head Attention
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
    head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_scaling: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_scaling = use_scaling

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Stash for visualisation/diagnostics; cleared each forward.
        self.last_attn: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B,h,L,dk]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, h, L, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, h * dk)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))
        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask, use_scaling=self.use_scaling)
        out = self._merge_heads(out)
        out = self.W_o(self.dropout(out))
        # Detach so we don't keep a graph alive for diagnostics.
        self.last_attn = attn.detach()
        return out


# ============================================================================
# Positional Encoding (sinusoidal) + learned variant for §2.4 ablation
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, registered as a non-trainable buffer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Shape [1, max_len, d_model] for easy broadcast over batch.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learnable positional embedding for the §2.4 ablation."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        positions = torch.arange(L, device=x.device).unsqueeze(0)  # [1, L]
        return self.dropout(x + self.pe(positions))


# ============================================================================
# Position-wise Feed-Forward
# ============================================================================

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ============================================================================
# Encoder + Decoder layers (Post-LN, matching the original paper)
# ============================================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, use_scaling: bool = True) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling=use_scaling)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, use_scaling: bool = True) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling=use_scaling)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scaling=use_scaling)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop2(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.drop3(self.ffn(x)))
        return x


def _clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


# ============================================================================
# Full Transformer
# ============================================================================

class Transformer(nn.Module):
    """
    Sequence-to-sequence Transformer.

    All constructor arguments have defaults so that the autograder may call
    `Transformer()` with no positional arguments. When called this way the
    model attempts to load vocabularies from `vocab_path` and weights from
    `checkpoint_path` (downloading from Google Drive via gdown if a drive
    file id is provided through `weights_drive_id`).
    """

    DEFAULT_SRC_VOCAB = 8000
    DEFAULT_TGT_VOCAB = 6000

    def __init__(
        self,
        src_vocab_size: int = DEFAULT_SRC_VOCAB,
        tgt_vocab_size: int = DEFAULT_TGT_VOCAB,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 1,
        sos_idx: int = 2,
        eos_idx: int = 3,
        use_scaling: bool = True,
        pos_encoding: str = "sinusoidal",   # "sinusoidal" | "learned"
        checkpoint_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        weights_drive_id: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Optionally pull vocabs from disk so that infer() works after a
        # bare `Transformer()` instantiation in the autograder.
        self._tokenizer_de = None
        self._tokenizer_en = None
        self._src_vocab = None
        self._tgt_vocab = None
        if vocab_path is not None and os.path.isfile(vocab_path):
            try:
                from dataset import load_vocabs
                self._src_vocab, self._tgt_vocab = load_vocabs(vocab_path)
                src_vocab_size = len(self._src_vocab)
                tgt_vocab_size = len(self._tgt_vocab)
            except Exception:
                pass

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.use_scaling = use_scaling
        self.pos_encoding = pos_encoding

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        if pos_encoding == "learned":
            self.src_pos = LearnedPositionalEncoding(d_model, dropout, max_len)
            self.tgt_pos = LearnedPositionalEncoding(d_model, dropout, max_len)
        else:
            self.src_pos = PositionalEncoding(d_model, dropout, max_len)
            self.tgt_pos = PositionalEncoding(d_model, dropout, max_len)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout, use_scaling=use_scaling)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout, use_scaling=use_scaling)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

        # Optional checkpoint loading -- triggered when the autograder runs
        # `Transformer().to(device).infer(...)`. This block downloads weights
        # from Google Drive if needed and loads them.
        if checkpoint_path is None:
            checkpoint_path = "checkpoint.pt"
        if weights_drive_id and not os.path.isfile(checkpoint_path):
            try:
                import gdown
                gdown.download(id=weights_drive_id, output=checkpoint_path, quiet=True)
            except Exception:
                pass
        if os.path.isfile(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                state = ckpt.get("model_state_dict", ckpt)
                self.load_state_dict(state, strict=False)
            except Exception:
                pass

    # ----------------------------------------------------------------- init
    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------- autograder contract
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.tgt_pos(x)
        h = self.decoder(x, memory, src_mask, tgt_mask)
        return self.generator(h)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ------------------------------------------------------- inference path
    @torch.no_grad()
    def infer(self, src_sentence: str, max_len: int = 100) -> str:
        """Greedy German -> English translation for a single sentence."""
        device = next(self.parameters()).device
        self.eval()

        if self._src_vocab is None or self._tgt_vocab is None:
            return ""

        # Lazy-init German tokenizer.
        if self._tokenizer_de is None:
            import spacy
            self._tokenizer_de = spacy.load(
                "de_core_news_sm",
                disable=["parser", "ner", "tagger", "lemmatizer"],
            )

        toks = [t.text.lower() for t in self._tokenizer_de.tokenizer(src_sentence.strip()) if t.text.strip()]
        ids = [self.sos_idx] + self._src_vocab.encode(toks)[: self.max_len - 2] + [self.eos_idx]
        src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, self.pad_idx)
        memory = self.encode(src, src_mask)

        ys = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, self.pad_idx)
            logits = self.decode(memory, src_mask, ys, tgt_mask)
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())
            ys = torch.cat([ys, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == self.eos_idx:
                break

        out_ids: List[int] = ys[0].tolist()[1:]  # drop <sos>
        words: List[str] = []
        for i in out_ids:
            if i == self.eos_idx:
                break
            if i in {self.pad_idx, self.sos_idx}:
                continue
            if 0 <= i < len(self._tgt_vocab):
                words.append(self._tgt_vocab.itos[i])
        return " ".join(words)

    # -------------------------------------- helpers used by training scripts
    def attach_vocabs(self, src_vocab, tgt_vocab) -> None:
        """Bind in-memory vocabs (used during training when loading from disk is unnecessary)."""
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
