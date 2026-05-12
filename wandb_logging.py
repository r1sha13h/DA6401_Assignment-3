"""
wandb_logging.py - Native Weights & Biases logging utilities.

Hard rule: every visualisation MUST use the W&B native primitives only.
No matplotlib images, no PNG uploads, no plotly figures. The allowed APIs
in this module are:

    wandb.log({...})                     # scalars + histograms + tables
    wandb.Histogram(values)
    wandb.Table(columns=..., data=...)
    wandb.plot.line / bar / scatter / heatmap / line_series / histogram

This is a deliberate constraint imposed by the assignment brief.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import math
import torch
import torch.nn.functional as F

import wandb


# ============================================================================
# Universal per-step / per-epoch metric logging
# ============================================================================

def log_step(metrics: Dict[str, float], step: int) -> None:
    """Tiny wrapper around wandb.log so callers don't pass `step` repeatedly."""
    wandb.log({k: float(v) for k, v in metrics.items()}, step=step)


# ============================================================================
# Gradient-norm logging (used by all experiments + critically by §2.2)
# ============================================================================

def log_grad_norms(
    model: torch.nn.Module,
    step: int,
    name_filters: Iterable[str] = ("W_q", "W_k", "W_v", "W_o"),
    layer_prefix: str = "encoder.layers",
    per_layer: bool = True,
) -> None:
    """
    Log per-parameter gradient L2 norms as scalar time series.

    For §2.2 we want Q,K,V,O grad norms in every encoder layer for the first
    1000 steps. For other experiments we only log the global norm. The whole
    thing is one wandb.log call so all metrics share the step.
    """
    payload: Dict[str, float] = {}
    total_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        n = float(g.detach().data.norm(2).item())
        total_sq += n * n
        if not per_layer:
            continue
        if any(f in name for f in name_filters) and layer_prefix in name:
            # Pull layer index out of "encoder.layers.0.self_attn.W_q.weight"
            try:
                parts = name.split(".")
                idx = parts[parts.index("layers") + 1]
                kind = next(f for f in name_filters if f in name)
                payload[f"grad_norm/{kind}/L{idx}"] = n
            except (ValueError, StopIteration):
                pass
    payload["grad_norm/global"] = math.sqrt(total_sq)
    wandb.log(payload, step=step)


# ============================================================================
# Attention diagnostics (entropy / max / pre-softmax std)
# ============================================================================

def log_attention_diagnostics(model: torch.nn.Module, step: int) -> None:
    """For every encoder MultiHeadAttention, log entropy and max of last attn map."""
    payload: Dict[str, float] = {}
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    for li, layer in enumerate(encoder.layers):
        attn = getattr(layer.self_attn, "last_attn", None)
        if attn is None:
            continue
        # attn: [B, h, Lq, Lk]
        eps = 1e-9
        entropy = -(attn * (attn + eps).log()).sum(-1).mean().item()  # mean over batch/heads/q
        max_w = attn.max().item()
        payload[f"attention/entropy/L{li}"] = entropy
        payload[f"attention/max_weight/L{li}"] = max_w
    if payload:
        wandb.log(payload, step=step)


# ============================================================================
# Confidence (§2.5 label-smoothing experiment)
# ============================================================================

def log_confidence(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_idx: int,
    step: int,
    log_histogram: bool = False,
) -> None:
    """
    Log mean / median probability the model assigns to the gold token.
    Optionally log a wandb.Histogram of the distribution.
    """
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        gold = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        mask = target != pad_idx
        gold = gold[mask].float().detach().cpu()
        if gold.numel() == 0:
            return
        payload = {
            "confidence/mean_correct_prob": gold.mean().item(),
            "confidence/median_correct_prob": gold.median().item(),
            "confidence/entropy_pred": float(-(probs * (probs.clamp_min(1e-9)).log()).sum(-1).mean().item()),
        }
        if log_histogram:
            payload["confidence/hist_correct_prob"] = wandb.Histogram(gold.numpy())
        wandb.log(payload, step=step)


# ============================================================================
# §2.3 attention head visualisation -- native wandb.plot.heatmap
# ============================================================================

def log_attention_heatmaps_last_encoder(
    model: torch.nn.Module,
    src_tokens: List[str],
    layer_idx: int,
    step: int,
    title: str = "encoder",
) -> None:
    """
    Pull `last_attn` from a chosen encoder layer (already populated by a forward
    pass) and log one wandb.plot.heatmap per head.

    The src_tokens list must match the seq dimension of last_attn (single
    sentence; batch=1).
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None or layer_idx >= len(encoder.layers):
        return
    attn = getattr(encoder.layers[layer_idx].self_attn, "last_attn", None)
    if attn is None:
        return
    # attn: [1, h, L, L]
    attn = attn[0].detach().float().cpu()  # [h, L, L]
    h, L, _ = attn.shape
    L = min(L, len(src_tokens))
    labels = src_tokens[:L]
    payload: Dict[str, object] = {}
    for head in range(h):
        m = attn[head, :L, :L].tolist()
        payload[f"attention/{title}_L{layer_idx}_head{head}"] = wandb.plot.heatmap(
            x_labels=labels,
            y_labels=labels,
            matrix_values=m,
            show_text=False,
        )
    wandb.log(payload, step=step)


def log_head_similarity_matrix(
    model: torch.nn.Module,
    layer_idx: int,
    step: int,
) -> None:
    """8x8 cosine-similarity matrix between flattened attention maps of all heads."""
    encoder = getattr(model, "encoder", None)
    if encoder is None or layer_idx >= len(encoder.layers):
        return
    attn = getattr(encoder.layers[layer_idx].self_attn, "last_attn", None)
    if attn is None:
        return
    attn = attn[0].detach().float().cpu()  # [h, L, L]
    h = attn.size(0)
    flat = attn.view(h, -1)
    flat = F.normalize(flat, dim=-1)
    sim = (flat @ flat.t()).tolist()
    labels = [f"h{i}" for i in range(h)]
    wandb.log({
        f"attention/head_similarity_L{layer_idx}": wandb.plot.heatmap(
            x_labels=labels, y_labels=labels, matrix_values=sim, show_text=True,
        )
    }, step=step)


def log_head_role_table(
    model: torch.nn.Module,
    layer_idx: int,
    step: int,
) -> None:
    """One row per head: prev/self/eos/long-range attention scores."""
    encoder = getattr(model, "encoder", None)
    if encoder is None or layer_idx >= len(encoder.layers):
        return
    attn = getattr(encoder.layers[layer_idx].self_attn, "last_attn", None)
    if attn is None:
        return
    attn = attn[0].detach().float().cpu()  # [h, L, L]
    h, L, _ = attn.shape
    rows = []
    for head in range(h):
        a = attn[head]
        diag = a.diagonal().mean().item()
        prev = a.diagonal(-1).mean().item() if L > 1 else 0.0
        long_range = float(a.triu(5).sum() / max(1, (a.triu(5) > 0).sum()))
        # eos sink: assume eos is the last token in the sentence
        eos = a[:, -1].mean().item()
        rows.append([f"head_{head}", diag, prev, eos, long_range])
    table = wandb.Table(
        columns=["head", "self_token", "prev_token", "eos_sink", "long_range"],
        data=rows,
    )
    wandb.log({f"attention/head_roles_L{layer_idx}": table}, step=step)


# ============================================================================
# §2.4 positional-encoding diagnostics
# ============================================================================

def log_pe_matrix(model: torch.nn.Module, step: int, max_pos: int = 100) -> None:
    """Heatmap of the source positional-encoding matrix for the first `max_pos` positions."""
    src_pos = getattr(model, "src_pos", None)
    if src_pos is None:
        return
    if hasattr(src_pos, "pe") and isinstance(src_pos.pe, torch.Tensor):
        # Sinusoidal: pe is a buffer of shape [1, max_len, d_model]
        m = src_pos.pe[0, :max_pos].detach().float().cpu()
    elif hasattr(src_pos, "pe") and isinstance(src_pos.pe, torch.nn.Embedding):
        m = src_pos.pe.weight[:max_pos].detach().float().cpu()
    else:
        return
    wandb.log({
        "pe/matrix": wandb.plot.heatmap(
            x_labels=[f"d{i}" for i in range(m.size(1))],
            y_labels=[str(i) for i in range(m.size(0))],
            matrix_values=m.tolist(),
            show_text=False,
        )
    }, step=step)


def log_pe_position_similarity(model: torch.nn.Module, step: int, anchor: int = 20, span: int = 30) -> None:
    """
    For a fixed anchor position p, plot dot-product similarity between PE[p]
    and PE[p+k] for k in [-span, +span]. Sinusoidal -> smooth bell.
    """
    src_pos = getattr(model, "src_pos", None)
    if src_pos is None:
        return
    if hasattr(src_pos, "pe") and isinstance(src_pos.pe, torch.Tensor):
        pe = src_pos.pe[0]  # [max_len, d_model]
    elif hasattr(src_pos, "pe") and isinstance(src_pos.pe, torch.nn.Embedding):
        pe = src_pos.pe.weight.detach()
    else:
        return
    pe = pe.detach().float().cpu()
    L = pe.size(0)
    p = min(max(anchor, span), L - span - 1)
    sims = []
    for k in range(-span, span + 1):
        sims.append([k, float(torch.dot(pe[p], pe[p + k]) / (pe[p].norm() * pe[p + k].norm() + 1e-9))])
    table = wandb.Table(columns=["offset_k", "cosine_sim"], data=sims)
    wandb.log({
        "pe/position_similarity": wandb.plot.line(
            table, x="offset_k", y="cosine_sim", title=f"PE cosine sim around pos={p}",
        )
    }, step=step)


# ============================================================================
# §2.1 LR schedule one-shot logging
# ============================================================================

def log_lr_schedule(d_model: int, warmup_steps: int, total_steps: int, run_label: str = "noam") -> None:
    """One-shot at the start of training: log the theoretical LR curve as a native line plot."""
    rows = []
    for step in range(1, total_steps + 1):
        lr = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
        rows.append([step, lr])
    table = wandb.Table(columns=["step", "lr"], data=rows)
    wandb.log({
        f"lr_schedule/{run_label}": wandb.plot.line(
            table, x="step", y="lr", title=f"Noam schedule (warmup={warmup_steps})",
        )
    })


# ============================================================================
# Final-evaluation tables (translations, BLEU, summaries)
# ============================================================================

def log_translation_table(samples: Sequence[Sequence[str]]) -> None:
    """samples: iterable of (src, ref, hyp, sentence_bleu)."""
    table = wandb.Table(
        columns=["source_de", "reference_en", "hypothesis_en", "sentence_bleu"],
        data=[list(s) for s in samples],
    )
    wandb.log({"eval/translations": table})


def log_summary(metrics: Dict[str, float]) -> None:
    """Push terminal scalars into wandb.run.summary so they show up in the run header."""
    if wandb.run is None:
        return
    for k, v in metrics.items():
        wandb.run.summary[k] = v


def log_calibration_curve(
    logits: torch.Tensor,
    target: torch.Tensor,
    pad_idx: int,
    step: int,
    n_bins: int = 10,
) -> None:
    """Reliability diagram for label-smoothing experiment, as a native line plot + ECE scalar."""
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        confidence, pred = probs.max(dim=-1)
        mask = target != pad_idx
        confidence = confidence[mask].float().cpu()
        correct = (pred[mask] == target[mask]).float().cpu()
        if confidence.numel() == 0:
            return
        bins = torch.linspace(0, 1, n_bins + 1)
        rows = []
        ece = 0.0
        N = confidence.numel()
        for i in range(n_bins):
            in_bin = (confidence > bins[i]) & (confidence <= bins[i + 1])
            n = int(in_bin.sum().item())
            if n == 0:
                rows.append([(bins[i].item() + bins[i + 1].item()) / 2, 0.0, 0.0, 0])
                continue
            acc = correct[in_bin].mean().item()
            conf = confidence[in_bin].mean().item()
            ece += (n / N) * abs(acc - conf)
            rows.append([(bins[i].item() + bins[i + 1].item()) / 2, conf, acc, n])
        table = wandb.Table(
            columns=["bin_center", "avg_confidence", "empirical_accuracy", "count"],
            data=rows,
        )
        wandb.log({
            "calibration/reliability": wandb.plot.line(
                table, x="avg_confidence", y="empirical_accuracy", title="Reliability diagram",
            ),
            "calibration/ECE": ece,
        }, step=step)


def log_logit_std(model: torch.nn.Module, step: int) -> None:
    """For §2.2: log the standard deviation of pre-softmax attention scores per encoder layer."""
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    payload: Dict[str, float] = {}
    for li, layer in enumerate(encoder.layers):
        a = getattr(layer.self_attn, "last_attn", None)
        if a is None:
            continue
        # last_attn is post-softmax; we approximate the pre-softmax std via log(attn) within rows.
        # A more direct option is to monkey-patch; we keep this as a useful proxy: entropy ~ std.
        eps = 1e-9
        log_a = (a + eps).log()
        payload[f"attention/log_attn_std/L{li}"] = float(log_a.std().item())
    if payload:
        wandb.log(payload, step=step)
