"""
train.py - Training pipeline, greedy decoding, BLEU evaluation and a shared
experiment runner used by ablations.py.

Autograder-facing signatures preserved:
    LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
    greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device)
    evaluate_bleu(model, test_dataloader, tgt_vocab, device, max_len=100)
    save_checkpoint(model, optimizer, scheduler, epoch, path)
    load_checkpoint(path, model, optimizer, scheduler) -> int
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import sacrebleu
import wandb

from dataset import (
    PAD_IDX, SOS_IDX, EOS_IDX,
    Multi30kDataset, Vocab,
    get_dataloaders, save_vocabs,
)
from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler

import wandb_logging as wlog


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# Label smoothing loss
# ============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    KL-divergence between a smoothed one-hot target and the model's softmax
    output, with the pad token receiving zero target probability.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [N, V]; target: [N]
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / max(1, self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0
            mask = (target == self.pad_idx).unsqueeze(1)
            true_dist.masked_fill_(mask, 0.0)
        loss = -(true_dist * log_probs).sum(dim=-1)
        # Normalise over non-pad tokens.
        non_pad = (target != self.pad_idx).sum().clamp_min(1)
        return loss.sum() / non_pad


# ============================================================================
# Per-epoch training / eval loop
# ============================================================================

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    pad_idx: int = PAD_IDX,
    log_extras: Optional[Dict[str, Any]] = None,
    global_step_ref: Optional[List[int]] = None,
    clip: float = 1.0,
) -> float:
    """
    One pass over `data_iter`. Returns average raw cross-entropy loss
    (independent of whether smoothing was applied -- this makes train and
    val comparable across smoothing settings).

    `log_extras` is a dict of switches that controls W&B logging:
        - log_grad_norms: bool  (every K steps)
        - grad_log_every:  int
        - log_attn_diag:   bool
        - log_confidence:  bool
        - max_grad_log_steps: int  (cap on grad-norm logging)
    `global_step_ref` is a single-element list used as a mutable counter
    across epochs.
    """
    log_extras = log_extras or {}
    grad_log_every = int(log_extras.get("grad_log_every", 50))
    max_grad_log_steps = int(log_extras.get("max_grad_log_steps", 1_000_000))
    do_attn_diag = bool(log_extras.get("log_attn_diag", False))
    do_confidence = bool(log_extras.get("log_confidence", False))
    do_grad_log = bool(log_extras.get("log_grad_norms", False))

    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    n_tok = 0
    epoch_start = time.time()
    if global_step_ref is None:
        global_step_ref = [0]

    raw_ce = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")

    desc = f"epoch {epoch_num} {'train' if is_train else 'val  '}"
    pbar = tqdm(data_iter, desc=desc, leave=False, dynamic_ncols=True)
    for src, tgt in pbar:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx)

        logits = model(src, tgt_in, src_mask, tgt_mask)        # [B, T-1, V]
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_target = tgt_out.reshape(-1)

        loss = loss_fn(flat_logits, flat_target)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient logging BEFORE clipping so we see true magnitudes.
            if do_grad_log and global_step_ref[0] < max_grad_log_steps and global_step_ref[0] % grad_log_every == 0:
                wlog.log_grad_norms(model, step=global_step_ref[0])

            if clip is not None and clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            ce = raw_ce(flat_logits, flat_target).item()
            n = (flat_target != pad_idx).sum().item()
            total_loss += ce
            total_tokens += n
            preds = flat_logits.argmax(-1)
            correct += ((preds == flat_target) & (flat_target != pad_idx)).sum().item()
            n_tok += n

        # Per-step W&B logging during training.
        if is_train:
            payload = {
                "train/loss_step": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            wandb.log(payload, step=global_step_ref[0])
            if do_attn_diag and global_step_ref[0] % grad_log_every == 0:
                wlog.log_attention_diagnostics(model, step=global_step_ref[0])
            if do_confidence and global_step_ref[0] % grad_log_every == 0:
                wlog.log_confidence(flat_logits.detach(), flat_target.detach(),
                                     pad_idx=pad_idx, step=global_step_ref[0])
            global_step_ref[0] += 1

        # Live progress: token-level CE on this batch.
        if total_tokens > 0:
            pbar.set_postfix(loss=f"{total_loss / total_tokens:.3f}", acc=f"{correct/max(1,n_tok):.3f}")

    avg = total_loss / max(1, total_tokens)
    acc = correct / max(1, n_tok)
    return avg, acc, time.time() - epoch_start


# ============================================================================
# Greedy decoding + BLEU
# ============================================================================

@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    memory = model.encode(src.to(device), src_mask.to(device))
    ys = torch.tensor([[start_symbol]], device=device, dtype=torch.long)
    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, model.pad_idx)
        logits = model.decode(memory, src_mask.to(device), ys, tgt_mask)
        next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device, dtype=torch.long)], dim=1)
        if next_id == end_symbol:
            break
    return ys


def _ids_to_tokens(ids: Sequence[int], vocab: Vocab, eos: int, sos: int, pad: int) -> List[str]:
    out: List[str] = []
    for i in ids:
        i = int(i)
        if i == eos:
            break
        if i in {sos, pad}:
            continue
        if 0 <= i < len(vocab):
            out.append(vocab.itos[i])
    return out


@torch.no_grad()
def batched_greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    pad_idx: int,
    device: str,
) -> torch.Tensor:
    """Vectorised greedy decoder over a batch -- ~20x faster than the per-sentence loop."""
    model.eval()
    B = src.size(0)
    memory = model.encode(src.to(device), src_mask.to(device))
    ys = torch.full((B, 1), start_symbol, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx)
        logits = model.decode(memory, src_mask.to(device), ys, tgt_mask)
        next_id = logits[:, -1, :].argmax(dim=-1)        # [B]
        # Once finished, keep emitting pad so we don't extend the meaningful prefix.
        next_id = torch.where(finished, torch.full_like(next_id, pad_idx), next_id)
        ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
        finished = finished | (next_id == end_symbol)
        if bool(finished.all().item()):
            break
    return ys


@torch.no_grad()
def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab: Vocab,
    device: str = "cpu",
    max_len: int = 100,
    return_samples: bool = False,
    src_vocab: Optional[Vocab] = None,
    n_samples: int = 20,
) -> float | Tuple[float, List[Tuple[str, str, str, float]]]:
    """Corpus BLEU on the dataloader using *batched* greedy decoding."""
    model.eval()
    hyps: List[str] = []
    refs: List[str] = []
    samples: List[Tuple[str, str, str, float]] = []

    for src_batch, tgt_batch in test_dataloader:
        src_mask = make_src_mask(src_batch, model.pad_idx)
        ys = batched_greedy_decode(
            model, src_batch, src_mask,
            max_len=max_len, start_symbol=model.sos_idx, end_symbol=model.eos_idx,
            pad_idx=model.pad_idx, device=device,
        )
        for i in range(src_batch.size(0)):
            hyp_ids = ys[i].tolist()[1:]
            ref_ids = tgt_batch[i].tolist()[1:]
            hyp_tok = _ids_to_tokens(hyp_ids, tgt_vocab, EOS_IDX, SOS_IDX, PAD_IDX)
            ref_tok = _ids_to_tokens(ref_ids, tgt_vocab, EOS_IDX, SOS_IDX, PAD_IDX)
            hyp_str = " ".join(hyp_tok)
            ref_str = " ".join(ref_tok)
            hyps.append(hyp_str)
            refs.append(ref_str)
            if return_samples and len(samples) < n_samples and src_vocab is not None:
                src_ids = src_batch[i].tolist()
                src_tok = _ids_to_tokens(src_ids, src_vocab, EOS_IDX, SOS_IDX, PAD_IDX)
                src_str = " ".join(src_tok)
                sb = sacrebleu.sentence_bleu(hyp_str, [ref_str]).score
                samples.append((src_str, ref_str, hyp_str, float(sb)))

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    if return_samples:
        return float(bleu), samples
    return float(bleu)


# ============================================================================
# Checkpoint utilities (autograder-compatible)
# ============================================================================

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    config = {
        "src_vocab_size": model.src_vocab_size,
        "tgt_vocab_size": model.tgt_vocab_size,
        "d_model": model.d_model,
        "N": model.N,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "dropout": model.dropout_p,
        "max_len": model.max_len,
        "pad_idx": model.pad_idx,
        "sos_idx": model.sos_idx,
        "eos_idx": model.eos_idx,
        "use_scaling": model.use_scaling,
        "pos_encoding": model.pos_encoding,
    }
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": config,
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt.get("epoch", 0))


# ============================================================================
# Experiment configuration + entry point
# ============================================================================

@dataclass
class ExperimentConfig:
    name: str = "baseline"
    seed: int = 42
    # Architecture
    d_model: int = 256
    N: int = 3
    num_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 100
    # Ablation switches
    use_scaling: bool = True
    pos_encoding: str = "sinusoidal"   # sinusoidal | learned
    label_smoothing: float = 0.1
    scheduler_kind: str = "noam"        # noam | fixed
    fixed_lr: float = 1e-4
    warmup_steps: int = 4000
    # Training
    batch_size: int = 64
    epochs: int = 10
    grad_clip: float = 1.0
    # Logging budget
    grad_log_every: int = 50
    max_grad_log_steps: int = 1500   # only first ~1500 steps for §2.2
    log_grad_norms: bool = False
    log_attn_diag: bool = False
    log_confidence: bool = False
    # I/O
    project: str = "da6401-a3"
    entity: Optional[str] = None
    ckpt_dir: str = "checkpoints"
    vocab_path: str = "checkpoints/vocab.pkl"
    notes: str = ""
    tags: list = field(default_factory=list)


def _build_optimizer_and_scheduler(model: nn.Module, cfg: ExperimentConfig):
    if cfg.scheduler_kind == "noam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.fixed_lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = None
    return optimizer, scheduler


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Train a single configuration end-to-end with full W&B logging.

    Returns a result dict with final val/test BLEU and the run id, used by the
    report builder.
    """
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=cfg.name,
        config=asdict(cfg),
        notes=cfg.notes,
        tags=cfg.tags,
        reinit=True,
    )

    # ----- data
    train_set, val_set, test_set, train_loader, val_loader, test_loader = get_dataloaders(batch_size=cfg.batch_size)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    save_vocabs(train_set.src_vocab, train_set.tgt_vocab, cfg.vocab_path)

    wandb.summary.update({
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "src_vocab_size": len(train_set.src_vocab),
        "tgt_vocab_size": len(train_set.tgt_vocab),
        "device": device,
    })

    # ----- model
    model = Transformer(
        src_vocab_size=len(train_set.src_vocab),
        tgt_vocab_size=len(train_set.tgt_vocab),
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=cfg.max_len + 2,
        pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX,
        use_scaling=cfg.use_scaling,
        pos_encoding=cfg.pos_encoding,
    ).to(device)
    model.attach_vocabs(train_set.src_vocab, train_set.tgt_vocab)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.summary["model/trainable_params"] = int(n_params)

    optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg)

    # One-shot LR schedule plot for §2.1.
    if cfg.scheduler_kind == "noam":
        wlog.log_lr_schedule(cfg.d_model, cfg.warmup_steps, total_steps=cfg.epochs * len(train_loader), run_label=cfg.name)

    # Loss function (label smoothing or vanilla CE).
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_set.tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=cfg.label_smoothing,
    )

    # ----- training loop
    global_step = [0]
    best_val_bleu = -1.0
    best_path = os.path.join(cfg.ckpt_dir, f"{cfg.name}_best.pt")
    exp_start = time.time()

    print(f"\n{'='*72}\n>>> {cfg.name}  |  epochs={cfg.epochs}  device={device}  params={n_params/1e6:.2f}M\n{'='*72}", flush=True)

    for epoch in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        train_loss, train_acc, dt_train = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device, pad_idx=PAD_IDX,
            log_extras={
                "log_grad_norms": cfg.log_grad_norms,
                "grad_log_every": cfg.grad_log_every,
                "max_grad_log_steps": cfg.max_grad_log_steps,
                "log_attn_diag": cfg.log_attn_diag,
                "log_confidence": cfg.log_confidence,
            },
            global_step_ref=global_step,
            clip=cfg.grad_clip,
        )
        # Validation pass (no grads).
        val_loss, val_acc, _ = run_epoch(
            val_loader, model, loss_fn, optimizer=None, scheduler=None,
            epoch_num=epoch, is_train=False, device=device, pad_idx=PAD_IDX,
            global_step_ref=global_step,
        )

        # BLEU on validation (batched, ~20x faster than per-sentence).
        bleu_t0 = time.time()
        val_bleu = evaluate_bleu(model, val_loader, train_set.tgt_vocab, device=device, max_len=cfg.max_len)
        dt_bleu = time.time() - bleu_t0

        wandb.log({
            "epoch": epoch,
            "train/loss_epoch": train_loss,
            "train/perplexity_epoch": math.exp(min(train_loss, 20)),
            "train/token_accuracy_epoch": train_acc,
            "val/loss_epoch": val_loss,
            "val/perplexity_epoch": math.exp(min(val_loss, 20)),
            "val/token_accuracy_epoch": val_acc,
            "val/BLEU": val_bleu,
            "epoch_seconds": dt_train,
        }, step=global_step[0])

        # Checkpoint best.
        improved = val_bleu > best_val_bleu
        if improved:
            best_val_bleu = val_bleu
            save_checkpoint(model, optimizer, scheduler, epoch, best_path)

        # ETA based on average epoch time so far.
        ep_secs = time.time() - ep_t0
        elapsed = time.time() - exp_start
        avg_ep = elapsed / epoch
        eta_sec = avg_ep * (cfg.epochs - epoch)
        marker = " *" if improved else "  "
        print(
            f"[{cfg.name}] epoch {epoch:>2}/{cfg.epochs} "
            f"| train {train_loss:5.3f} (acc {train_acc:.3f}) "
            f"| val {val_loss:5.3f} (acc {val_acc:.3f}) "
            f"| BLEU {val_bleu:5.2f}{marker} "
            f"| {ep_secs:5.1f}s (train {dt_train:.0f}s + bleu {dt_bleu:.0f}s) "
            f"| ETA {eta_sec/60:5.1f}m",
            flush=True,
        )

    # ----- final test evaluation
    test_bleu, samples = evaluate_bleu(
        model, test_loader, train_set.tgt_vocab,
        device=device, max_len=cfg.max_len, return_samples=True,
        src_vocab=train_set.src_vocab, n_samples=20,
    )
    wandb.summary["test/BLEU"] = test_bleu
    wandb.summary["val/best_BLEU"] = best_val_bleu
    wlog.log_translation_table(samples)

    # ----- post-hoc diagnostics (single batch from val) to feed §2.3 / §2.4 / §2.5
    try:
        _post_hoc_diagnostics(model, val_loader, train_set, cfg, global_step[0])
    except Exception as e:
        wandb.alert(title="post-hoc diagnostics failed", text=str(e))

    run_id = run.id
    run.finish()
    return {
        "name": cfg.name,
        "run_id": run_id,
        "test_bleu": float(test_bleu),
        "val_best_bleu": float(best_val_bleu),
        "ckpt_path": best_path,
    }


# ============================================================================
# Post-hoc diagnostics (§2.3 / §2.4 / §2.5)
# ============================================================================

@torch.no_grad()
def _post_hoc_diagnostics(model, val_loader, train_set, cfg: ExperimentConfig, step: int) -> None:
    """Final-pass diagnostics: attention heatmaps, head similarity, PE plots, calibration."""
    device = next(model.parameters()).device
    model.eval()

    # Pick a single sentence with reasonable length.
    src_b, tgt_b = next(iter(val_loader))
    chosen = 0
    for i in range(src_b.size(0)):
        if (src_b[i] != PAD_IDX).sum().item() in range(8, 18):
            chosen = i
            break
    src = src_b[chosen:chosen + 1].to(device)
    tgt = tgt_b[chosen:chosen + 1].to(device)
    src_mask = make_src_mask(src, PAD_IDX)
    tgt_in = tgt[:, :-1]
    tgt_mask = make_tgt_mask(tgt_in, PAD_IDX)

    # Forward populates last_attn on every MHA module.
    logits = model(src, tgt_in, src_mask, tgt_mask)

    # §2.3 attention heatmaps
    src_tokens = []
    for i in src[0].tolist():
        if i == EOS_IDX: break
        if i in {SOS_IDX, PAD_IDX}: continue
        src_tokens.append(train_set.src_vocab.itos[i])
    last_layer = len(model.encoder.layers) - 1
    wlog.log_attention_heatmaps_last_encoder(model, src_tokens, layer_idx=last_layer, step=step, title="enc_last")
    wlog.log_attention_heatmaps_last_encoder(model, src_tokens, layer_idx=0, step=step, title="enc_first")
    wlog.log_head_similarity_matrix(model, layer_idx=last_layer, step=step)
    wlog.log_head_role_table(model, layer_idx=last_layer, step=step)

    # §2.4 PE diagnostics
    wlog.log_pe_matrix(model, step=step, max_pos=80)
    wlog.log_pe_position_similarity(model, step=step)

    # §2.5 calibration on a chunk of validation logits
    # Recompute on a wider batch for statistics
    src_b2, tgt_b2 = next(iter(val_loader))
    src_b2 = src_b2.to(device); tgt_b2 = tgt_b2.to(device)
    sm2 = make_src_mask(src_b2, PAD_IDX)
    tgt_in2 = tgt_b2[:, :-1]
    tgt_out2 = tgt_b2[:, 1:]
    tm2 = make_tgt_mask(tgt_in2, PAD_IDX)
    logits2 = model(src_b2, tgt_in2, sm2, tm2)
    flat_logits = logits2.reshape(-1, logits2.size(-1))
    flat_target = tgt_out2.reshape(-1)
    wlog.log_calibration_curve(flat_logits, flat_target, pad_idx=PAD_IDX, step=step)
    wlog.log_confidence(flat_logits, flat_target, pad_idx=PAD_IDX, step=step, log_histogram=True)


# ============================================================================
# Single-run CLI entry point (the autograder won't call this).
# ============================================================================

def run_training_experiment() -> None:
    cfg = ExperimentConfig(
        name="baseline",
        epochs=10,
        log_attn_diag=True,
        log_confidence=True,
        log_grad_norms=True,
    )
    res = run_experiment(cfg)
    print(res)


if __name__ == "__main__":
    run_training_experiment()
