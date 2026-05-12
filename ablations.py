"""
ablations.py - Run baseline + 5 ablations and persist their run IDs / scores
into a JSON manifest that make_report.py picks up.

Experiments (mapping to the brief):
    baseline               -- canonical Transformer reference run
    ablation_2_1_noam      -- Noam scheduler ON  (deliverable for §2.1)
    ablation_2_1_fixedlr   -- Fixed LR, no warmup
    ablation_2_2_scaling   -- with sqrt(d_k) scaling (=baseline twin, gradient-heavy logging)
    ablation_2_2_noscaling -- WITHOUT sqrt(d_k) scaling
    ablation_2_4_learnedpe -- learned positional embeddings
    ablation_2_5_smooth0   -- label smoothing eps=0.0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import List

from train import ExperimentConfig, run_experiment


def _common(epochs: int, project: str, entity: str | None) -> dict:
    return dict(
        epochs=epochs,
        project=project,
        entity=entity,
        d_model=256,
        N=3,
        num_heads=8,
        d_ff=1024,
        dropout=0.1,
        batch_size=64,
        warmup_steps=2000,
        seed=42,
    )


def build_configs(epochs: int, project: str, entity: str | None) -> List[ExperimentConfig]:
    base = _common(epochs, project, entity)

    cfgs: List[ExperimentConfig] = []

    # 0. Baseline: full instrumentation
    cfgs.append(ExperimentConfig(
        name="baseline",
        log_grad_norms=True, log_attn_diag=True, log_confidence=True,
        tags=["baseline"],
        notes="Canonical Transformer (Noam, sinusoidal PE, smoothing=0.1).",
        **base,
    ))

    # 1. §2.1 Noam vs fixed
    cfgs.append(ExperimentConfig(
        name="ablation_2_1_noam",
        scheduler_kind="noam", log_grad_norms=True, log_attn_diag=True,
        tags=["ablation_2_1", "noam"],
        notes="§2.1: Noam warmup-decay schedule.",
        **base,
    ))
    cfgs.append(ExperimentConfig(
        name="ablation_2_1_fixedlr",
        scheduler_kind="fixed", fixed_lr=1e-4, log_grad_norms=True, log_attn_diag=True,
        tags=["ablation_2_1", "fixed"],
        notes="§2.1: constant LR, no warmup -- expected to diverge or stagnate.",
        **base,
    ))

    # 2. §2.2 scaling factor ablation: gradient-norm logging is the key deliverable
    cfgs.append(ExperimentConfig(
        name="ablation_2_2_scaling",
        use_scaling=True,
        log_grad_norms=True, log_attn_diag=True, max_grad_log_steps=1500, grad_log_every=20,
        tags=["ablation_2_2", "scaled"],
        notes="§2.2: standard 1/sqrt(d_k) scaling.",
        **base,
    ))
    cfgs.append(ExperimentConfig(
        name="ablation_2_2_noscaling",
        use_scaling=False,
        log_grad_norms=True, log_attn_diag=True, max_grad_log_steps=1500, grad_log_every=20,
        tags=["ablation_2_2", "unscaled"],
        notes="§2.2: NO scaling -- expect attention to saturate, Q/K grads to vanish.",
        **base,
    ))

    # 3. §2.4 sinusoidal vs learned PE
    cfgs.append(ExperimentConfig(
        name="ablation_2_4_learnedpe",
        pos_encoding="learned",
        log_attn_diag=False, log_confidence=False,
        tags=["ablation_2_4", "learned_pe"],
        notes="§2.4: nn.Embedding(max_len,d_model) positions, learned.",
        **base,
    ))
    # the sinusoidal counterpart is the baseline

    # 4. §2.5 label smoothing 0.0 vs 0.1
    cfgs.append(ExperimentConfig(
        name="ablation_2_5_smooth0",
        label_smoothing=0.0,
        log_confidence=True,
        tags=["ablation_2_5", "no_smoothing"],
        notes="§2.5: standard cross-entropy (no smoothing).",
        **base,
    ))
    # eps=0.1 counterpart is the baseline

    return cfgs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--project", type=str, default="da6401-a3")
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--manifest", type=str, default="checkpoints/manifest.json")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated experiment names to run; default = all.")
    args = p.parse_args()

    cfgs = build_configs(args.epochs, args.project, args.entity)
    if args.only:
        keep = set(args.only.split(","))
        cfgs = [c for c in cfgs if c.name in keep]

    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    if os.path.isfile(args.manifest):
        with open(args.manifest, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"project": args.project, "entity": args.entity, "runs": {}}

    suite_start = time.time()
    total = len(cfgs)
    print(f"\n{'#'*72}\n# Ablation suite: {total} runs x {args.epochs} epochs\n{'#'*72}", flush=True)

    for idx, cfg in enumerate(cfgs, start=1):
        elapsed_min = (time.time() - suite_start) / 60.0
        avg_per_run = elapsed_min / max(1, idx - 1) if idx > 1 else 0.0
        eta_min = avg_per_run * (total - idx + 1) if avg_per_run > 0 else 0.0
        print(
            f"\n>>> SUITE [{idx}/{total}] starting `{cfg.name}` "
            f"| suite-elapsed {elapsed_min:.1f}m"
            f"{f' | suite-ETA {eta_min:.1f}m' if eta_min > 0 else ''}",
            flush=True,
        )
        t0 = time.time()
        try:
            res = run_experiment(cfg)
            wall_min = (time.time() - t0) / 60.0
            res["wall_minutes"] = wall_min
            manifest["runs"][cfg.name] = res
            with open(args.manifest, "w") as f:
                json.dump(manifest, f, indent=2)
            print(
                f"<<< [{idx}/{total}] DONE `{cfg.name}` in {wall_min:.1f}m "
                f"| test_BLEU={res.get('test_bleu', 0):.2f} "
                f"| val_best_BLEU={res.get('val_best_bleu', 0):.2f}",
                flush=True,
            )
        except Exception as e:
            manifest["runs"][cfg.name] = {"error": str(e)}
            with open(args.manifest, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"!!! [{idx}/{total}] {cfg.name} FAILED: {e}", flush=True)
            continue

    total_min = (time.time() - suite_start) / 60.0
    print(f"\n{'#'*72}\n# Suite finished in {total_min:.1f}m ({total_min/60:.2f}h)\n{'#'*72}", flush=True)

    print("\nManifest:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
