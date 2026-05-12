# DA6401 Assignment 3 — Transformer NMT (DE → EN)

Implementation of "Attention Is All You Need" trained on Multi30k, plus the
five required ablation studies and an automatically generated W&B report.

## Files

| File | Purpose |
|------|---------|
| `model.py` | Full Transformer (attention, PE, encoder/decoder, `infer()`). |
| `dataset.py` | Multi30k loader, spaCy tokenizers, vocab, collator. |
| `lr_scheduler.py` | Noam learning-rate scheduler. |
| `train.py` | Loss, training loop, greedy decoding, BLEU eval, checkpoints, single-run entry point. |
| `wandb_logging.py` | Native W&B plot helpers (no matplotlib images). |
| `ablations.py` | Run baseline + all 5 ablations; writes `checkpoints/manifest.json`. |
| `make_report.py` | Builds the public W&B report titled **A3** programmatically. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
wandb login
```

## Reproducing

```bash
# Run the full ablation suite (≈3-5 GPU-hours total).
python ablations.py --epochs 10 --project da6401-a3

# Build the public W&B report named "A3".
python make_report.py --manifest checkpoints/manifest.json --title A3
```

To run a single ablation:

```bash
python ablations.py --epochs 10 --only ablation_2_2_noscaling
```

## Autograder contract

`Transformer()` is callable with no arguments. To reproduce the autograder
flow locally after training:

```python
from model import Transformer
m = Transformer(weights_drive_id="<your-drive-id>", checkpoint_path="checkpoint.pt", vocab_path="checkpoints/vocab.pkl")
m.eval()
m.infer("Ein kleiner Hund läuft im Park.")
```

The `Transformer.__init__` downloads weights from Google Drive via `gdown`
when a `weights_drive_id` is provided and the file is not present locally.
