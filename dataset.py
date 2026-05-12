"""
dataset.py - Multi30k dataset loading, spaCy tokenization, vocabulary, and
DataLoader collator for the German -> English NMT task.
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from typing import Iterable, List, Sequence

import spacy
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# Special token indices (used everywhere in the project).
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]


class Vocab:
    """Minimal frozen vocabulary mapping token -> int index."""

    def __init__(self, itos: List[str]):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(tok, unk) for tok in tokens]

    def decode(self, ids: Sequence[int], strip_specials: bool = True) -> List[str]:
        toks = [self.itos[i] for i in ids if 0 <= i < len(self.itos)]
        if strip_specials:
            toks = [t for t in toks if t not in SPECIAL_TOKENS]
        return toks

    @classmethod
    def from_counter(cls, counter: Counter, min_freq: int = 2, max_size: int | None = None) -> "Vocab":
        items = [(tok, c) for tok, c in counter.items() if c >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        if max_size is not None:
            items = items[: max(0, max_size - len(SPECIAL_TOKENS))]
        itos = list(SPECIAL_TOKENS) + [tok for tok, _ in items]
        return cls(itos)


def _spacy_tokenize(nlp: spacy.language.Language, text: str) -> List[str]:
    return [t.text.lower() for t in nlp.tokenizer(text.strip()) if t.text.strip()]


class Multi30kDataset(Dataset):
    """
    Multi30k German <-> English machine translation dataset wrapper.

    Loads bentrevett/multi30k from Hugging Face, tokenizes both sides with
    spaCy and converts tokens to integer indices using a shared vocabulary
    object built once on the training split.
    """

    HF_NAME = "bentrevett/multi30k"

    def __init__(
        self,
        split: str = "train",
        src_vocab: Vocab | None = None,
        tgt_vocab: Vocab | None = None,
        max_len: int = 100,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        assert split in {"train", "validation", "test"}
        self.split = split
        self.max_len = max_len

        self._nlp_de = spacy.load("de_core_news_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
        self._nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer"])

        ds = load_dataset(self.HF_NAME, split=split, cache_dir=cache_dir)
        self._raw = [(ex["de"], ex["en"]) for ex in ds]

        # Build vocab from training split if not provided.
        if src_vocab is None or tgt_vocab is None:
            assert split == "train", "Vocab must be passed for non-train splits."
            src_vocab, tgt_vocab = self.build_vocab()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # Pre-tokenize and pre-index everything (the dataset is small).
        self._encoded: List[tuple[List[int], List[int]]] = []
        for de, en in self._raw:
            src_ids = self.src_vocab.encode(_spacy_tokenize(self._nlp_de, de))[: max_len - 2]
            tgt_ids = self.tgt_vocab.encode(_spacy_tokenize(self._nlp_en, en))[: max_len - 2]
            self._encoded.append((src_ids, tgt_ids))

    # ------------------------------------------------------------------ vocab
    def build_vocab(self, min_freq: int = 2) -> tuple[Vocab, Vocab]:
        de_counter: Counter[str] = Counter()
        en_counter: Counter[str] = Counter()
        for de, en in self._raw:
            de_counter.update(_spacy_tokenize(self._nlp_de, de))
            en_counter.update(_spacy_tokenize(self._nlp_en, en))
        src_vocab = Vocab.from_counter(de_counter, min_freq=min_freq)
        tgt_vocab = Vocab.from_counter(en_counter, min_freq=min_freq)
        return src_vocab, tgt_vocab

    # --------------------------------------------------------------- standard
    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_ids, tgt_ids = self._encoded[idx]
        src = torch.tensor([SOS_IDX] + src_ids + [EOS_IDX], dtype=torch.long)
        tgt = torch.tensor([SOS_IDX] + tgt_ids + [EOS_IDX], dtype=torch.long)
        return src, tgt

    # ----------------------------------------------------------- processing
    def process_data(self) -> List[tuple[List[int], List[int]]]:
        """Return the pre-tokenized integer-encoded pairs."""
        return list(self._encoded)

    # -------------------------------------------------- single-sentence path
    def encode_src_sentence(self, sentence: str) -> torch.Tensor:
        ids = self.src_vocab.encode(_spacy_tokenize(self._nlp_de, sentence))
        ids = [SOS_IDX] + ids[: self.max_len - 2] + [EOS_IDX]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def decode_tgt_ids(self, ids: Sequence[int]) -> str:
        toks: List[str] = []
        for i in ids:
            i = int(i)
            if i == EOS_IDX:
                break
            if i in {SOS_IDX, PAD_IDX}:
                continue
            toks.append(self.tgt_vocab.itos[i] if 0 <= i < len(self.tgt_vocab) else "<unk>")
        return " ".join(toks)


# ---------------------------------------------------------- collator + helpers

def make_collate_fn(pad_idx: int = PAD_IDX):
    """Pad a batch of (src, tgt) variable-length sequences to a rectangular tensor."""

    def collate(batch: Iterable[tuple[torch.Tensor, torch.Tensor]]):
        srcs, tgts = zip(*batch)
        src_max = max(s.size(0) for s in srcs)
        tgt_max = max(t.size(0) for t in tgts)
        src_pad = torch.full((len(srcs), src_max), pad_idx, dtype=torch.long)
        tgt_pad = torch.full((len(tgts), tgt_max), pad_idx, dtype=torch.long)
        for i, (s, t) in enumerate(zip(srcs, tgts)):
            src_pad[i, : s.size(0)] = s
            tgt_pad[i, : t.size(0)] = t
        return src_pad, tgt_pad

    return collate


def get_dataloaders(
    batch_size: int = 64,
    num_workers: int = 0,
    cache_dir: str | None = None,
):
    """Build train / val / test DataLoaders sharing the same vocabularies."""
    train_set = Multi30kDataset("train", cache_dir=cache_dir)
    val_set = Multi30kDataset("validation", train_set.src_vocab, train_set.tgt_vocab, cache_dir=cache_dir)
    test_set = Multi30kDataset("test", train_set.src_vocab, train_set.tgt_vocab, cache_dir=cache_dir)
    collate = make_collate_fn(PAD_IDX)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    return train_set, val_set, test_set, train_loader, val_loader, test_loader


# ----------------------------------------------------------- vocab persistence

def save_vocabs(src_vocab: Vocab, tgt_vocab: Vocab, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"src_itos": src_vocab.itos, "tgt_itos": tgt_vocab.itos}, f)


def load_vocabs(path: str) -> tuple[Vocab, Vocab]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return Vocab(d["src_itos"]), Vocab(d["tgt_itos"])