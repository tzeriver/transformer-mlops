"""Synthetic demo dataset: topic classification of short research-note sentences.

Three classes (methods / results / background), generated from templates with
random slot filling. 100% synthetic - no real, proprietary, or employer data
anywhere in this repo.
"""

from __future__ import annotations

import random
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ["methods", "results", "background"]
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MAX_LEN = 24

_TEMPLATES = {
    "methods": [
        "we trained the {model} for {n} epochs with {opt}",
        "data were split into train validation and test sets",
        "we applied {aug} augmentation to the training inputs",
        "the encoder uses {h} attention heads per layer",
        "optimization used {opt} with learning rate {lr}",
        "we normalized each feature to zero mean unit variance",
        "checkpoints were saved every {n} epochs during training",
        "the pipeline tokenizes text before batching sequences",
    ],
    "results": [
        "accuracy improved from {a} to {b} percent on test",
        "the model reached {a} f1 with {n} training examples",
        "ablation shows {aug} adds {a} points of accuracy",
        "validation loss decreased steadily across {n} epochs",
        "our approach beats the baseline by {a} percent",
        "error analysis found most mistakes involve rare tokens",
        "precision and recall both exceed {a} percent",
        "the largest gains appear with {n} or more samples",
    ],
    "background": [
        "transformers have reshaped sequence modeling in recent years",
        "self attention lets each token weight every other token",
        "prior work focused on recurrent architectures for text",
        "large scale pretraining improved downstream accuracy",
        "evaluation practices vary widely across published studies",
        "reproducibility remains a concern in applied research",
        "open datasets accelerated progress on benchmarks",
        "regularization reduces overfitting on small corpora",
    ],
}

_FILLERS = {
    "model": ["transformer", "cnn", "lstm", "mlp", "resnet"],
    "opt": ["adam", "sgd", "adamw"],
    "n": ["ten", "twenty", "fifty", "one hundred"],
    "h": ["four", "eight"],
    "lr": ["1e-3", "3e-4", "1e-4"],
    "aug": ["dropout", "masking", "noise"],
    "a": ["eighty", "ninety", "seventy five"],
    "b": ["ninety two", "ninety five", "eighty eight"],
}


def _fill(template: str, rng: random.Random) -> str:
    out = template
    for key, choices in _FILLERS.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, rng.choice(choices), 1)
    return out


def generate_samples(n_per_class: int = 1200, seed: int = 42):
    """Return a shuffled list of (text, label) pairs."""
    rng = random.Random(seed)
    samples = []
    for label in CLASSES:
        templates = _TEMPLATES[label]
        reps = max(1, n_per_class // len(templates))
        for template in templates:
            for _ in range(reps):
                samples.append((_fill(template, rng), label))
    rng.shuffle(samples)
    return samples


def build_vocab(samples, min_freq: int = 2):
    counts = Counter()
    for text, _ in samples:
        counts.update(text.split())
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok in sorted(counts):
        if counts[tok] >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab, max_len: int = MAX_LEN):
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in text.split()]
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))


def make_datasets(n_per_class=1200, seed=42, batch_size=64):
    """Stratified 80/10/10 split; returns (loaders, vocab, label2id)."""
    samples = generate_samples(n_per_class, seed)
    by_class: dict[str, list] = {c: [] for c in CLASSES}
    for s in samples:
        by_class[s[1]].append(s)
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for items in by_class.values():
        rng.shuffle(items)
        n = len(items)
        n_train, n_val = int(0.8 * n), int(0.1 * n)
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]

    vocab = build_vocab(splits["train"])
    label2id = {c: i for i, c in enumerate(CLASSES)}

    loaders = {}
    for name, items in splits.items():
        xs = torch.tensor([encode(t, vocab) for t, _ in items], dtype=torch.long)
        ys = torch.tensor([label2id[l] for _, l in items], dtype=torch.long)
        loaders[name] = DataLoader(
            TensorDataset(xs, ys), batch_size=batch_size, shuffle=(name == "train")
        )
    return loaders, vocab, label2id
