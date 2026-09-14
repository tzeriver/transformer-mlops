import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, MAX_LEN, build_vocab, encode, generate_samples


def test_class_balance():
    samples = generate_samples(n_per_class=120, seed=0)
    counts = {c: 0 for c in CLASSES}
    for _, label in samples:
        counts[label] += 1
    assert all(v == 120 for v in counts.values())


def test_vocab_and_encode():
    samples = generate_samples(n_per_class=40, seed=1)
    vocab = build_vocab(samples)
    assert vocab["<pad>"] == 0 and vocab["<unk>"] == 1
    ids = encode("we trained the transformer", vocab)
    assert len(ids) == MAX_LEN
    assert all(0 <= i < len(vocab) for i in ids)
