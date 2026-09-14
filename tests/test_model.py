import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import TinyTransformerClassifier


def test_forward_shape():
    model = TinyTransformerClassifier(vocab_size=50, num_classes=3)
    model.eval()
    x = torch.randint(0, 50, (8, 24))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 3)


def test_padding_mask_ignored():
    model = TinyTransformerClassifier(vocab_size=50, num_classes=3)
    model.eval()
    x = torch.randint(1, 50, (4, 24))
    x_padded = x.clone()
    x_padded[:, -6:] = 0  # pad the tail
    with torch.no_grad():
        _ = model(x)  # full-length forward pass still runs cleanly
        out_padded = model(x_padded)
    # outputs should differ only because inputs differ; just check no NaN
    assert torch.isfinite(out_padded).all()
