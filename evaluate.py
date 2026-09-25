"""Evaluation and metrics."""


def evaluate(model, loader, device):
    """Returns (loss, top1, top5)."""
    raise NotImplementedError


def confusion(model, loader, device, n_classes: int):
    raise NotImplementedError
