"""Dataset and batching.

Clips keep their real length -- there is no resampling. Sequences are padded to
the longest clip in the batch and a boolean mask travels alongside the tensor so
every downstream component can tell real frames from padding.
"""

from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class SkeletonDataset(Dataset):
    """Yields (clip, label) with clip of shape [T_i, V, C]; T_i varies."""

    def __init__(self, manifest_path, keypoints_dir, split: str):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError


def collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of variable-length clips into one batch.

    Returns:
        x:     [N, T, V, C] float, padded with zeros
        mask:  [N, T] bool, True = real frame
        label: [N] long
    """
    raise NotImplementedError
