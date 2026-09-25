"""Shape and batch-agnosticism tests for the temporal cell."""

import pytest
import torch

from models.temporal import TransformerTemporalCell

C = 16
V = 6
HEADS = 4


def make_cell(dropout: float = 0.0) -> TransformerTemporalCell:
    torch.manual_seed(0)
    cell = TransformerTemporalCell(C, C, n_heads=HEADS, dropout=dropout)
    cell.eval()
    return cell


@pytest.mark.parametrize("N", [1, 7])
def test_batch_agnostic(N):
    """The cell must not care how many clips are in the batch."""
    cell = make_cell()
    T = 10
    x = torch.randn(N, V, T, C)
    mask = torch.ones(N, T, dtype=torch.bool)

    out = cell(x, mask)

    assert out.shape == (N, V, T, C)


@pytest.mark.parametrize("T", [1, 2, 3, 5, 12, 13, 101])
def test_shape_preserved(T):
    """T in == T out, for even, odd and degenerate lengths. Interleaving
    depends on this: the spatial block preserves V only if T is unchanged."""
    cell = make_cell()
    N = 3
    x = torch.randn(N, V, T, C)
    mask = torch.ones(N, T, dtype=torch.bool)

    out = cell(x, mask)

    assert out.shape == (N, V, T, C)
    assert out.shape == x.shape


def test_rejects_projection():
    """out_channels != in_channels is a mistake, not a silent reshape: the
    convs are C -> C and width is set by the spatial block upstream."""
    with pytest.raises(ValueError, match="does not project"):
        TransformerTemporalCell(C, 2 * C, n_heads=HEADS)


def test_rejects_even_kernel():
    with pytest.raises(ValueError, match="must be odd"):
        TransformerTemporalCell(C, C, n_heads=HEADS, kernel_size=4)


def test_rejects_mismatched_mask():
    cell = make_cell()
    x = torch.randn(2, V, 10, C)
    bad_mask = torch.ones(2, 9, dtype=torch.bool)
    with pytest.raises(ValueError, match="expected mask"):
        cell(x, bad_mask)


def test_parameter_count_reported():
    """Recorded so the BiLSTM arm can be matched to the same ballpark."""
    cell = TransformerTemporalCell(64, 64, n_heads=4, dropout=0.1)
    n_params = sum(p.numel() for p in cell.parameters())
    assert n_params > 0
    # the positional table is a buffer, not a parameter
    assert not any(name == "pos_encoding.pe" for name, _ in cell.named_parameters())
