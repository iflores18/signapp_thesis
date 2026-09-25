"""Positional encoding is optional and owned by the caller.

The model stacks (Spatial -> Temporal) x N. PE belongs on the first temporal
cell only; these tests pin down that turning it off really removes it, really
changes the output, and does not weaken the masking guarantees.
"""

import torch

from models.temporal import TransformerTemporalCell

C = 16
V = 6
HEADS = 4


def make_cell(add_pos_encoding: bool, seed: int = 0) -> TransformerTemporalCell:
    torch.manual_seed(seed)
    cell = TransformerTemporalCell(
        C, C, n_heads=HEADS, dropout=0.0, add_pos_encoding=add_pos_encoding
    )
    cell.eval()
    return cell


def test_no_pe_module_when_disabled():
    """Not constructed, not merely skipped: an idle submodule still registers
    its buffer and inflates the per-configuration counts being reported."""
    cell = make_cell(False)

    assert cell.pos_encoding is None
    assert cell.add_pos_encoding is False
    assert not [k for k in cell.state_dict() if "pos_encoding" in k]
    assert not [n for n, _ in cell.named_buffers() if "pos_encoding" in n]


def test_pe_module_present_when_enabled():
    cell = make_cell(True)

    assert cell.pos_encoding is not None
    assert cell.add_pos_encoding is True
    assert [n for n, _ in cell.named_buffers() if "pos_encoding" in n]


def test_flag_changes_output():
    """Same weights, same input -- only the flag differs. If the outputs match,
    the flag is doing nothing."""
    with_pe = make_cell(True, seed=0)
    without_pe = make_cell(False, seed=0)

    torch.manual_seed(1)
    x = torch.randn(2, V, 9, C)
    mask = torch.ones(2, 9, dtype=torch.bool)

    out_with = with_pe(x, mask)
    out_without = without_pe(x, mask)

    assert out_with.shape == out_without.shape
    assert not torch.allclose(out_with, out_without, rtol=1e-5, atol=1e-6)


def test_padding_stays_zero_without_pe():
    """The re-zero guards must not have become conditional on PE."""
    cell = make_cell(False)
    N, T, real = 3, 12, 5

    x = torch.randn(N, V, T, C)
    mask = torch.zeros(N, T, dtype=torch.bool)
    mask[0, :real] = True
    mask[1:, :] = True

    out = cell(x, mask)

    assert torch.all(out[0, :, real:, :] == 0.0)
    assert not torch.all(out[0, :, :real, :] == 0.0)


def test_batch_invariance_without_pe():
    """The batch-invariance guarantee has to survive the flag too."""
    cell = make_cell(False)
    torch.manual_seed(1)
    short_len, long_len = 5, 12

    clip_short = torch.randn(V, short_len, C)
    clip_long = torch.randn(V, long_len, C)

    out_alone = cell(clip_short.unsqueeze(0), torch.ones(1, short_len, dtype=torch.bool))

    x_batch = torch.zeros(2, V, long_len, C)
    x_batch[0, :, :short_len] = clip_short
    x_batch[1] = clip_long
    mask_batch = torch.zeros(2, long_len, dtype=torch.bool)
    mask_batch[0, :short_len] = True
    mask_batch[1, :] = True
    out_batch = cell(x_batch, mask_batch)

    torch.testing.assert_close(
        out_alone[0], out_batch[0, :, :short_len], rtol=1e-5, atol=1e-6
    )


def test_parameter_count_unchanged():
    """PE is a buffer, not a parameter. If this fails, something that should be
    a constant is being learned."""
    with_pe = make_cell(True)
    without_pe = make_cell(False)

    assert sum(p.numel() for p in with_pe.parameters()) == sum(
        p.numel() for p in without_pe.parameters()
    )


def test_max_len_ignored_when_disabled():
    """A caller passing both is not making an error."""
    cell = TransformerTemporalCell(
        C, C, n_heads=HEADS, add_pos_encoding=False, max_len=32
    )
    cell.eval()

    x = torch.randn(1, V, 100, C)          # far beyond max_len=32
    out = cell(x, torch.ones(1, 100, dtype=torch.bool))

    assert out.shape == (1, V, 100, C)
