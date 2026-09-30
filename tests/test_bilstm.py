"""BiLSTM temporal cell.

Mirrors the Transformer cell's suite. The failure modes are the same shape --
silent rather than loud -- with one extra: forgetting `total_length` shrinks T
only when the batch's longest clip is shorter than the padded length, so it
hides completely in any test where some clip happens to be full length.
"""

import pytest
import torch

from models.temporal import BiLSTMTemporalCell, expand_mask_to_joints

C = 16
V = 6


def make_cell(in_channels=C, out_channels=64, num_layers=1) -> BiLSTMTemporalCell:
    torch.manual_seed(0)
    cell = BiLSTMTemporalCell(in_channels, out_channels, num_layers=num_layers, dropout=0.0)
    cell.eval()
    return cell


# --- width contract ----------------------------------------------------------

def test_width_contract():
    """Bidirectional concatenation means out_channels = 2 * hidden."""
    cell = make_cell(out_channels=64)

    assert cell.hidden == 32
    assert cell.out_channels == 64

    out = cell(torch.randn(2, V, 7, C), torch.ones(2, 7, dtype=torch.bool))
    assert out.shape == (2, V, 7, 64)


def test_odd_out_channels_raises():
    with pytest.raises(ValueError, match="must be even"):
        BiLSTMTemporalCell(C, 65)


def test_projection_is_legal():
    """Unlike the Transformer cell, this one changes width by nature."""
    cell = make_cell(in_channels=8, out_channels=64)
    out = cell(torch.randn(2, V, 5, 8), torch.ones(2, 5, dtype=torch.bool))
    assert out.shape == (2, V, 5, 64)


def test_has_no_transformer_machinery():
    """Nothing borrowed from the other arm: no PE, no conv, no internal norm."""
    import torch.nn as nn

    cell = make_cell()
    assert not hasattr(cell, "pos_encoding") or cell.pos_encoding is None
    assert not any(isinstance(m, nn.Conv1d) for m in cell.modules())
    assert not any(isinstance(m, nn.LayerNorm) for m in cell.modules())


# --- masking -----------------------------------------------------------------

def test_padding_invariance():
    """The test that catches an unpacked LSTM or mis-expanded lengths.

    Unpacked, the backward pass starts inside the padding, so the padded copy of
    a clip gets different numbers from the same clip run alone.
    """
    cell = make_cell()
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


def test_padding_stays_zero():
    cell = make_cell()
    N, T, real = 3, 12, 5

    x = torch.randn(N, V, T, C)
    mask = torch.zeros(N, T, dtype=torch.bool)
    mask[0, :real] = True
    mask[1:, :] = True

    out = cell(x, mask)

    assert torch.all(out[0, :, real:, :] == 0.0)
    assert not torch.all(out[0, :, :real, :] == 0.0)


def test_rejects_fully_padded_clip():
    cell = make_cell()
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, :] = False

    with pytest.raises(ValueError, match="at least one real frame"):
        cell(torch.randn(2, V, 6, C), mask)


def test_length_expansion_alignment():
    """Row i*V + j must carry clip i's length, matching the tensor fold."""
    N, T = 3, 9
    mask = torch.zeros(N, T, dtype=torch.bool)
    mask[0, :2] = True
    mask[1, :5] = True
    mask[2, :] = True

    joint_lengths = expand_mask_to_joints(mask, V).sum(dim=1)
    lengths = mask.sum(dim=1)

    assert joint_lengths.shape == (N * V,)
    for i in range(N):
        for j in range(V):
            assert joint_lengths[i * V + j] == lengths[i], f"row {i * V + j} misaligned"


# --- shapes ------------------------------------------------------------------

@pytest.mark.parametrize("N", [1, 7])
def test_batch_agnostic(N):
    cell = make_cell()
    T = 10
    out = cell(torch.randn(N, V, T, C), torch.ones(N, T, dtype=torch.bool))
    assert out.shape == (N, V, T, 64)


@pytest.mark.parametrize("T", [1, 2, 5, 12, 13, 101])
def test_shape_preserved(T):
    cell = make_cell()
    N = 3
    out = cell(torch.randn(N, V, T, C), torch.ones(N, T, dtype=torch.bool))
    assert out.shape == (N, V, T, 64)


def test_total_length_regression():
    """T is padded to 12 but no clip exceeds 7 frames.

    Without total_length=T, pad_packed_sequence pads only to the longest
    sequence present -- 7 -- and the returned tensor silently changes shape,
    breaking the seam residual in model.py.
    """
    cell = make_cell()
    N, T = 3, 12
    mask = torch.zeros(N, T, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, :7] = True
    mask[2, :3] = True

    out = cell(torch.randn(N, V, T, C), mask)

    assert out.shape == (N, V, T, 64), "T was not preserved: total_length missing?"
    assert torch.all(out[1, :, 7:, :] == 0.0)


def test_num_layers_two():
    cell = make_cell(num_layers=2)
    assert cell.lstm.num_layers == 2

    N, T, real = 2, 9, 4
    mask = torch.zeros(N, T, dtype=torch.bool)
    mask[0, :real] = True
    mask[1, :] = True

    out = cell(torch.randn(N, V, T, C), mask)

    assert out.shape == (N, V, T, 64)
    assert torch.all(out[0, :, real:, :] == 0.0)


def test_no_dropout_warning_at_one_layer(recwarn):
    """nn.LSTM warns when dropout is set at num_layers=1, where it does nothing."""
    BiLSTMTemporalCell(C, 64, num_layers=1, dropout=0.5)
    assert not [w for w in recwarn if "dropout" in str(w.message).lower()]
