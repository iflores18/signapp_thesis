"""Masking tests.

These target failures that are silent rather than loud: padding leaking into
real frames produces no error, no shape mismatch, and no obvious symptom -- only
slightly wrong numbers that get worse with depth.
"""

import torch

from models.temporal import TransformerTemporalCell, expand_mask_to_joints

C = 16
V = 6
HEADS = 4


def make_cell() -> TransformerTemporalCell:
    torch.manual_seed(0)
    cell = TransformerTemporalCell(C, C, n_heads=HEADS, dropout=0.0)
    cell.eval()  # dropout off, so two runs of the same input agree exactly
    return cell


def test_padding_stays_zero():
    """Every stage that can write into a padded position must be followed by a
    re-zero: the positional encoding, both convs, and attention."""
    cell = make_cell()
    N, T, real = 3, 12, 5

    x = torch.randn(N, V, T, C)
    mask = torch.zeros(N, T, dtype=torch.bool)
    mask[0, :real] = True
    mask[1:, :] = True

    out = cell(x, mask)

    assert torch.all(out[0, :, real:, :] == 0.0), "padding is not exactly zero"
    assert not torch.all(out[0, :, :real, :] == 0.0), "real frames were zeroed too"


def test_batch_invariance_under_padding():
    """The highest-value test here.

    A clip run alone must produce identical numbers to the same clip run in a
    batch where it is padded. If padding leaks through either conv or through
    attention, this diverges -- typically at the last few real frames first.
    """
    cell = make_cell()
    torch.manual_seed(1)
    short_len, long_len = 5, 12

    clip_short = torch.randn(V, short_len, C)
    clip_long = torch.randn(V, long_len, C)

    # alone: no padding at all
    x_alone = clip_short.unsqueeze(0)
    mask_alone = torch.ones(1, short_len, dtype=torch.bool)
    out_alone = cell(x_alone, mask_alone)

    # batched: the same clip, now padded out to the longer one
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


def test_batch_invariance_is_sensitive():
    """Guard the guard: if the mask is ignored, the previous test must fail.

    Running the padded clip with an all-True mask is what a missing mask looks
    like, and it has to produce different numbers -- otherwise
    test_batch_invariance_under_padding would pass even with broken masking.
    """
    cell = make_cell()
    torch.manual_seed(1)
    short_len, long_len = 5, 12

    clip_short = torch.randn(V, short_len, C)
    x_padded = torch.zeros(1, V, long_len, C)
    x_padded[0, :, :short_len] = clip_short

    correct = cell(x_padded, torch.cat([
        torch.ones(1, short_len, dtype=torch.bool),
        torch.zeros(1, long_len - short_len, dtype=torch.bool),
    ], dim=1))
    ignored = cell(x_padded, torch.ones(1, long_len, dtype=torch.bool))

    assert not torch.allclose(
        correct[0, :, :short_len], ignored[0, :, :short_len], rtol=1e-5, atol=1e-6
    ), "masking made no difference -- the mask is not being used"


def test_mask_expansion_alignment():
    """Row i*V + j of the folded tensor must carry clip i's mask.

    repeat_interleave gives [clip0]*V, [clip1]*V, ...; repeat would give
    [clip0, clip1, ...]*V. Both have shape [N*V, T], so getting it wrong pairs
    sequences with other clips' masks and raises nothing.
    """
    N, T = 3, 7
    mask = torch.zeros(N, T, dtype=torch.bool)
    mask[0, :2] = True
    mask[1, :5] = True
    mask[2, :] = True

    expanded = expand_mask_to_joints(mask, V)
    assert expanded.shape == (N * V, T)

    for i in range(N):
        for j in range(V):
            assert torch.equal(expanded[i * V + j], mask[i]), f"row {i * V + j} misaligned"

    # and the same indexing must hold for the tensor reshape the cell uses
    x = torch.arange(N, dtype=torch.float32).view(N, 1, 1, 1).expand(N, V, T, C)
    folded = x.reshape(N * V, T, C)
    for i in range(N):
        for j in range(V):
            assert torch.all(folded[i * V + j] == i), f"tensor row {i * V + j} misaligned"


def test_mask_expansion_would_catch_repeat():
    """The failure mode above, made explicit."""
    N, T = 3, 7
    mask = torch.zeros(N, T, dtype=torch.bool)
    mask[0, :2] = True
    mask[1, :5] = True
    mask[2, :] = True

    wrong = mask.repeat(V, 1)
    assert wrong.shape == expand_mask_to_joints(mask, V).shape  # same shape, silent
    assert not torch.equal(wrong, expand_mask_to_joints(mask, V))


def test_rejects_fully_padded_clip():
    """A clip with no real frames masks out every key.

    What that produces downstream is version-dependent -- raw MultiheadAttention
    returns NaN, nn.TransformerEncoderLayer currently absorbs it, the readout
    would give -inf -- so the invariant is checked directly rather than trusting
    any one symptom. No correct collate_fn produces such a clip; the guard exists
    so a later change to padding fails on its first bad batch."""
    import pytest

    cell = make_cell()
    N, T = 2, 6
    x = torch.randn(N, V, T, C)
    mask = torch.ones(N, T, dtype=torch.bool)
    mask[1, :] = False

    with pytest.raises(ValueError, match="at least one real frame"):
        cell(x, mask)
