"""Depth parameters: N, L_spatial and L_temporal must stay independent.

N=3/L_temporal=1 and N=1/L_temporal=3 hold the same number of attention layers
and are different architectures. These tests pin that down structurally, so a
later refactor cannot quietly collapse one into another.
"""

import pytest
import torch
import torch.nn as nn

from models.model import STModel
from models.temporal import TransformerTemporalCell

C_IN, HIDDEN, V, HEADS, N_CLASSES = 3, 16, 6, 4, 5


class DummySpatial(nn.Module):
    """Minimal stand-in with the real spatial signature -- GCN and GAT are still
    stubs, but the block wiring they plug into is testable without them."""

    def __init__(self, in_channels, out_channels, A, num_layers=1, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            nn.Linear(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_layers)
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x)
        return x


def make_model(n_blocks=2, L_spatial=1, L_temporal=1):
    torch.manual_seed(0)
    model = STModel(
        spatial_cls=DummySpatial,
        temporal_cls=TransformerTemporalCell,
        n_blocks=n_blocks,
        C_in=C_IN,
        hidden=HIDDEN,
        n_classes=N_CLASSES,
        A=torch.eye(V),
        L_spatial=L_spatial,
        L_temporal=L_temporal,
        n_heads=HEADS,
        dropout=0.0,
    )
    model.eval()
    return model


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


# --- the three parameters are independent -----------------------------------

def test_depths_are_independent():
    model = make_model(n_blocks=3, L_spatial=2, L_temporal=4)

    assert model.n_blocks == 3 and len(model.spatial) == 3 and len(model.temporal) == 3
    assert all(s.num_layers == 2 for s in model.spatial)
    assert all(t.num_layers == 4 for t in model.temporal)


def test_same_attention_layers_different_architecture():
    """Same attention-layer budget, different arrangement, different model."""
    interleaved = make_model(n_blocks=3, L_temporal=1)
    pipeline = make_model(n_blocks=1, L_temporal=3)

    assert len(interleaved.temporal) == 3 and interleaved.temporal[0].num_layers == 1
    assert len(pipeline.temporal) == 1 and pipeline.temporal[0].num_layers == 3
    assert n_params(interleaved) != n_params(pipeline)


# --- internal depth does not multiply the brackets --------------------------

def test_num_layers_stacks_encoder_only():
    """k encoder layers, but still exactly 2 convs and at most 1 PE module --
    the convs bracket the attention stack, not each layer."""
    for k in (1, 2, 3):
        cell = TransformerTemporalCell(HIDDEN, HIDDEN, n_heads=HEADS, num_layers=k)

        assert len(cell.encoder.layers) == k
        assert isinstance(cell.conv_pre, nn.Conv1d)
        assert isinstance(cell.conv_post, nn.Conv1d)
        assert sum(1 for m in cell.modules() if isinstance(m, nn.Conv1d)) == 2
        assert sum(1 for n, _ in cell.named_buffers() if "pos_encoding" in n) <= 1


def test_padding_zero_at_depth():
    torch.manual_seed(0)
    cell = TransformerTemporalCell(HIDDEN, HIDDEN, n_heads=HEADS, num_layers=3, dropout=0.0)
    cell.eval()
    T, real = 12, 5

    x = torch.randn(2, V, T, HIDDEN)
    mask = torch.zeros(2, T, dtype=torch.bool)
    mask[0, :real] = True
    mask[1, :] = True

    out = cell(x, mask)

    assert torch.all(out[0, :, real:, :] == 0.0)
    assert not torch.all(out[0, :, :real, :] == 0.0)


# --- block wiring ------------------------------------------------------------

def test_positional_encoding_on_first_block_only():
    model = make_model(n_blocks=3)

    assert model.temporal[0].pos_encoding is not None
    assert all(t.pos_encoding is None for t in model.temporal[1:])


def test_blocks_do_not_share_weights():
    model = make_model(n_blocks=2)

    assert model.spatial[0] is not model.spatial[1]
    assert model.temporal[0] is not model.temporal[1]
    assert id(model.spatial[0]) != id(model.spatial[1])

    for (n0, p0), (n1, p1) in zip(
        model.spatial[0].named_parameters(), model.spatial[1].named_parameters()
    ):
        assert n0 == n1
        assert p0 is not p1
        assert p0.data_ptr() != p1.data_ptr()


def test_model_forward_shapes_and_padding():
    model = make_model(n_blocks=2, L_spatial=2, L_temporal=2)
    N, T, real = 3, 10, 4

    x = torch.randn(N, T, V, C_IN)
    mask = torch.zeros(N, T, dtype=torch.bool)
    mask[0, :real] = True
    mask[1:, :] = True
    x = x.masked_fill(~mask[:, :, None, None], 0.0)

    logits = model(x, mask)

    assert logits.shape == (N, N_CLASSES)
    assert torch.isfinite(logits).all()


# --- seam projection rule ----------------------------------------------------


class DummySpatialHeads(DummySpatial):
    """Stands in for GAT: same contract, but takes n_heads like the real one."""

    def __init__(self, in_channels, out_channels, A, num_layers=1, n_heads=4, dropout=0.1):
        super().__init__(in_channels, out_channels, A, num_layers=num_layers, dropout=dropout)


class DummyTemporal(nn.Module):
    """Stands in for the BiLSTM cell: width-preserving, no add_pos_encoding."""

    def __init__(self, in_channels, out_channels, num_layers=1, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x, mask):
        return self.proj(x)


def test_seam_projection_rule_is_uniform_across_configurations():
    """The rule -- project iff the widths differ -- must be the same in all four
    configurations. The modules it produces need not be, and at blocks 1+ every
    configuration already matches, so a projection there prevents no confound.
    """
    decisions = {}
    for spatial_cls in (DummySpatial, DummySpatialHeads):
        for temporal_cls in (TransformerTemporalCell, DummyTemporal):
            torch.manual_seed(0)
            model = STModel(
                spatial_cls=spatial_cls,
                temporal_cls=temporal_cls,
                n_blocks=3,
                C_in=C_IN,
                hidden=HIDDEN,
                n_classes=N_CLASSES,
                A=torch.eye(V),
                n_heads=HEADS,
                dropout=0.0,
            )
            key = (spatial_cls.__name__, temporal_cls.__name__)
            decisions[key] = [type(p).__name__ for p in model.seam_proj]

    assert len(set(map(tuple, decisions.values()))) == 1, decisions
    assert next(iter(decisions.values())) == ["Linear", "Identity", "Identity"]


def test_seam_projection_absent_where_widths_match():
    """C_in == hidden means no projection anywhere, including block 0."""
    torch.manual_seed(0)
    model = STModel(
        spatial_cls=DummySpatial,
        temporal_cls=TransformerTemporalCell,
        n_blocks=2,
        C_in=HIDDEN,          # same width as hidden
        hidden=HIDDEN,
        n_classes=N_CLASSES,
        A=torch.eye(V),
        n_heads=HEADS,
        dropout=0.0,
    )
    assert [type(p).__name__ for p in model.seam_proj] == ["Identity", "Identity"]


# --- readout -----------------------------------------------------------------


def test_readout_shape():
    model = make_model(n_blocks=2)
    N, T = 4, 9
    x = torch.randn(N, T, V, C_IN)
    mask = torch.ones(N, T, dtype=torch.bool)

    assert model(x, mask).shape == (N, N_CLASSES)


def test_readout_ignores_padding():
    """Same clip alone and padded inside a longer batch must give the same
    logits. If padding reaches the max, this diverges."""
    model = make_model(n_blocks=2)
    torch.manual_seed(3)
    short_len, long_len = 5, 12

    clip = torch.randn(short_len, V, C_IN)
    other = torch.randn(long_len, V, C_IN)

    logits_alone = model(clip.unsqueeze(0), torch.ones(1, short_len, dtype=torch.bool))

    x = torch.zeros(2, long_len, V, C_IN)
    x[0, :short_len] = clip
    x[1] = other
    mask = torch.zeros(2, long_len, dtype=torch.bool)
    mask[0, :short_len] = True
    mask[1, :] = True
    logits_batch = model(x, mask)

    torch.testing.assert_close(logits_alone[0], logits_batch[0], rtol=1e-5, atol=1e-6)


def test_readout_rejects_fully_padded_row():
    """A clip with no real frames is a collate_fn bug and must fail loudly,
    not return -inf on every channel."""
    model = make_model(n_blocks=1)
    N, T = 2, 6
    x = torch.zeros(N, T, V, C_IN)
    mask = torch.ones(N, T, dtype=torch.bool)
    mask[1, :] = False          # clip 1 is entirely padding

    with pytest.raises(ValueError, match="at least one real frame"):
        model(x, mask)
