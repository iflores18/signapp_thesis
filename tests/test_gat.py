"""GAT spatial cell.

The structural properties of alpha are asserted directly rather than inferred
from the output: a neighbourhood mask that silently stops working leaves a model
that still trains, just on a complete graph.
"""

import pytest
import torch
import torch.nn.functional as F

from models.graph import N_NODES, build_adjacency
from models.spatial import GAT

C_IN, C_OUT, HEADS = 3, 64, 4
V = N_NODES


def make_cell(in_channels=C_IN, out_channels=C_OUT, **kwargs) -> GAT:
    torch.manual_seed(0)
    cell = GAT(in_channels, out_channels, n_heads=kwargs.pop("n_heads", HEADS),
               dropout=0.0, **kwargs)
    cell.eval()
    return cell


# --- shapes ------------------------------------------------------------------


@pytest.mark.parametrize("T", [1, 2, 5, 12, 13, 101])
def test_shape_preserved(T):
    cell = make_cell()
    out = cell(torch.randn(3, T, V, C_IN), torch.ones(3, T, dtype=torch.bool))
    assert out.shape == (3, T, V, C_OUT)


@pytest.mark.parametrize("N", [1, 7])
def test_batch_agnostic(N):
    cell = make_cell()
    out = cell(torch.randn(N, 6, V, C_IN), torch.ones(N, 6, dtype=torch.bool))
    assert out.shape == (N, 6, V, C_OUT)


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_output_width_independent_of_heads(n_heads):
    """concat=False must mean averaging, not concatenation: GAT's width has to
    match GCN's or channel dimensionality becomes a confound."""
    cell = make_cell(n_heads=n_heads)
    out = cell(torch.randn(2, 4, V, C_IN), torch.ones(2, 4, dtype=torch.bool))
    assert out.shape[-1] == C_OUT


def test_no_edge_importance_parameter():
    """edge_importance is GCN-only; alpha already weights edges, per frame."""
    cell = make_cell()
    assert cell.edge_importance is None
    assert not [k for k in cell.state_dict() if "edge_importance" in k]
    assert "edge_importance" not in GAT.__init__.__code__.co_varnames


def test_no_bias():
    """GAT Eq. 1-6 have no bias term."""
    cell = make_cell()
    assert not [k for k in cell.state_dict() if "bias" in k]


# --- attention coefficients ---------------------------------------------------


def alpha_of(cell: GAT, h: torch.Tensor, layer: int = 0) -> torch.Tensor:
    with torch.no_grad():
        return cell.attention(h, layer)


def test_alpha_rows_sum_to_one():
    cell = make_cell()
    h = torch.randn(8, V, C_IN)

    alpha = alpha_of(cell, h)

    assert alpha.shape == (8, HEADS, V, V)
    torch.testing.assert_close(
        alpha.sum(dim=-1), torch.ones(8, HEADS, V), rtol=1e-5, atol=1e-6
    )


def test_alpha_is_exactly_zero_off_the_graph():
    """Structural, not empirical: -inf before the softmax gives exact zeros, so
    no amount of training can put weight on a non-edge."""
    cell = make_cell()
    alpha = alpha_of(cell, torch.randn(4, V, C_IN))

    non_edge = ~cell.support.bool()
    assert non_edge.any()
    assert torch.all(alpha[..., non_edge] == 0.0)


def test_self_loops_are_attended():
    """N(i) includes i, per the paper. If the diagonal were masked out, nodes
    would lose their own features at every hop."""
    cell = make_cell()
    alpha = alpha_of(cell, torch.randn(4, V, C_IN))

    diagonal = alpha[..., torch.arange(V), torch.arange(V)]
    assert torch.all(diagonal > 0.0)


def test_neighbourhood_mask_is_load_bearing():
    """Without the mask, every node attends over all V. Confirms the mask
    actually changes the result rather than being incidentally satisfied."""
    cell = make_cell()
    h = torch.randn(4, V, C_IN)

    with torch.no_grad():
        Wh = cell._transform(h, 0)
        e = F.leaky_relu(
            Wh @ cell.att_center[0] + (Wh @ cell.att_neighbor[0]).transpose(-2, -1),
            negative_slope=0.2,
        )
        unmasked = e.softmax(dim=-1)
    masked = alpha_of(cell, h)

    assert not torch.allclose(masked, unmasked)
    # unmasked attention is spread over all 75 nodes; masked over ~3
    assert (unmasked > 1e-6).float().sum(-1).mean() > 10 * (
        masked > 1e-6
    ).float().sum(-1).mean()


# --- masking ------------------------------------------------------------------


def test_padding_stays_zero():
    cell = make_cell()
    N, T, real = 3, 12, 5
    mask = torch.zeros(N, T, dtype=torch.bool)
    mask[0, :real] = True
    mask[1:, :] = True

    out = cell(torch.randn(N, T, V, C_IN), mask)

    assert torch.all(out[0, real:] == 0.0)
    assert not torch.all(out[0, :real] == 0.0)


def test_padding_propagates_zero_without_the_re_zero():
    """With no bias the zeros survive the layer unaided: Wh = 0 -> e = 0 ->
    uniform softmax -> alpha @ 0 = 0 -> ELU(0) = 0. The base's re-zero is
    defensive here, unlike in GCN."""
    cell = make_cell()
    zeros = torch.zeros(1, V, C_IN)

    with torch.no_grad():
        out = cell._layer(zeros, 0)

    assert torch.all(out == 0.0)


def test_padding_invariance():
    cell = make_cell()
    torch.manual_seed(1)
    short_len, long_len = 5, 12

    clip_short = torch.randn(short_len, V, C_IN)
    clip_long = torch.randn(long_len, V, C_IN)

    out_alone = cell(clip_short.unsqueeze(0), torch.ones(1, short_len, dtype=torch.bool))

    x_batch = torch.zeros(2, long_len, V, C_IN)
    x_batch[0, :short_len] = clip_short
    x_batch[1] = clip_long
    mask_batch = torch.zeros(2, long_len, dtype=torch.bool)
    mask_batch[0, :short_len] = True
    mask_batch[1, :] = True
    out_batch = cell(x_batch, mask_batch)

    torch.testing.assert_close(
        out_alone[0], out_batch[0, :short_len], rtol=1e-5, atol=1e-6
    )


def test_rejects_fully_padded_clip():
    cell = make_cell()
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, :] = False
    with pytest.raises(ValueError, match="at least one real frame"):
        cell(torch.randn(2, 6, V, C_IN), mask)


def test_support_includes_self_loops():
    """The invariant the construction-time check protects.

    `support` is built as (A + I) > 0, so with the current graph module the
    check cannot fire -- it is a tripwire for a later change to how the support
    is derived, which is exactly the change that would silently let a node
    softmax over an all--inf row.
    """
    cell = make_cell()

    assert torch.all(cell.support.diagonal() > 0)
    assert torch.all(cell.support.any(dim=1)), "a node with no unmasked entry"


def test_self_loop_check_fires_when_the_diagonal_is_missing():
    """Break the support the way a future refactor might, and confirm the guard
    catches it rather than producing NaN rows."""
    import unittest.mock

    A = build_adjacency()
    # a support derived without self-loops
    with unittest.mock.patch(
        "models.spatial.torch.eye", return_value=torch.zeros(A.shape[0], A.shape[0])
    ):
        with pytest.raises(ValueError, match="self-loops"):
            GAT(C_IN, C_OUT, A=A)


# --- gradients and depth -------------------------------------------------------


def test_all_heads_receive_gradient():
    """Averaging must not leave a head unused."""
    torch.manual_seed(0)
    cell = GAT(C_IN, C_OUT, n_heads=HEADS, dropout=0.0)
    cell.train()

    out = cell(torch.randn(2, 4, V, C_IN), torch.ones(2, 4, dtype=torch.bool))
    out.pow(2).mean().backward()

    for name, param in [
        ("weight", cell.weight[0]),
        ("att_center", cell.att_center[0]),
        ("att_neighbor", cell.att_neighbor[0]),
    ]:
        for head in range(HEADS):
            g = param.grad[head]
            assert g.abs().max() > 0, f"{name} head {head} received no gradient"


def test_num_layers_two():
    cell = make_cell(num_layers=2)

    assert len(cell.weight) == 2
    assert len(cell.att_center) == 2
    assert len(cell.att_neighbor) == 2
    assert cell.weight[0].shape == (HEADS, C_IN, C_OUT)
    assert cell.weight[1].shape == (HEADS, C_OUT, C_OUT)
    assert not torch.allclose(cell.att_center[0].flatten()[:1], cell.att_center[1].flatten()[:1])

    out = cell(torch.randn(2, 6, V, C_IN), torch.ones(2, 6, dtype=torch.bool))
    assert out.shape == (2, 6, V, C_OUT)


# --- reference implementations --------------------------------------------------


def test_matches_pyg_gatconv():
    """One layer against torch_geometric.nn.GATConv on the real skeleton.

    Note the naming swap: PyG's att_src pairs with the *source* node of an edge
    j -> i, which is the neighbour whose features flow; att_dst pairs with the
    target, the centre. Ours are named for that role directly.
    """
    pyg_nn = pytest.importorskip("torch_geometric.nn")

    torch.manual_seed(0)
    A = build_adjacency()
    edge_index = torch.nonzero(A).t().contiguous()

    ours = GAT(C_IN, C_OUT, A=A, num_layers=1, n_heads=HEADS, dropout=0.0)
    ours.eval()
    theirs = pyg_nn.GATConv(
        C_IN, C_OUT, heads=HEADS, concat=False, dropout=0.0, bias=False,
        add_self_loops=True, negative_slope=0.2,
    )
    theirs.eval()

    with torch.no_grad():
        for k in range(HEADS):
            theirs.lin.weight[k * C_OUT : (k + 1) * C_OUT, :].copy_(ours.weight[0][k].t())
        theirs.att_src[0].copy_(ours.att_neighbor[0][:, :, 0])
        theirs.att_dst[0].copy_(ours.att_center[0][:, :, 0])

    x = torch.randn(1, 1, V, C_IN)
    got = ours(x, torch.ones(1, 1, dtype=torch.bool))[0, 0]
    expected = F.elu(theirs(x[0, 0], edge_index))

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_matches_hand_computed_toy_graph():
    """Dependency-free: one head, one channel, 3-node path graph 0-1-2.

    Node 1 has N(1) = {0, 1, 2}, so alpha_1. is computed by hand from the
    LeakyReLU scores and checked against the layer.
    """
    A = torch.zeros(3, 3)
    for i, j in [(0, 1), (1, 2)]:
        A[i, j] = A[j, i] = 1.0

    cell = GAT(1, 1, A=A, num_layers=1, n_heads=1, dropout=0.0)
    cell.eval()
    with torch.no_grad():
        cell.weight[0].copy_(torch.tensor([[[2.0]]]))        # W = 2
        cell.att_center[0].copy_(torch.tensor([[[1.0]]]))    # a_centre = 1
        cell.att_neighbor[0].copy_(torch.tensor([[[-1.0]]]))  # a_neighbour = -1

    x = torch.tensor([[1.0], [2.0], [3.0]])                  # Wh = [2, 4, 6]

    # e[1, j] = leaky_relu(1 * Wh_1 + (-1) * Wh_j) = leaky_relu(4 - Wh_j)
    Wh = torch.tensor([2.0, 4.0, 6.0])
    scores = F.leaky_relu(4.0 - Wh, negative_slope=0.2)       # [2, 0, -0.4]
    alpha_1 = scores.softmax(dim=-1)
    expected_node1 = F.elu((alpha_1 * Wh).sum())

    got = cell(x.view(1, 1, 3, 1), torch.ones(1, 1, dtype=torch.bool))[0, 0]

    torch.testing.assert_close(got[1, 0], expected_node1)

    # node 0 has N(0) = {0, 1} only -- node 2 must not contribute
    scores_0 = F.leaky_relu(2.0 - Wh[:2], negative_slope=0.2)
    alpha_0 = scores_0.softmax(dim=-1)
    expected_node0 = F.elu((alpha_0 * Wh[:2]).sum())
    torch.testing.assert_close(got[0, 0], expected_node0)
