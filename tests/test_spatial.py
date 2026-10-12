"""GCN spatial cell.

The failure modes here are silent: a wrong self-loop ordering, an adjacency
that drifts during training, or edge importance leaking onto non-edges all
produce a model that trains and converges to something slightly wrong. So the
arithmetic is checked against an independent reference rather than trusted.
"""

import pytest
import torch

from models.graph import (
    N_NODES,
    build_adjacency,
    build_edges,
    normalize_adjacency,
)
from models.spatial import GCN

C_IN, C_OUT = 3, 64
V = N_NODES


def make_cell(in_channels=C_IN, out_channels=C_OUT, **kwargs) -> GCN:
    torch.manual_seed(0)
    cell = GCN(in_channels, out_channels, dropout=0.0, **kwargs)
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


# --- masking -----------------------------------------------------------------


def test_padding_stays_zero_with_bias():
    """With bias=True a padded frame of zeros does not stay zero on its own:
    the bias is added and ReLU keeps what is positive."""
    cell = make_cell(bias=True)
    # a bias of exactly zero would make this pass for the wrong reason
    with torch.no_grad():
        for b in cell.bias:
            b.fill_(0.5)

    N, T, real = 3, 12, 5
    mask = torch.zeros(N, T, dtype=torch.bool)
    mask[0, :real] = True
    mask[1:, :] = True

    out = cell(torch.randn(N, T, V, C_IN), mask)

    assert torch.all(out[0, real:] == 0.0)
    assert not torch.all(out[0, :real] == 0.0)


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


# --- the adjacency is a constant ---------------------------------------------


def test_adjacency_is_not_a_parameter():
    cell = make_cell()

    assert "A_hat" in dict(cell.named_buffers())
    assert not any("A_hat" in n for n, _ in cell.named_parameters())

    before = cell.A_hat.clone()
    cell.train()
    opt = torch.optim.SGD(cell.parameters(), lr=1.0)
    out = cell(torch.randn(2, 4, V, C_IN), torch.ones(2, 4, dtype=torch.bool))
    out.sum().backward()
    opt.step()

    torch.testing.assert_close(cell.A_hat, before, rtol=0, atol=0)


def test_self_loops_added_before_normalisation():
    """Normalising first and adding I afterwards gives a different matrix, and
    nothing about the result looks wrong -- hence an explicit check."""
    # path graph 0-1-2-3
    A = torch.zeros(4, 4)
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        A[i, j] = A[j, i] = 1.0

    got = normalize_adjacency(A)

    # hand-computed D~^-1/2 (A + I) D~^-1/2; degrees with self-loops are 2,3,3,2
    d = torch.tensor([2.0, 3.0, 3.0, 2.0]).pow(-0.5)
    A_tilde = A + torch.eye(4)
    expected = d.unsqueeze(1) * A_tilde * d.unsqueeze(0)
    torch.testing.assert_close(got, expected)

    assert got[0, 0] == pytest.approx(0.5)              # 1/sqrt(2*2)
    assert got[0, 1] == pytest.approx(1 / (2 * 3) ** 0.5)

    # the wrong order: normalise A, then add I
    d_wrong = A.sum(1).pow(-0.5)
    wrong = d_wrong.unsqueeze(1) * A * d_wrong.unsqueeze(0) + torch.eye(4)
    assert not torch.allclose(got, wrong)


# --- learnable edge importance ------------------------------------------------


def test_edge_importance_absent_when_disabled():
    off = make_cell(edge_importance=False)
    on = make_cell(edge_importance=True)

    assert off.edge_importance is None
    assert not [k for k in off.state_dict() if "edge_importance" in k]
    assert [k for k in on.state_dict() if "edge_importance" in k]

    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    assert n_on == n_off + V * V


def test_edge_importance_starts_as_identity():
    """Ones initialisation means the flag changes nothing at step zero."""
    off = make_cell(edge_importance=False)
    on = make_cell(edge_importance=True)
    with torch.no_grad():
        for a, b in zip(on.weight, off.weight):
            a.copy_(b)
        for a, b in zip(on.bias, off.bias):
            a.copy_(b)

    x = torch.randn(2, 5, V, C_IN)
    mask = torch.ones(2, 5, dtype=torch.bool)

    torch.testing.assert_close(on(x, mask), off(x, mask))


def test_edge_importance_stays_off_non_edges():
    """Training must not create weight on a non-edge; otherwise the model
    quietly builds a denser graph than the skeleton."""
    torch.manual_seed(0)
    cell = GCN(C_IN, C_OUT, num_layers=2, edge_importance=True, dropout=0.0)
    cell.train()
    opt = torch.optim.SGD(cell.parameters(), lr=0.1)
    before = cell.edge_importance[0].detach().clone()

    for _ in range(5):
        opt.zero_grad()
        out = cell(torch.randn(2, 4, V, C_IN), torch.ones(2, 4, dtype=torch.bool))
        # A bounded objective. Descending on out.sum() is unbounded below and
        # diverges within a few steps, which tests the optimiser rather than
        # the masking.
        out.pow(2).mean().backward()
        opt.step()

    non_edge = cell.support == 0
    assert non_edge.any()
    # the edges did move, so the run was not a no-op
    assert (cell.edge_importance[0][~non_edge] - before[~non_edge]).abs().max() > 0

    for layer in range(cell.num_layers):
        A_l = cell._adjacency(layer)
        assert torch.isfinite(A_l).all(), f"layer {layer} went non-finite"
        assert torch.all(A_l[non_edge] == 0.0), f"layer {layer} leaked onto non-edges"
        assert torch.all(
            cell.edge_importance[layer][non_edge] == 1.0
        ), f"layer {layer} moved M off its initialisation at a non-edge"


def test_edge_importance_gets_no_gradient_on_non_edges():
    """The structural reason the previous test holds: a non-edge entry of M is
    multiplied by a zero in A_hat, so it receives exactly zero gradient and
    never moves from its initialised 1.0.

    Note the limit of that guarantee -- it is about gradients, not about
    arithmetic. If training diverges and every parameter becomes NaN, then
    0 * NaN = NaN and the masked product is NaN rather than zero. Masking
    confines what the model can *learn*; it does not sanitise a blown-up run.
    """
    torch.manual_seed(0)
    cell = GCN(C_IN, C_OUT, num_layers=2, edge_importance=True, dropout=0.0)
    cell.train()
    cell(torch.randn(2, 4, V, C_IN), torch.ones(2, 4, dtype=torch.bool)).sum().backward()

    non_edge = cell.support == 0
    for layer, M in enumerate(cell.edge_importance):
        assert torch.all(M.grad[non_edge] == 0.0), f"layer {layer} non-edges got gradient"
        assert M.grad[~non_edge].abs().max() > 0, f"layer {layer} edges got no gradient"


def test_num_layers_two():
    cell = make_cell(num_layers=2, edge_importance=True)

    assert len(cell.weight) == 2
    assert len(cell.bias) == 2
    assert len(cell.edge_importance) == 2
    assert cell.weight[0].shape == (C_IN, C_OUT)
    assert cell.weight[1].shape == (C_OUT, C_OUT)

    out = cell(torch.randn(2, 6, V, C_IN), torch.ones(2, 6, dtype=torch.bool))
    assert out.shape == (2, 6, V, C_OUT)


# --- graph topology -----------------------------------------------------------


def test_graph_matches_pipeline():
    edges = build_edges()
    A = build_adjacency(edges)

    assert N_NODES == 75
    assert len(edges) == 82
    assert torch.equal(A, A.T)
    assert torch.all(A.diagonal() == 0), "build_adjacency must not add self-loops"


def test_graph_is_connected():
    A = build_adjacency()
    seen, frontier = {0}, [0]
    while frontier:
        frontier = [
            int(j)
            for i in frontier
            for j in torch.nonzero(A[i]).flatten()
            if int(j) not in seen and not seen.add(int(j))
        ]
    assert len(seen) == N_NODES


def test_pose_edges_match_mediapipe():
    """The hardcoded pose edges exist so building a model does not import
    MediaPipe; this checks them against the library when it is available."""
    mp = pytest.importorskip("mediapipe")
    from models.graph import POSE_EDGES, POSE_LANDMARK_IDS

    local = {mp_id: i for i, mp_id in enumerate(POSE_LANDMARK_IDS)}
    expected = {
        tuple(sorted((local[a], local[b])))
        for a, b in mp.solutions.pose.POSE_CONNECTIONS
        if a in local and b in local
    }
    assert {tuple(sorted(e)) for e in POSE_EDGES} == expected


def test_hand_edges_match_mediapipe():
    mp = pytest.importorskip("mediapipe")
    from models.graph import HAND_EDGES

    expected = {tuple(sorted(e)) for e in mp.solutions.hands.HAND_CONNECTIONS}
    assert {tuple(sorted(e)) for e in HAND_EDGES} == expected


# --- reference implementation --------------------------------------------------


def test_matches_pyg_gcnconv():
    """One layer against torch_geometric.nn.GCNConv on the real skeleton.

    PyG is a dev-only dependency: the model must not need it at runtime, but
    hand-written propagation deserves an independent check.
    """
    pyg_nn = pytest.importorskip("torch_geometric.nn")

    torch.manual_seed(0)
    A = build_adjacency()
    edge_index = torch.nonzero(A).t().contiguous()

    ours = GCN(C_IN, C_OUT, A=A, num_layers=1, edge_importance=False, dropout=0.0)
    ours.eval()
    theirs = pyg_nn.GCNConv(C_IN, C_OUT, add_self_loops=True, normalize=True, bias=True)
    theirs.eval()

    with torch.no_grad():
        # PyG stores [out, in]; ours is [in, out]
        theirs.lin.weight.copy_(ours.weight[0].t())
        theirs.bias.copy_(ours.bias[0])

    x = torch.randn(1, 1, V, C_IN)
    mask = torch.ones(1, 1, dtype=torch.bool)

    got = ours(x, mask)[0, 0]                                  # ReLU applied
    expected = torch.relu(theirs(x[0, 0], edge_index))

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_matches_hand_computed_toy_graph():
    """Dependency-free check of the same arithmetic on a 4-node path graph."""
    A = torch.zeros(4, 4)
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        A[i, j] = A[j, i] = 1.0

    cell = GCN(2, 3, A=A, num_layers=1, dropout=0.0)
    cell.eval()
    with torch.no_grad():
        cell.weight[0].copy_(torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]))
        cell.bias[0].fill_(0.1)

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])

    d = torch.tensor([2.0, 3.0, 3.0, 2.0]).pow(-0.5)
    A_hat = d.unsqueeze(1) * (A + torch.eye(4)) * d.unsqueeze(0)
    expected = torch.relu(A_hat @ (x @ cell.weight[0]) + 0.1)

    got = cell(x.view(1, 1, 4, 2), torch.ones(1, 1, dtype=torch.bool))[0, 0]

    torch.testing.assert_close(got, expected)
