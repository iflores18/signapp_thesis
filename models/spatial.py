"""Spatial cells.

A spatial cell mixes information across joints within a frame. It preserves V
and is the only place in the model where channel width changes -- the temporal
cells are C -> C. Graph topology is fixed (see `models/graph.py`); only the
weight matrices, and optionally the edge importance, are learned.

GCN and GAT share `_GraphCell`, which owns everything that is not aggregation:
shape handling, masking, the boundary guard, the adjacency buffers and the
edge-importance parameters. Keeping the shared parts in one place is what makes
the two arms differ *only* in how a node combines its neighbours, which is the
quantity the comparison is about.

Depth
-----
`num_layers` is depth inside one cell, distinct from the model's `n_blocks`.
Each layer extends the receptive field by one hop of the graph.

Keep it small. Over-smoothing is the practical ceiling -- repeated
neighbourhood averaging drives node features toward each other until they stop
being distinguishable. GCN is the more susceptible of the two: its coefficients
come from the fixed normalised adjacency, so a node can only average its
neighbours uniformly, whereas GAT's learned coefficients can concentrate.

The dense form is deliberate. `torch.matmul(A_hat, h)` broadcasts [V, V]
against [N*T, V, C] in one call, with no per-sample replication and no
edge_index bookkeeping, and hand-writing both cells from this base keeps them
from being separated by a library's implementation choices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import build_adjacency, normalize_adjacency


class _GraphCell(nn.Module):
    """Shared scaffolding for the graph cells. Subclasses implement `_layer`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor | None = None,
        num_layers: int = 1,
        edge_importance: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        if A is None:
            A = build_adjacency()
        if A.dim() != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"expected a square adjacency, got {tuple(A.shape)}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_edge_importance = edge_importance
        self.V = A.shape[0]

        # Constants derived from topology: never parameters, and never
        # recomputed per forward pass. persistent=False keeps them out of
        # checkpoints, since they are rebuilt from graph.py rather than restored.
        self.register_buffer("A_hat", normalize_adjacency(A), persistent=False)
        # Binary support of A~ (self-loops included), used to confine learned
        # edge importance to edges that exist.
        self.register_buffer(
            "support", (A + torch.eye(self.V)).gt(0).to(torch.float32), persistent=False
        )

        if edge_importance:
            # One per layer, not shared: each hop may want a different emphasis.
            # Initialised to ones, so switching the flag on starts from exactly
            # the plain function.
            self.edge_importance = nn.ParameterList(
                nn.Parameter(torch.ones(self.V, self.V)) for _ in range(num_layers)
            )
        else:
            # Not constructed at all. An unused parameter would still show up in
            # state_dict and in the per-configuration counts for the setup table.
            self.edge_importance = None

        self.dropout = nn.Dropout(dropout)

    def _adjacency(self, layer: int) -> torch.Tensor:
        """A_hat for this layer, optionally reweighted by learned importance."""
        if self.edge_importance is None:
            return self.A_hat
        # `support` confines M to real edges. A_hat is already zero off the
        # support, so for this ordering the mask is redundant to the product --
        # it is here so the invariant survives a later change to the ordering,
        # and so the constraint is visible in the code rather than implied.
        return self.A_hat * (self.edge_importance[layer] * self.support)

    def _layer(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [N, T, V, C_in]
        mask: [N, T] bool, True = real frame
        returns: [N, T, V, C_out]
        """
        if x.dim() != 4:
            raise ValueError(f"expected [N, T, V, C], got {tuple(x.shape)}")
        N, T, V, C = x.shape
        if V != self.V:
            raise ValueError(f"expected {self.V} joints, got {V}")
        if C != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {C}")
        if mask.shape != (N, T):
            raise ValueError(f"expected mask [{N}, {T}], got {tuple(mask.shape)}")
        # What a malformed batch looks like downstream varies by module and by
        # PyTorch version, so the invariant is asserted here rather than left to
        # surface as some other module's error.
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every clip must have at least one real frame")

        h = x.reshape(N * T, V, C)
        for layer in range(self.num_layers):
            h = self._layer(h, layer)
        out = h.reshape(N, T, V, self.out_channels)

        # Padded frames pass straight through this cell, and with a bias term a
        # frame of zeros does not stay zero: the bias is added, ReLU keeps
        # whatever is positive, and the mask still marks those positions
        # invalid. The same silent failure as positional encoding in the
        # temporal cell.
        return out.masked_fill(~mask[:, :, None, None], 0.0)


class GCN(_GraphCell):
    """Graph convolution against a fixed normalised adjacency.

    One layer, following Kipf & Welling Eq. 8 plus the reference
    implementation's bias:

        H_out = ReLU( A_hat @ (dropout(H) @ W) + b )

    Dropout is on the layer *input*, as in the paper. W uses Glorot
    initialisation, as in the reference implementation.

    Activation
    ----------
    ReLU after every layer, the final one included, which makes this cell's
    contribution to the seam residual non-negative while the temporal cell's
    output is linear. That is a real asymmetry between the arms and belongs in
    the setup section: it is a choice (Kipf Eq. 14, the residual variant), not
    an oversight.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor | None = None,
        num_layers: int = 1,
        edge_importance: bool = False,
        bias: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels,
            out_channels,
            A,
            num_layers=num_layers,
            edge_importance=edge_importance,
            dropout=dropout,
        )
        widths = [
            (in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_layers)
        ]
        self.weight = nn.ParameterList(
            nn.Parameter(torch.empty(c_in, c_out)) for c_in, c_out in widths
        )
        self.bias = (
            nn.ParameterList(
                nn.Parameter(torch.zeros(out_channels)) for _ in range(num_layers)
            )
            if bias
            else None
        )
        for W in self.weight:
            nn.init.xavier_uniform_(W)

    def _layer(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        h = self.dropout(h)                              # Kipf: on the input
        h = h @ self.weight[layer]                       # [.., V, C_in] @ [C_in, C_out]
        h = torch.matmul(self._adjacency(layer), h)      # [V, V] broadcasts over the batch
        if self.bias is not None:
            h = h + self.bias[layer]
        return F.relu(h)


class GAT(_GraphCell):
    """Graph attention over the same fixed topology, following Velickovic et al.
    (2018).

    This is not a controlled attention-versus-aggregation ablation. GCN and GAT
    are compared as two architecturally distinct spatial operators, each in its
    canonical configuration, so this cell follows the paper where it diverges
    from the GCN cell. The one control retained is output width.

        aspect          GCN                   GAT
        activation      ReLU                  ELU
        bias            on                    off (Eq. 1-6 have no bias term)
        normalisation   D~^-1/2 A~ D~^-1/2    softmax over the neighbourhood
        heads           --                    n_heads, averaged
        dropout         input only            input and attention coefficients

    `edge_importance` is deliberately absent: it is GCN's way of learning a
    per-edge weight, and alpha already does that here, per frame.

    No A_hat
    --------
    GAT has no degree normalisation -- the softmax does the normalising -- so
    this cell uses only the boolean support of A~, which the base already
    derives from the same adjacency the GCN path normalises. There is no second,
    independently built graph. The base's `A_hat` buffer goes unused here; it is
    a buffer rather than a parameter, so it costs 22 KB and no parameters.

    Heads
    -----
    All n_heads are computed and all contribute: the output is their mean, which
    keeps GAT's width equal to GCN's so channel dimensionality is not a
    confound. Velickovic et al. concatenate at intermediate depth and average
    only at the output layer, so averaging at every layer is a deliberate
    deviation made for width-matching, not an approximation of the paper.

    Masking
    -------
    The base re-zeroes padded frames after the cell. Here that is defensive
    rather than load-bearing: with no bias, a padded frame of zeros stays zero
    on its own -- Wh = 0, so e = 0, the softmax is uniform over the
    neighbourhood, alpha @ 0 = 0, and ELU(0) = 0. Contrast GCN, where the bias
    genuinely writes non-zero values into padded positions.

    Masking attention to the neighbourhood, by contrast, is load-bearing:
    without it every node attends over all V, which discards the skeleton
    entirely and degenerates toward uniform averaging on a complete graph.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor | None = None,
        num_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__(
            in_channels,
            out_channels,
            A,
            num_layers=num_layers,
            edge_importance=False,  # alpha already weights edges, per frame
            dropout=dropout,
        )
        self.n_heads = n_heads

        widths = [
            (in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_layers)
        ]
        # One W per head per layer, shared across nodes. Heads are a batched
        # axis rather than a Python loop, so a layer is three matmuls whatever
        # n_heads is.
        self.weight = nn.ParameterList(
            nn.Parameter(torch.empty(n_heads, c_in, c_out)) for c_in, c_out in widths
        )
        # `a` is a vector in R^(2*C_out) per head, split into the two halves that
        # multiply the centre node and the neighbour. Keeping it split is what
        # avoids materialising [V, V, 2*C_out] (see `_layer`).
        self.att_center = nn.ParameterList(
            nn.Parameter(torch.empty(n_heads, c_out, 1)) for _, c_out in widths
        )
        self.att_neighbor = nn.ParameterList(
            nn.Parameter(torch.empty(n_heads, c_out, 1)) for _, c_out in widths
        )
        for p in list(self.weight) + list(self.att_center) + list(self.att_neighbor):
            nn.init.xavier_uniform_(p)

        # GAT applies dropout to the attention coefficients as well as the
        # input (section 3.3). GCN has no analogue.
        self.attn_dropout = nn.Dropout(dropout)

        # Self-loops guarantee at least one unmasked entry per row, so the
        # softmax can never see an all--inf row. Asserted once here rather than
        # relied on silently.
        if not bool(self.support.diagonal().gt(0).all()):
            raise ValueError(
                "the adjacency support must include self-loops: GAT's "
                "neighbourhood N(i) contains i, and without it a node with no "
                "neighbours would softmax over an all--inf row"
            )

    def attention(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Attention coefficients for one layer: [.., V, C_in] -> [.., K, V, V].

        Exposed so the tests can assert the structural properties of alpha --
        rows summing to one over the neighbourhood, exact zeros off the edges,
        non-zero self-attention -- rather than inferring them from the output.
        """
        Wh = self._transform(h, layer)
        return self._coefficients(Wh, layer)

    def _transform(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Shared linear transform, per head: [.., V, C_in] -> [.., K, V, C_out]."""
        h = self.dropout(h)                      # on the layer input, as in GCN
        return h.unsqueeze(-3) @ self.weight[layer]

    def _coefficients(self, Wh: torch.Tensor, layer: int) -> torch.Tensor:
        """Masked, normalised attention: [.., K, V, C_out] -> [.., K, V, V]."""
        # a^T [Wh_i || Wh_j] splits into two terms, so two [V, 1] vectors
        # broadcast into [V, V] and the [V, V, 2*C_out] concatenation is never
        # built. e[i, j] weights neighbour j from the perspective of centre i.
        e_center = Wh @ self.att_center[layer]                    # [.., K, V, 1]
        e_neighbor = Wh @ self.att_neighbor[layer]                # [.., K, V, 1]
        e = F.leaky_relu(
            e_center + e_neighbor.transpose(-2, -1), negative_slope=0.2
        )                                                         # [.., K, V, V]

        e = e.masked_fill(~self.support.bool(), float("-inf"))
        return e.softmax(dim=-1)

    def _layer(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        Wh = self._transform(h, layer)                   # [.., K, V, C_out]
        alpha = self.attn_dropout(self._coefficients(Wh, layer))
        out = alpha @ Wh                                 # [.., K, V, C_out]
        return F.elu(out.mean(dim=-3))                   # average heads, then ELU
