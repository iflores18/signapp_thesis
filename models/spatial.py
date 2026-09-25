"""Spatial cells.

A spatial cell mixes information across joints within a frame. It must preserve
V, and it is the only place in the model where channel width changes -- the
temporal cells are C -> C. The graph adjacency is fixed, not learned; only the
weight matrix W is.

Both cells operate on [N, T, V, C] and fold frames into the batch themselves,
mirroring how the temporal cells fold joints.

Depth
-----
`num_layers` is depth *inside* one cell, distinct from the model's `n_blocks`.
Layers stack as repeated propagation: each one extends the receptive field by a
single hop of the graph.

Keep it small. Over-smoothing is the practical ceiling -- repeated neighbourhood
averaging drives node features toward each other until they stop being
distinguishable. GCN is the more susceptible of the two: its weights come from
the fixed normalised adjacency, so a node cannot do anything but average its
neighbours uniformly, whereas GAT's learned coefficients can concentrate on a
few neighbours and resist the collapse.
"""

import torch
import torch.nn as nn


class GCN(nn.Module):
    """Graph convolution against a fixed normalised adjacency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [N, T, V, C_in], mask: [N, T] -> [N, T, V, C_out]"""
        raise NotImplementedError


class GAT(nn.Module):
    """Graph attention over the same fixed topology, used as a mask rather than
    as weights.

    Head handling across internal layers follows Velickovic et al.: concat=True
    on internal layers, so heads stay separable while depth accumulates, and
    concat=False (average) on the final layer, so the cell's external output
    width matches what GCN produces. The `out_channels` contract is therefore
    unaffected by `num_layers` -- callers see the same width either way.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        n_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [N, T, V, C_in], mask: [N, T] -> [N, T, V, C_out]"""
        raise NotImplementedError
