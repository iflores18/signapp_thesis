"""The interleaved model: (Spatial -> Temporal) x N, readout, MLP head.

Spatial and temporal cells are passed in as classes, not hard-wired, so the four
configurations in the comparison differ only by two arguments. Wiring them as
four separate model classes would let the arms drift apart and turn the
comparison into a confound.

Three depth parameters, kept separate on purpose:

    n_blocks (N)  how many times spatial and temporal mixing alternate
    L_spatial     graph layers inside one spatial cell
    L_temporal    encoder / recurrent layers inside one temporal cell

N=3 with L_temporal=1 and N=1 with L_temporal=3 contain the same number of
attention layers and are different architectures -- the first interleaves, the
second collapses toward a pipeline. The arrangement is what the thesis argues
about, so these never derive from one another.
"""

import torch
import torch.nn as nn


class STModel(nn.Module):
    def __init__(
        self,
        spatial_cls,
        temporal_cls,
        n_blocks: int,
        C_in: int,
        hidden: int,
        n_classes: int,
        A: torch.Tensor,
        L_spatial: int = 1,
        L_temporal: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.L_spatial = L_spatial
        self.L_temporal = L_temporal

        self.spatial = nn.ModuleList()
        self.temporal = nn.ModuleList()
        self.seam_norm = nn.ModuleList()
        self.seam_proj = nn.ModuleList()

        for block in range(n_blocks):
            c_in = C_in if block == 0 else hidden

            # Freshly constructed per block. Reusing one instance would tie the
            # weights across depth, which is a different architecture.
            self.spatial.append(
                spatial_cls(
                    c_in, hidden, A, num_layers=L_spatial, n_heads=n_heads, dropout=dropout
                )
                if _accepts(spatial_cls, "n_heads")
                else spatial_cls(c_in, hidden, A, num_layers=L_spatial, dropout=dropout)
            )
            self.temporal.append(
                temporal_cls(
                    hidden,
                    hidden,
                    num_layers=L_temporal,
                    dropout=dropout,
                    # Absolute PE belongs before the stack, not at every depth.
                    # Residuals carry the signal forward; re-adding it after a
                    # spatial block would inject it into joint-mixed features.
                    **({"add_pos_encoding": block == 0, "n_heads": n_heads}
                       if _accepts(temporal_cls, "add_pos_encoding") else {}),
                )
            )
            # Seam LayerNorm over C only. Normalising over (V, C) jointly would
            # mix joints and partly undo what the spatial block just did.
            self.seam_norm.append(nn.LayerNorm(hidden))
            # Project only where the widths actually differ. What has to be
            # identical across the four configurations is the *rule*, not the
            # resulting module: a projection present in one arm and absent in
            # another at the same seam would confound the comparison, but an
            # identity-shaped Linear at a seam where every configuration already
            # matches prevents no confound and costs parameters for nothing.
            # Fires at block 0 (C_in -> hidden), and at any temporal cell whose
            # output width differs from its input.
            c_out = hidden
            self.seam_proj.append(
                nn.Linear(c_in, c_out) if c_in != c_out else nn.Identity()
            )

        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [N, T, V, C_in], mask: [N, T] bool -> logits [N, n_classes]"""
        keep = mask[:, :, None, None]

        for spatial, temporal, norm, proj in zip(
            self.spatial, self.temporal, self.seam_norm, self.seam_proj
        ):
            h = spatial(x, mask)                       # [N, T, V, hidden]
            h = temporal(h.permute(0, 2, 1, 3), mask)  # [N, V, T, hidden]
            h = h.permute(0, 2, 1, 3)                  # back to [N, T, V, hidden]

            x = norm(proj(x) + h)                      # post-norm: add, then norm
            x = x.masked_fill(~keep, 0.0)

        return self.head(masked_max_pool(x, mask))


def masked_max_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over joints, then masked max over frames.

    x: [N, T, V, C], mask: [N, T] bool, True = real frame -> [N, C]

    V is reduced first so that the max stays the final, selective operation.
    The other order -- max over T, then mean over V -- collapses time into one
    summary per joint and then averages all V of them together, diluting the few
    joints that discriminate a given sign against the ~74 that do not.

    Joints the detector never found are stored as exact zeros, but that is a
    data-representation problem, addressed by a visibility channel alongside the
    coordinates. It is not something a choice of pooling order can fix: a mean
    over zero-valued joints is contaminated exactly as much as a max would be.

    Only the T reduction takes the mask. V carries no padding -- every clip has
    all V joints in every frame, detected or not -- so the mean over V is
    unmasked. The max must be masked: a plain max would include padded positions
    and floor any feature whose real activations are all negative.
    """
    if x.dim() != 4:
        raise ValueError(f"expected [N, T, V, C], got {tuple(x.shape)}")
    if mask.shape != x.shape[:2]:
        raise ValueError(f"expected mask {tuple(x.shape[:2])}, got {tuple(mask.shape)}")
    # A fully padded row would come out as -inf on every channel and propagate
    # silently. That is a collate_fn bug, and it should fail here, loudly.
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every clip must have at least one real frame")

    x = x.mean(dim=2)                                        # [N, T, C]
    x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    return x.max(dim=1).values                               # [N, C]


def _accepts(cls, name: str) -> bool:
    """Whether `cls.__init__` takes a keyword called `name`.

    Lets STModel pass n_heads to GAT but not GCN, and add_pos_encoding to the
    Transformer cell but not the BiLSTM, without branching on class identity.
    """
    import inspect

    try:
        return name in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False
