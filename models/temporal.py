"""Temporal cells.

A temporal cell mixes information along the frame axis. It must preserve T:
the interleaved arrangement alternates spatial and temporal blocks, and a
spatial block preserves V on the assumption that its input still has the same
number of frames it started with. Nothing here downsamples.

Joint folding
-------------
Both cells operate on [N, V, T, C] and fold the joints into the batch
themselves, so `model.py` only ever permutes -- it never reshapes. The fold is
`[N, V, T, C] -> [N*V, T, C]`, which puts row `i*V + j` at clip `i`, joint `j`,
and the mask is expanded to match with `repeat_interleave` (see
`expand_mask_to_joints`).
"""

import math

import torch
import torch.nn as nn


def expand_mask_to_joints(mask: torch.Tensor, V: int) -> torch.Tensor:
    """[N, T] -> [N*V, T], aligned with `x.reshape(N*V, T, C)`.

    `repeat_interleave` gives [clip0]*V, [clip1]*V, ...; `repeat` would give
    [clip0, clip1, ...]*V, which silently pairs each sequence with the wrong
    clip's mask -- no shape error, just wrong numbers.
    """
    return mask.repeat_interleave(V, dim=0)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos position signal, registered as a buffer rather than a
    parameter: it is a constant, and making it learnable would add capacity to
    one arm of the comparison only."""

    def __init__(self, channels: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, channels)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, channels, 2, dtype=torch.float32)
            * (-math.log(10000.0) / channels)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] -> [B, T, C]"""
        T = x.size(1)
        if T > self.pe.size(0):
            raise ValueError(
                f"sequence of {T} frames exceeds positional encoding table "
                f"({self.pe.size(0)}); raise max_len"
            )
        return x + self.pe[:T].unsqueeze(0)


class TransformerTemporalCell(nn.Module):
    """Conv1d(k=3) -> TransformerEncoderLayer -> Conv1d(k=3), all length-preserving.

    The convs are C -> C and do not project; channel width is set by the spatial
    block upstream. Their job is local temporal mixing with weights shared across
    t -- the inductive bias that distinguishes this arm from the BiLSTM arm. The
    second conv occupies the structural slot Informer's distilling would have
    had, minus the stride-2 pooling.

    Positional encoding
    -------------------
    Optional, and the caller decides. The intended arrangement is PE once, on
    the first temporal cell only, which is what Vaswani 2017, BERT, ViT and
    Informer all do: absolute position is added before the stack, and the
    residual connections carry that signal forward, so re-injecting it at every
    depth is not required.

    Re-adding the same absolute vector at every layer is non-standard. The
    approaches that do supply position at every layer -- relative position
    representations (Shaw et al. 2018) and rotary embeddings (Su et al. 2021) --
    change the attention computation itself rather than re-adding an absolute
    offset to the values.

    There is a further reason to inject once here specifically: a spatial block
    sits between consecutive temporal cells and recombines features across
    joints. PE added after that block lands on spatially mixed features, not on
    raw per-joint sequences, so the second injection is not even the same
    operation as the first. It would also be an effect present on only one arm
    of a four-way comparison.

    Masking
    -------
    Convolution over time smears padding into real positions: at the last real
    frame a k=3 kernel reads t+1, which is padding, while the mask still marks t
    valid. The corruption creeps one frame inward per conv and is invisible.
    Everything that can write a non-zero value into a padded position is
    therefore followed by a re-zero: the positional encoding (which is non-zero
    everywhere), both convs, and attention (whose padded *queries* still produce
    output even though padded *keys* are masked out of real queries' attention).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        kernel_size: int = 3,
        dim_feedforward: int | None = None,
        add_pos_encoding: bool = True,
        max_len: int = 5000,
    ):
        """`max_len` sizes the positional table and is ignored when
        `add_pos_encoding` is False. Passing both is not an error."""
        super().__init__()
        if out_channels != in_channels:
            raise ValueError(
                f"TransformerTemporalCell does not project: got in_channels="
                f"{in_channels}, out_channels={out_channels}. Channel width is "
                f"set by the spatial block upstream; the convs here are C -> C."
            )
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd to preserve T, got {kernel_size}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_pos_encoding = add_pos_encoding
        padding = (kernel_size - 1) // 2

        # Not constructed when unused: an idle submodule still registers its
        # buffer and shows up in per-configuration parameter and buffer counts.
        self.pos_encoding = (
            SinusoidalPositionalEncoding(in_channels, max_len=max_len)
            if add_pos_encoding
            else None
        )
        self.num_layers = num_layers

        # The convs bracket the whole attention stack, once per cell -- not once
        # per internal layer. The receptive field of the conv path is therefore
        # 4N+1 frames over N blocks, independent of num_layers.
        self.conv_pre = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=n_heads,
            dim_feedforward=dim_feedforward or 4 * in_channels,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # post-norm, as in Vaswani 2017
        )
        # nn.TransformerEncoder injects no positional signal of its own, which is
        # what we want: PE is applied once at the start of the cell, never
        # between internal layers.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # keep one numeric path regardless of mask
        )
        self.conv_post = nn.Conv1d(in_channels, in_channels, kernel_size, padding=padding)

    @staticmethod
    def _apply_conv(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C] -> [B, T, C]. Conv1d wants channels first."""
        return conv(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [N, V, T, C_in]
        mask: [N, T] bool, True = real frame
        returns: [N, V, T, C_out]
        """
        if x.dim() != 4:
            raise ValueError(f"expected [N, V, T, C], got {tuple(x.shape)}")
        N, V, T, C = x.shape
        if C != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {C}")
        if mask.shape != (N, T):
            raise ValueError(f"expected mask [{N}, {T}], got {tuple(mask.shape)}")
        # A clip with no real frames masks out every key, leaving attention to
        # softmax over an empty set. What that produces is version- and
        # path-dependent: raw nn.MultiheadAttention returns NaN, while
        # nn.TransformerEncoderLayer currently sanitises the row (checked on
        # torch 2.14). Downstream, the readout would hit -inf on every channel.
        #
        # So the symptom is not stable enough to rely on, which is the argument
        # for checking the invariant directly. No correct collate_fn can produce
        # such a clip; this is here so that a later change to padding or batching
        # fails on its first bad batch instead of degrading quietly.
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every clip must have at least one real frame")

        h = x.reshape(N * V, T, C)
        joint_mask = expand_mask_to_joints(mask, V)      # [N*V, T]
        keep = joint_mask.unsqueeze(-1)                  # [N*V, T, 1]

        h = h.masked_fill(~keep, 0.0)                    # trust nothing upstream
        if self.pos_encoding is not None:
            h = self.pos_encoding(h)
            h = h.masked_fill(~keep, 0.0)                # PE is non-zero on padding

        h = self._apply_conv(self.conv_pre, h)
        h = h.masked_fill(~keep, 0.0)

        h = self.encoder(h, src_key_padding_mask=~joint_mask)
        h = h.masked_fill(~keep, 0.0)                    # before conv_post reads it

        h = self._apply_conv(self.conv_post, h)
        h = h.masked_fill(~keep, 0.0)

        return h.reshape(N, V, T, self.out_channels)


class BiLSTMTemporalCell(nn.Module):
    """Bidirectional LSTM over the frame axis. Not implemented yet.

    Notes for when it is: packing is mandatory, not an optimisation -- the
    backward pass would otherwise start inside the padding and read it before
    reaching any real frame. Output width is 2 * hidden when concatenating, so
    `out_channels` is a real constraint here, unlike in the Transformer cell.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
