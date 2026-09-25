"""Run configuration.

Holds every value that is a choice rather than a consequence. No model code
lives here, and no module reads this file -- `main.py` constructs objects from
it and passes what they need explicitly, so the modules stay importable and
testable on their own.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # data / optimisation
    batch_size: int = 16
    lr: float = 1e-3
    epochs: int = 100

    # architecture
    #
    # Three depth parameters, deliberately independent. n_blocks is the
    # arrangement parameter -- how many times spatial and temporal mixing
    # alternate -- while the two L_* are depth *inside* a cell. N=3 with
    # L_temporal=1 and N=1 with L_temporal=3 have the same number of attention
    # layers and are different architectures: the first interleaves, the second
    # collapses toward a pipeline. Never derive one from another.
    n_blocks: int = 2        # N: (Spatial -> Temporal) x N
    L_spatial: int = 1       # graph layers inside one spatial cell (1 hop each)
    L_temporal: int = 1      # encoder / recurrent layers inside one temporal cell
    C_in: int = 3            # x, y, z per joint
    V: int = 75              # joints in the skeleton graph
    hidden: int = 64         # channel width after the first spatial block
    n_heads: int = 4
    dropout: float = 0.1

    n_classes: int = 10
