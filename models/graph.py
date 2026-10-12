"""Skeleton graph topology.

One graph, fixed for every frame of every clip. It lives here rather than
inside a cell because GCN and GAT must consume exactly the same edge set -- if
the two arms disagreed about the topology, the comparison would measure that
instead of the aggregation rule -- and because the edge set itself is a planned
ablation (bone-only against bone plus semantic shortcuts).

Node layout, matching the 237-value feature vector produced by the data
pipeline:

    pose        0 - 11   MediaPipe pose landmarks 11-22
    face       12 - 32   21 of the 468 face-mesh points
    left hand  33 - 53   MediaPipe hand landmarks 0-20
    right hand 54 - 74   MediaPipe hand landmarks 0-20

Every edge below is anatomical. No shortcut edges are added to shorten paths
between parts that are not physically attached; that is the other arm of the
topology ablation, and it belongs in a separate edge set rather than mixed in
here.
"""

from typing import List, Tuple

import torch

# --- landmark sets, mirroring section 2 of data_pipeline.ipynb ---------------

POSE_LANDMARK_IDS = list(range(11, 23))  # shoulders, elbows, wrists, hand stubs

FACE_LANDMARK_IDS = [
    61, 291, 0, 17,             # mouth: corners, upper lip, lower lip
    33, 263, 133, 362,          # eyes: outer corners, inner corners
    159, 145, 386, 374,         # eyelids: right upper/lower, left upper/lower
    105, 334, 107, 336,         # eyebrows: mid-brow, inner ends
    1, 6, 10, 9, 152,           # nose tip, nose bridge, top of face, forehead, chin
]

N_POSE = len(POSE_LANDMARK_IDS)   # 12
N_FACE = len(FACE_LANDMARK_IDS)   # 21
N_HAND = 21
N_NODES = N_POSE + N_FACE + 2 * N_HAND  # 75

POSE_OFFSET = 0
FACE_OFFSET = N_POSE
LEFT_HAND_OFFSET = FACE_OFFSET + N_FACE
RIGHT_HAND_OFFSET = LEFT_HAND_OFFSET + N_HAND

# --- edge groups -------------------------------------------------------------

# mediapipe.solutions.pose.POSE_CONNECTIONS restricted to landmarks 11-22 and
# re-indexed to 0-11. Hardcoded rather than imported so that building a model
# does not pull in MediaPipe and its TF Lite runtime; test_graph.py checks this
# list against the library when it is importable.
POSE_EDGES: List[Tuple[int, int]] = [
    (0, 1),           # shoulder - shoulder
    (0, 2), (2, 4),   # left: shoulder - elbow - wrist
    (1, 3), (3, 5),   # right: shoulder - elbow - wrist
    (4, 6), (4, 8), (4, 10), (6, 8),    # left wrist - pinky / index / thumb
    (5, 7), (5, 9), (5, 11), (7, 9),    # right wrist - pinky / index / thumb
]

# mediapipe.solutions.hands.HAND_CONNECTIONS, identical for both hands.
HAND_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm closure
]

# The face mesh is not a bone structure and only 21 of its points are kept, so
# these are defined by hand in MediaPipe face-mesh ids.
FACE_EDGES_BY_ID: List[Tuple[int, int]] = [
    (152, 17), (17, 61), (17, 291), (61, 0), (291, 0),   # chin - lips - corners
    (0, 1), (1, 6),                                      # upper lip - nose tip - bridge
    (6, 133), (6, 362),                                  # bridge - inner eye corners
    (133, 159), (133, 145), (159, 33), (145, 33),        # right eye ring
    (362, 386), (362, 374), (386, 263), (374, 263),      # left eye ring
    (6, 9), (9, 10),                                     # bridge - forehead - crown
    (9, 105), (9, 334), (105, 107), (334, 336),          # forehead - brows - inner ends
]

# Bridges between the four parts. Without them the skeleton is four
# disconnected components and no information reaches the hands from the face.
FACE_TO_SHOULDERS = 152          # the chin, lowest face point, nearest the shoulders
POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12
POSE_L_WRIST, POSE_R_WRIST = 15, 16
HAND_WRIST = 0


def build_edges() -> List[Tuple[int, int]]:
    """Undirected edges as node-index pairs. No self-loops -- GCN adds those."""
    pose_local = {mp_id: i for i, mp_id in enumerate(POSE_LANDMARK_IDS)}
    face_local = {mp_id: i for i, mp_id in enumerate(FACE_LANDMARK_IDS)}

    edges: List[Tuple[int, int]] = []
    edges += [(POSE_OFFSET + a, POSE_OFFSET + b) for a, b in POSE_EDGES]
    edges += [
        (FACE_OFFSET + face_local[a], FACE_OFFSET + face_local[b])
        for a, b in FACE_EDGES_BY_ID
    ]
    for offset in (LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET):
        edges += [(offset + a, offset + b) for a, b in HAND_EDGES]

    chin = FACE_OFFSET + face_local[FACE_TO_SHOULDERS]
    edges += [
        (chin, POSE_OFFSET + pose_local[POSE_L_SHOULDER]),
        (chin, POSE_OFFSET + pose_local[POSE_R_SHOULDER]),
        (LEFT_HAND_OFFSET + HAND_WRIST, POSE_OFFSET + pose_local[POSE_L_WRIST]),
        (RIGHT_HAND_OFFSET + HAND_WRIST, POSE_OFFSET + pose_local[POSE_R_WRIST]),
    ]
    return edges


def build_adjacency(edges: List[Tuple[int, int]] | None = None, n_nodes: int = N_NODES) -> torch.Tensor:
    """Binary, symmetric [V, V] adjacency with a zero diagonal.

    Self-loops are deliberately absent: GCN forms A~ = A + I itself, and adding
    them here would double the diagonal.
    """
    if edges is None:
        edges = build_edges()
    A = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return A


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """A_hat = D~^-1/2 (A + I) D~^-1/2, Kipf & Welling's propagation rule.

    Self-loops are added *before* the degrees are computed. Normalising first
    and adding I afterwards gives a different matrix -- the diagonal ends up
    unscaled -- and nothing about the result looks wrong, so the ordering is
    asserted in the tests rather than trusted.
    """
    A_tilde = A + torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    d_inv_sqrt = A_tilde.sum(dim=1).pow(-0.5)
    return d_inv_sqrt.unsqueeze(1) * A_tilde * d_inv_sqrt.unsqueeze(0)
