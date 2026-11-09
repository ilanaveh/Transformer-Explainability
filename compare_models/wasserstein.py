"""Validated spatial Wasserstein-1 utilities, independent of model generation."""
from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F


def normalize_map(x, eps=1e-12, clip_negative=True):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty, finite 2D map.")
    if clip_negative:
        x = np.maximum(x, 0)
    elif np.any(x < 0):
        raise ValueError("Negative relevance cannot be probability mass.")
    total = x.sum()
    if not np.isfinite(total) or total <= eps:
        raise ValueError("Map has no positive relevance mass.")
    return x / total


def load_attention_map(path):
    x = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError(f"Expected a nonempty, finite 2D map: {path}")
    return x


def resize_map(x, output_size):
    if len(output_size) != 2 or any(int(n) != n or n < 1 for n in output_size):
        raise ValueError("Map size must contain two positive integers.")
    if tuple(x.shape) == tuple(output_size):
        return np.asarray(x, dtype=np.float64).copy()
    return F.interpolate(torch.as_tensor(x, dtype=torch.float64)[None, None],
                         size=tuple(output_size), mode="bilinear",
                         align_corners=False)[0, 0].cpu().numpy()


def prepare_map(x, output_size=None, clip_negative=True):
    x = normalize_map(x, clip_negative=clip_negative)
    if output_size is not None:
        x = resize_map(x, output_size)
    return normalize_map(x, clip_negative=False)


@lru_cache(maxsize=8)
def make_2d_cost_matrix(height, width, normalize_cost=True):
    """Euclidean costs divided by the grid diagonal by default (range 0..1)."""
    if any(int(n) != n or n < 1 for n in (height, width)):
        raise ValueError("Grid dimensions must be positive integers.")
    coords = np.indices((height, width)).reshape(2, -1).T.astype(float)
    cost = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    if normalize_cost and cost.max() > 0:
        cost /= cost.max()
    cost.setflags(write=False)
    return cost


def wasserstein_distance_2d(map_a, map_b, cost_matrix=None, normalize_cost=True):
    """Exact W1 of same-shaped maps; normalization removes total magnitude."""
    import ot
    a = normalize_map(map_a)
    b = normalize_map(map_b)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if cost_matrix is None:
        cost_matrix = make_2d_cost_matrix(*a.shape, normalize_cost)
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.shape != (a.size, a.size) or not np.isfinite(cost).all() or np.any(cost < 0):
        raise ValueError("Invalid transport cost matrix.")
    return float(ot.emd2(a.ravel(), b.ravel(), cost))
