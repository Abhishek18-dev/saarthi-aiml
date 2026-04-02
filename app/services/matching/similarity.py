"""Similarity computation between user embedding vectors.

Uses ``sklearn.metrics.pairwise.cosine_similarity`` which is
highly optimised for small-to-medium batch sizes on CPU.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return the full cosine-similarity matrix for all embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(N, D)`` where *N* is the number of users and *D* the
        embedding dimensionality.

    Returns
    -------
    np.ndarray
        Shape ``(N, N)`` symmetric matrix with 1.0 on the diagonal.
    """
    return cosine_similarity(embeddings)


def compute_pairwise(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return the cosine similarity between two 1-D vectors.

    Both vectors are reshaped automatically so callers don't have to.
    """
    a = vec_a.reshape(1, -1)
    b = vec_b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])


def get_top_similar(
    sim_matrix: np.ndarray,
    user_index: int,
    top_k: int = 5,
    exclude_self: bool = True,
) -> list[tuple[int, float]]:
    """Return the indices and scores of the *top_k* most similar users.

    Parameters
    ----------
    sim_matrix : np.ndarray
        Pre-computed ``(N, N)`` similarity matrix.
    user_index : int
        Row index for the target user.
    top_k : int
        Number of matches to return.
    exclude_self : bool
        Whether to exclude the user's own row (score ≈ 1.0).

    Returns
    -------
    list[tuple[int, float]]
        List of ``(index, score)`` tuples sorted descending by score.
    """
    scores = sim_matrix[user_index].copy()

    if exclude_self:
        scores[user_index] = -1.0  # ensure self is never included

    # Clamp top_k to not exceed available users.
    top_k = min(top_k, len(scores) - (1 if exclude_self else 0))

    # argpartition is O(N) vs argsort's O(N log N) — faster for large N.
    if top_k >= len(scores):
        ranked = np.argsort(-scores)[:top_k]
    else:
        partitioned = np.argpartition(-scores, top_k)[:top_k]
        ranked = partitioned[np.argsort(-scores[partitioned])]

    return [(int(idx), round(float(scores[idx]), 4)) for idx in ranked]
