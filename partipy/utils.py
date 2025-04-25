import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist


def align_archetypes(ref_arch: np.ndarray, query_arch: np.ndarray) -> np.ndarray:
    """Align the query archetypes to the reference archetypes"""
    assert np.all(ref_arch.shape == query_arch.shape)
    euclidean_d = cdist(ref_arch, query_arch)
    ref_idx, query_idx = linear_sum_assignment(euclidean_d)
    return query_arch[query_idx, :]


def compute_rowwise_l2_distance(mtx_1: np.ndarray, mtx_2: np.ndarray) -> np.ndarray:
    """Compute l2 distance between the rows of mtx 1 and the rows of mtx 2"""
    assert np.all(mtx_1.shape == mtx_2.shape)
    dist = np.sqrt(np.sum(np.square(mtx_1 - mtx_2), axis=1))
    return dist


def compute_rowwise_l1_distance(mtx_1: np.ndarray, mtx_2: np.ndarray) -> np.ndarray:
    """Compute l1 distance between the rows of mtx 1 and the rows of mtx 2"""
    assert np.all(mtx_1.shape == mtx_2.shape)
    dist = np.sum(np.abs(mtx_1 - mtx_2), axis=1)
    return dist


def compute_relative_rowwise_l2_distance(mtx_1: np.ndarray, mtx_2: np.ndarray) -> np.ndarray:
    """Compute relative l2 distance between the rows of mtx 1 and the rows of mtx 2"""
    rowwise_l2 = compute_rowwise_l2_distance(mtx_1, mtx_2)
    archetype_dispersion = np.mean(pdist(mtx_1))  # average pairwise distance
    rowwise_l2_normalized = rowwise_l2 / archetype_dispersion
    return rowwise_l2_normalized
