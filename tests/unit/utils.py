import numpy as np
from partipy.const import OPTIM_ALGS
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist

# for regularized_nnls the tests take much longer, and this algorithm is not recommended
FAST_OPTIM_ALGS = tuple(alg for alg in OPTIM_ALGS if alg != "regularized_nnls")


def align_archetypes(ref_arch, query_arch):
    # not sure if copy here is needed, compute_dist_mtx should not modify the matrices
    euclidean_d = cdist(ref_arch, query_arch.copy())
    ref_idx, query_idx = linear_sum_assignment(euclidean_d)
    return query_arch[query_idx, :]


def compute_rowwise_l2_distance(mtx_1, mtx_2):
    assert np.all(mtx_1.shape == mtx_2.shape)
    dist = np.sqrt(np.sum(np.square(mtx_1 - mtx_2), axis=1))
    return dist


def compute_rowwise_l1_distance(mtx_1, mtx_2):
    assert np.all(mtx_1.shape == mtx_2.shape)
    dist = np.sum(np.abs(mtx_1 - mtx_2), axis=1)
    return dist


def compute_rowwise_correlation(mtx_1, mtx_2):
    assert np.all(mtx_1.shape == mtx_2.shape)
    mtx_1 = mtx_1 - mtx_1.mean(axis=1, keepdims=True)
    mtx_1 /= mtx_1.std(axis=1, keepdims=True)
    mtx_2 = mtx_2 - mtx_2.mean(axis=1, keepdims=True)
    mtx_2 /= mtx_2.std(axis=1, keepdims=True)
    corr_vec = np.mean(mtx_1 * mtx_2, axis=1)
    return corr_vec


def compute_relative_rowwise_l2_distance(mtx_1, mtx_2):
    rowwise_l2 = compute_rowwise_l2_distance(mtx_1, mtx_2)
    archetype_dispersion = np.mean(pdist(mtx_1))  # average pairwise distance
    rowwise_l2_normalized = rowwise_l2 / archetype_dispersion
    return rowwise_l2_normalized
