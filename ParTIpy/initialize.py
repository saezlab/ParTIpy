import numpy as np
from scipy.sparse import csr_matrix


def _random_init(X: np.ndarray, n_archetypes: int, exclude: None | list = None, seed: int = 42) -> np.ndarray:
    if exclude is None:
        exclude = []
    B = np.eye(N=n_archetypes, M=X.shape[0])
    return B


def _furthest_sum_init(X: np.ndarray, n_archetypes: int, exclude: None | list = None, seed: int = 42) -> np.ndarray:
    """Furthest sum algorithm, to efficiently generat initial archetypes.

    Parameters
    ----------
    X: numpy 2d-array
        data matrix with shape (n_samples x n_features)

    n_archetypes: int
        number of candidate archetypes to extract.

    exclude: numpy 1d-array
        entries in X that can not be used as candidates.

    seed: int
        reproducibility

    Output
    ------
    B : numpy 2d-array
        B matrix with shape (n_archetypes, n_samples).
    """
    np.random.seed(seed)

    if exclude is None:
        exclude = []

    def max_ind_val(input_list):
        return max(zip(range(len(input_list)), input_list, strict=False), key=lambda x: x[1])

    K = X.T
    D, N = K.shape
    index = np.array(range(N))
    index[exclude] = 0
    i = [int(np.ceil(N * np.random.rand()))]
    index[i] = -1
    ind_t = i
    sum_dist = np.zeros((1, N), np.complex128)

    if N > n_archetypes * D:
        Kt = K
        Kt2 = np.sum(Kt**2, axis=0)
        for k in range(1, n_archetypes + 11):
            if k > n_archetypes - 1:
                Kq = Kt[:, i[0]] @ Kt
                sum_dist -= np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[i[0]])
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            Kq = Kt[:, ind_t].T @ Kt
            sum_dist += np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[ind_t])
            ind, _ = max_ind_val(sum_dist[:, t][0].real)
            ind_t = t[ind]
            i.append(ind_t)  # type: ignore
            index[ind_t] = -1
    else:
        if D != N or np.sum(K - K.T) != 0:  # Generate kernel if K not one
            Kt = K
            K = Kt.T @ Kt
            K = np.lib.scimath.sqrt(np.matlib.repmat(np.diag(K), N, 1) - 2 * K + np.matlib.repmat((np.diag(K)).T, 1, N))  # type: ignore[attr-defined]

        Kt2 = np.diag(K)  # Horizontal
        for k in range(1, n_archetypes + 11):
            if k > n_archetypes - 1:
                sum_dist -= np.lib.scimath.sqrt(Kt2 - 2 * K[i[0], :] + Kt2[i[0]])
                index[i[0]] = i[0]
                i = i[1:]
            t = np.where(index != -1)[0]
            sum_dist += np.lib.scimath.sqrt(Kt2 - 2 * K[ind_t, :] + Kt2[ind_t])
            ind, _ = max_ind_val(sum_dist[:, t][0].real)
            ind_t = t[ind]
            i.append(ind_t)  # type: ignore
            index[ind_t] = -1

    B = csr_matrix((np.ones(len(i)), (i, range(n_archetypes))), shape=(N, n_archetypes)).toarray().T
    B = np.ascontiguousarray(B)
    return B
