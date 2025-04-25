# def construct_coreset(X: np.ndarray,
#                      coreset_size: int,
#                      seed: int):
#
#    n_samples = X.shape[0]
#
#    sq_dists = np.square(cdist(XA=X, XB=X.mean(axis=0, keepdims=True)).flatten())
#    probs = sq_dists / sq_dists.sum()
#
#    rng = np.random.default_rng(seed=seed)
#    coreset_indices = rng.choice(a=np.arange(n_samples), size=coreset_size, replace=False, p=probs)
#
#    weights = (probs[coreset_indices] * coreset_size)**(-1)
#
#    return X[coreset_indices], weights
