import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from partipy.arch import AA

X = np.array(pd.read_csv(Path("benchmarking") / "aa_toy_data.csv").values)

# X = np.vstack((X, np.array([30, 31, 30, 33.5, 31, 32.1]).reshape(-1, 2)))

feature_means = X.mean(axis=0, keepdims=True)
feature_stds = X.std(axis=0, keepdims=True)
X -= feature_means
X /= feature_stds

A, B, Z, RSS, varexpl = (
    AA(
        n_archetypes=3,
        init="furthest_sum",
        # optim="regularized_nnls",
        optim="projected_gradients",
        # weight="bisquare",
        weight=None,
    )
    .fit(X=X)
    .return_all()
)

plt.style.use("dark_background")
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.scatter(Z[:, 0], Z[:, 1], color="red")
plt.show()
