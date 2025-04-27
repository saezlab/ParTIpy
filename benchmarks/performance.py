import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import partipy as pt
import seaborn as sns
from partipy.utils import align_archetypes, compute_rowwise_l2_distance
from scipy.spatial.distance import pdist
from tqdm import tqdm

benchmark_path = Path(".") / "benchmarks"

benchmark_start_time = time.time()

### define simulation setups ###
simulation_seed_list = list(range(5))
n_samples_list = [100, 1_000, 10_000]
n_features_list = [10, 20]
n_archetypes_list = [3, 7]
noise_std_list = [0.00, 0.05, 0.10]
simulation_settings_list = []
for n_samples in n_samples_list:
    for n_features in n_features_list:
        for n_archetypes in n_archetypes_list:
            for noise_std in noise_std_list:
                simulation_settings_list.append(
                    {
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "n_archetypes": n_archetypes,
                        "noise_std": noise_std,
                    }
                )
print(f"{len(simulation_settings_list)=}")

### define optimization settings to consider ###
init_alg_list = pt.const.INIT_ALGS
optim_alg_list = [alg for alg in pt.const.OPTIM_ALGS if alg != "regularized_nnls"]
optim_settings_list = []
for init_alg in init_alg_list:
    for optim_alg in optim_alg_list:
        optim_settings_list.append(
            {
                "init_alg": init_alg,
                "optim_alg": optim_alg,
            }
        )
print(f"{len(optim_settings_list)=}")

### prepare lists to save the results ###
result_list = []

### execution loop ###
for simulation_dict in simulation_settings_list:
    print("\n\t", simulation_dict)
    for optim_dict in optim_settings_list:
        print("\t\t", optim_dict)
        for _simulation_iter, simulation_seed in tqdm(enumerate(simulation_seed_list)):
            # simulate the data
            X, A, Z = pt.simulate_archetypes(
                n_samples=int(simulation_dict["n_samples"]),
                n_archetypes=int(simulation_dict["n_archetypes"]),
                n_dimensions=int(simulation_dict["n_features"]),
                noise_std=simulation_dict["noise_std"],
                seed=simulation_seed,
            )
            X, A, Z = X.astype(np.float32), A.astype(np.float32), Z.astype(np.float32)

            # fit the model and measure execution time and performance
            start_time = time.time()
            model = pt.AA(
                n_archetypes=int(simulation_dict["n_archetypes"]),
                init=optim_dict["init_alg"],
                optim=optim_dict["optim_alg"],
                weight=None,
                verbose=False,
            )
            model.fit(X=X)
            end_time = time.time()

            # compute the l2 distance between the true archetypes and the (aligned) learned archetypes
            Z_aligned = align_archetypes(ref_arch=Z, query_arch=model.Z)
            l2_distances = compute_rowwise_l2_distance(mtx_1=Z, mtx_2=Z_aligned)
            l2_distances_mean = l2_distances.mean()
            l2_distances_mean_normalized = l2_distances_mean / np.mean(pdist(Z))

            # record the results
            execution_time = end_time - start_time

            result_dict = {
                "time": execution_time,
                "rss": model.RSS,
                "varexpl": model.varexpl,
                "seed": simulation_seed,
                "l2_dist": l2_distances_mean,
                "l2_dist_norm": l2_distances_mean_normalized,
            }
            result_dict = result_dict | model.fitting_info  # type: ignore[operator, assignment, attr-defined]
            result_dict = result_dict | simulation_dict  # type: ignore[operator, assignment, attr-defined]
            result_dict = result_dict | optim_dict  # type: ignore[operator, assignment, attr-defined]
            result_list.append(result_dict)

result_df = pd.DataFrame(result_list)
result_df["time_log10"] = np.log10(result_df["time"])
result_df.to_csv(benchmark_path / "results.csv", index=False)

all_features = ["time_log10", "varexpl", "l2_dist_norm"]

for noise_std in noise_std_list:
    for feature in all_features:
        unique_features = result_df.loc[result_df["noise_std"] == noise_std, :]["n_features"].sort_values().unique()
        unique_archetypes = result_df.loc[result_df["noise_std"] == noise_std, :]["n_archetypes"].sort_values().unique()

        fig, axes = plt.subplots(
            nrows=len(unique_features),
            ncols=len(unique_archetypes),
            figsize=(3 * len(unique_archetypes) + 3, 3 * len(unique_features)),
            squeeze=False,
        )

        # use this to store handles/labels just once
        legend_handles = None
        legend_labels = None

        rng = np.random.default_rng(seed=42)
        for row_idx, n_features in enumerate(unique_features):
            for col_idx, n_archetypes in enumerate(unique_archetypes):
                ax = axes[row_idx, col_idx]
                selection_vec = (
                    (result_df["noise_std"] == noise_std)
                    & (result_df["n_features"] == n_features)
                    & (result_df["n_archetypes"] == n_archetypes)
                )
                result_df_subset = result_df.loc[selection_vec, :].copy()

                # summarize accross the different seeds
                result_df_subset = (
                    result_df_subset.groupby(  # type: ignore[assignment]
                        ["n_samples", "n_features", "n_archetypes", "optim_alg", "init_alg"], as_index=False
                    )[feature].agg("mean")  # or use .agg(["mean", "std"]) for multiple statistics
                )

                # adding some jitter
                result_df_subset["n_samples"] = result_df_subset["n_samples"] + result_df_subset[
                    "n_samples"
                ] * rng.normal(loc=0, scale=0.1, size=len(result_df_subset))
                result_df_subset["n_samples_log10"] = np.log10(result_df_subset["n_samples"])

                # draw plot and capture handles/labels only once
                plot = sns.scatterplot(
                    data=result_df_subset,
                    x="n_samples_log10",
                    y=feature,
                    hue="optim_alg",
                    style="init_alg",
                    ax=ax,
                    alpha=0.9,
                )
                if legend_handles is None:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()

                ax.legend_.remove()  # remove local legend
                ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)  # <-- Add this line

        # put global legend outside the figure
        fig.legend(
            handles=legend_handles, labels=legend_labels, loc="center right", bbox_to_anchor=(1.1, 0.5), frameon=False
        )

        # Add row labels (n_features)
        for row_idx, n_features in enumerate(unique_features):
            fig.text(
                x=0.01,
                y=0.5 - (row_idx - len(unique_features) / 2 + 0.5) / len(unique_features),
                s=f"{n_features} features",
                va="center",
                ha="left",
                fontsize=12,
                rotation=90,
            )

        # Add column labels (n_archetypes)
        for col_idx, n_archetypes in enumerate(unique_archetypes):
            fig.text(
                x=(col_idx + 0.5) / len(unique_archetypes),
                y=0.95,
                s=f"{n_archetypes} archetypes",
                va="bottom",
                ha="center",
                fontsize=12,
            )

        fig.suptitle(f"Comparison at {noise_std} Noise Level | {feature}", fontsize=16, y=1.02)
        plt.tight_layout(rect=(0.02, 0.00, 0.85, 0.98))  # Adjust for both legend and title
        fig.savefig(benchmark_path / f"{feature}_{noise_std:.2f}.png", bbox_inches="tight")
        plt.close()

benchmark_end_time = time.time()
print(f"Benchmark took {(benchmark_end_time - benchmark_start_time) / 60} minutes")
