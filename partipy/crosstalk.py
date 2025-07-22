import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def get_specific_genes_per_archetype(archetype_expression, min_score=0.05, max_score=np.inf):
    """
    Calculate gene specificity scores for each archetype and filter genes based on specificity.

    This function identifies genes that are specifically expressed in each archetype by comparing
    their expression levels across all archetypes. It calculates a specificity score for each gene
    in each archetype, representing how much more highly expressed the gene is in that archetype
    compared to others.

    Parameters
    ----------
    archetype_expression : pd.DataFrame
        DataFrame with archetypes as rows and genes as columns, containing expression values
        (typically z-scored logp1 normalized gene expression)
    min_score : float, default=0.05
        Minimum specificity score threshold for gene filtering
    max_score : float, default=np.inf
        Maximum specificity score threshold for gene filtering

    Returns
    -------
    dict
        Dictionary where keys are archetype column names and values are DataFrames containing:
        - 'z_score': Original expression value for the gene in this archetype
        - 'max_z_score_others': Maximum expression value of this gene across all other archetypes
        - 'specificity_score': Minimum difference between this archetype's expression and all others
        - 'gene': Gene identifier
        Each DataFrame is sorted by specificity_score in descending order and filtered by score thresholds

    Notes
    -----
    The specificity score is calculated as the minimum difference between the gene's expression
    in the current archetype and its expression in all other archetypes. Higher positive values
    indicate genes that are more specifically expressed in that archetype.
    """
    expr_df = archetype_expression.T
    archetype_dict = {}
    for col in expr_df.columns:
        other_cols = [c for c in expr_df.columns if c != col]
        archetype_dict[col] = (
            pd.DataFrame(
                {
                    "z_score": expr_df[col].to_numpy(),
                    "max_z_score_others": (expr_df[other_cols].values).max(axis=1),
                    "specificity_score": (expr_df[[col]].values - expr_df[other_cols].values).min(axis=1),
                    "gene": expr_df.index.to_numpy(),
                }
            )
        ).sort_values("specificity_score", ascending=False)
        # filtering
        archetype_dict[col] = (
            archetype_dict[col]
            .loc[
                (archetype_dict[col]["specificity_score"] >= min_score)
                & (archetype_dict[col]["specificity_score"] <= max_score),
                :,
            ]
            .copy()
        )
    return archetype_dict


def get_archetype_crosstalk(archetype_genes: dict, lr_resource):
    """
    Identify potential ligand-receptor interactions between different archetypes.

    This function analyzes crosstalk between archetypes by finding ligand-receptor pairs
    where the ligand is specifically expressed in one archetype (sender) and the receptor
    is specifically expressed in another archetype (receiver). It enriches the interaction
    data with specificity scores from both sender and receiver archetypes.

    Parameters
    ----------
    archetype_genes : dict
        Dictionary from get_specific_genes_per_archetype() where keys are archetype identifiers
        and values are DataFrames containing gene specificity information with columns:
        - 'gene': Gene identifier
        - 'z_score': Expression z-score in the archetype
        - 'specificity_score': Specificity score for the gene in this archetype
    lr_resource : pd.DataFrame
        Ligand-receptor resource DataFrame containing at minimum:
        - 'ligand': Ligand gene identifier
        - 'receptor': Receptor gene identifier
        Additional columns (e.g., pathway, database source) will be preserved

    Returns
    -------
    dict
        Nested dictionary structure where:
        - First level keys: sender archetype identifiers
        - Second level keys: receiver archetype identifiers
        - Values: DataFrames containing ligand-receptor interactions with columns:
            - Original lr_resource columns (ligand, receptor, etc.)
            - 'ligand_z_score': Expression z-score of ligand in sender archetype
            - 'ligand_specificity_score': Specificity score of ligand in sender archetype
            - 'receptor_z_score': Expression z-score of receptor in receiver archetype
            - 'receptor_specificity_score': Specificity score of receptor in receiver archetype

    Notes
    -----
    - Self-interactions (same archetype as sender and receiver) are included
    - Only interactions where both ligand and receptor are found in the respective
      archetype-specific gene lists are retained
    - The function preserves all original columns from lr_resource while adding
      expression and specificity information
    """
    interactions_dict: dict[int, dict] = {}
    for sender_arch in archetype_genes.keys():
        interactions_dict[sender_arch] = {}
        for receiver_arch in archetype_genes.keys():
            sender_genes = archetype_genes[sender_arch]
            receiver_genes = archetype_genes[receiver_arch]

            lr_df = lr_resource.copy()
            lr_df = lr_df.loc[
                (lr_df["ligand"].isin(sender_genes["gene"])) & (lr_df["receptor"].isin(receiver_genes["gene"])), :
            ].copy()
            lr_df = lr_df.join(
                (
                    sender_genes.rename(
                        columns={"z_score": "ligand_z_score", "specificity_score": "ligand_specificity_score"}
                    )
                    .loc[:, ["gene", "ligand_z_score", "ligand_specificity_score"]]
                    .set_index("gene")
                ),
                how="left",
                on="ligand",
            )
            lr_df = lr_df.join(
                (
                    receiver_genes.rename(
                        columns={"z_score": "receptor_z_score", "specificity_score": "receptor_specificity_score"}
                    )
                    .loc[:, ["gene", "receptor_z_score", "receptor_specificity_score"]]
                    .set_index("gene")
                ),
                how="left",
                on="receptor",
            )
            interactions_dict[sender_arch][receiver_arch] = lr_df
    return interactions_dict


def plot_weighted_network(
    specific_genes_per_archetype,
    archetype_crosstalk_dict,
    threshold=0.0,
    layout="circular",
    seed=42,
    figsize=(8, 8),
    plot_edge_labels=False,
    return_fig=False,
    show=True,
):  # pragma: no cover
    """Create a visualization with angle-based edge label placement.

    Parameters
    ----------
    specific_genes_per_archetype : dict
        Dictionary mapping archetype indices to specific genes
    archetype_crosstalk_dict : dict
        Dictionary containing crosstalk information between archetypes
    threshold : float, default=0.0
        Minimum weight threshold for edges to be displayed
    layout : str, default="circular"
        Layout algorithm ("circular", "spring", "shell")
    seed : int, default=42
        Random seed for reproducible layouts
    figsize : tuple, default=(8, 8)
        Figure size in inches
    plot_edge_labels : bool, default=False
        Whether to plot edge labels
    return_fig : bool, default=False
        If True, return the figure object instead of showing it
    show : bool, default=True
        Whether to display the figure using plt.show()

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns figure object if return_fig=True, otherwise None
    """
    # create interaction matrix
    interactions_mtx = np.zeros((len(specific_genes_per_archetype), len(specific_genes_per_archetype)))
    for sender_arch in specific_genes_per_archetype.keys():
        for receiver_arch in specific_genes_per_archetype.keys():
            interactions_mtx[sender_arch, receiver_arch] = len(archetype_crosstalk_dict[sender_arch][receiver_arch])

    G = nx.DiGraph()
    num_nodes = interactions_mtx.shape[0]

    # Build graph
    for i in range(num_nodes):
        for j in range(num_nodes):
            if (weight := interactions_mtx[i, j]) > threshold:
                G.add_edge(i, j, weight=weight)

    # Layout
    layout_fns = {
        "circular": nx.circular_layout,
        "spring": lambda G: nx.spring_layout(G, k=1.5, iterations=100, seed=seed),
        "shell": nx.shell_layout,
    }
    pos = layout_fns.get(layout, nx.circular_layout)(G)

    # Setup plot
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # Draw nodes
    node_degree = dict(G.degree(weight="weight"))
    max_degree = max(node_degree.values()) if node_degree else 1
    node_sizes = [500 + 100 * (node_degree[n] / max_degree) for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos, node_color="lightblue", node_size=node_sizes, edgecolors="darkblue", linewidths=1.5, alpha=0.9, ax=ax
    )

    # Draw edges
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    cmap = plt.colormaps["viridis"]
    norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)) if edge_weights else None

    # Calculate edge widths based on weights (scaled)
    edge_widths = []
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        if max_weight == min_weight:
            edge_widths = [2.0] * len(edge_weights)  # Default width
        else:
            edge_widths = [1.0 + 7.0 * (w - min_weight) / (max_weight - min_weight) for w in edge_weights]

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_weights,
        edge_cmap=cmap,
        alpha=0.7,
        width=edge_widths,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="darkblue", ax=ax)

    # ANGLE-BASED EDGE LABELS
    if (G.number_of_edges() < 50) and plot_edge_labels:
        edge_labels = {(i, j): f"{interactions_mtx[i, j]:.2f}" for i, j in G.edges()}

        def get_label_pos_rotation(u, v, offset=0.08):
            """Calculate label position and rotation angle"""
            x1, y1 = pos[v]
            x2, y2 = pos[u]
            dx, dy = x2 - x1, y2 - y1

            # Midpoint coordinates
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # Perpendicular offset direction
            angle = np.arctan2(dy, dx)
            perp_angle = angle + np.pi / 2  # 90 degree offset

            # Calculate offset coordinates
            label_x = mid_x + offset * np.cos(perp_angle)
            label_y = mid_y + offset * np.sin(perp_angle)

            # Convert angle to degrees for matplotlib
            rotation = np.degrees(angle)

            return (label_x, label_y), rotation

        for (u, v), label in edge_labels.items():
            label_pos, rotation = get_label_pos_rotation(u, v)

            # Draw label with alignment following edge angle
            ax.text(
                label_pos[0],
                label_pos[1],
                label,
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=8,
                color="darkred",
                ha="center",
                va="center",
                bbox={"alpha": 0.7, "facecolor": "white", "edgecolor": "none"},
            )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Edge Weight", shrink=0.7)

    ax.axis("off")
    plt.tight_layout()

    # Return figure or show it
    if return_fig:
        plt.close()
        return fig
    else:
        if show:
            plt.show()
        plt.close()
        return None
