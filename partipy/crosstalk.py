import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def get_specific_genes_per_archetype(archetype_expression, min_score=-np.inf, max_score=np.inf):
    """TODO"""
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
    """TODO"""
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
):
    """Create a visualization with angle-based edge label placement."""
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
    plt.show()
