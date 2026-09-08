import os
import sys

if __name__ == "__main__":

    sys.path.append("../../geography/scripts")
    from mds_analysis import subsample_beast_tree_file, pairwise_distances_parallel

    posterior_file = "../data/paper_run/measles-filtered.trees"
    subsample_file = "../data/paper_run/subsampled.trees"

    if not os.path.exists(subsample_file):
        subsample_beast_tree_file(
            posterior_file,
            subsample_file,
            nsamples=750,
        )

    from brokilon.ccd import read_breath_nexus
    from brokilon.ccd.domain.transmission import get_transmission_maps
    from brokilon.ccd.domain.transmission.transmission_ccd import (
        sample_trees_from_transmission_ccd1
    )

    subsample, taxon_map = read_breath_nexus(subsample_file, parse_taxon_map=True)
    m1, m2, blockcount_map, branch_lengths_map = get_transmission_maps(
        subsample,
        type_str="Ancestry")

    ccd_sample = sample_trees_from_transmission_ccd1(
        len(subsample),
        m1,
        m2,
        blockcount_map,
        branch_lengths_map
    )

    labels = ["posterior"] * len(subsample)
    ccd_labels = ["ccd-sample"] * len(ccd_sample)
    labels.extend(ccd_labels)

    all_trees = subsample + ccd_sample

    extra_tree = (
        ("../data/paper_run/tCCD.10b.tree", "ext-CCD1"),
        ("../data/paper_run/ccd0.10b.tree", "CCD0"),
        ("../data/paper_run/mcc.10b.tree", "MCC"),
    )

    for file, label in extra_tree:
        cur_tree, cur_map = read_breath_nexus(file, parse_taxon_map=True)
        if not cur_map == taxon_map:
            raise NotImplementedError("Maps are not the same!")

        all_trees.append(cur_tree[0])
        labels.append(label)

    import numpy as np
    # RF distances
    rf_file = "pwd_rf.npy"
    if os.path.exists(rf_file):
        pwd_rf = np.load(rf_file)
    else:
        from brokilon.metrics import robinson_foulds
        pwd_rf = pairwise_distances_parallel(all_trees, dist=robinson_foulds, ncores=3)
        np.save(rf_file, pwd_rf)

    # eRF distances
    erf_file = "pwd_erf.npy"
    if os.path.exists(erf_file):
        pwd_extended_rf = np.load(erf_file)
    else:
        from brokilon.metrics import deme_robinson_foulds
        from functools import partial
        deme_rf = partial(deme_robinson_foulds, annotation_str="blockcount")
        pwd_extended_rf = pairwise_distances_parallel(all_trees, dist=deme_rf, ncores=3)
        np.save(erf_file, pwd_extended_rf)

    from sklearn.manifold import MDS

    embedding = MDS(
        n_components=2,
        n_init=1,
        dissimilarity="precomputed",
        random_state=1337
    )
    rf_coords = embedding.fit_transform(pwd_rf)

    erf_coords = embedding.fit_transform(pwd_extended_rf)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        1, 2,
        figsize=(10, 5),
    )

    style = {
        "posterior":         dict(s=20, marker="o", alpha=0.4, color="black"),
        "ccd-sample":        dict(s=20, marker="P", alpha=0.4, color="#0072B2"),
        "MCC":               dict(s=120, marker="v", color="#CC79A7"),
        "ext-CCD1":          dict(s=120, marker="v", color="#E69F00"),
        "CCD0":              dict(s=120, marker="v", color="#009E73"),
        # "CCD0": dict(s=60, marker="v", color="purple"),
        # "ext-CCD1": dict(s=60, marker="v", color="orange"),
        # "MCC": dict(s=60, marker="v", color="green"),
        # "CCD0": dict(s=60, marker="v", color="red"),
        # "CCD1": dict(s=60, marker="v", color="green"),
    }

    
    post_idx = [i for i, l in enumerate(labels) if l == "posterior"]
    ccd_idx = [i for i, l in enumerate(labels) if l == "ccd-sample"]

    # Posterior points
    ax[0].scatter(
        rf_coords[post_idx, 0],
        rf_coords[post_idx, 1],
        **style["posterior"],
        # label="Posterior",
    )
    ax[0].set_title("RF coordinates", fontsize=18)

    ax[1].scatter(
        erf_coords[post_idx, 0],
        erf_coords[post_idx, 1],
        **style["posterior"],
        # label="Posterior",
    )
    ax[1].set_title("Extended RF coordinates", fontsize=18)

    # CCD sampled points
    ax[0].scatter(
        rf_coords[ccd_idx, 0],
        rf_coords[ccd_idx, 1],
        **style["ccd-sample"],
        # label="CCD sample",
    )

    ax[1].scatter(
        erf_coords[ccd_idx, 0],
        erf_coords[ccd_idx, 1],
        **style["ccd-sample"],
        # label="CCD sample",
    )

    # Other points
    for i, label in enumerate(labels):
        if label in ("posterior", "ccd-sample"):
            continue

        ax[0].scatter(
            rf_coords[i, 0],
            rf_coords[i, 1],
            **style[label],
            label=label,
        )

        ax[1].scatter(
            erf_coords[i, 0],
            erf_coords[i, 1],
            **style[label],
            label=label,
        )


    # ---------------------------------------------------------
    # Shared legend
    # ---------------------------------------------------------

    # Get handles/labels from the first axis
    handles, legend_labels = ax[0].get_legend_handles_labels()

    # Remove duplicate labels while preserving order
    unique = dict(zip(legend_labels, handles))

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=4,
        fontsize=14,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )


    # ---------------------------------------------------------
    # Axis formatting
    # ---------------------------------------------------------

    for a in ax:
        a.set_aspect("equal", adjustable="box")
        a.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    # Leave room for the shared legend
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig("../plots/mds_comparison.pdf", bbox_inches='tight')
