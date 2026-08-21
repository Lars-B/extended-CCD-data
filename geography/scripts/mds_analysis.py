import os
import logging
import random
from itertools import combinations
from multiprocessing import Pool, cpu_count

import numpy as np

from brokilon.core import read_nexus_trees


def subsample_beast_tree_file(infile, outputfile, nsamples=50):
    ntotal = 0
    with open(infile) as fin:
        for line in fin:
            if line.lstrip().lower().startswith("tree"):
                ntotal += 1

    if ntotal < nsamples:
        raise ValueError(f"Too many trees requested {nsamples}, file only contains {ntotal}")

    indices = {round(i * (ntotal - 1) / (nsamples - 1)) + 1 for i in range(nsamples)}

    # print(f"Will pick these trees: {indices}")

    tree_counter = 0
    in_trees_block = False
    in_translate = False

    with open(infile) as fin, open(outputfile, "w") as fout:
        for line in fin:
            stripped = line.strip()

            if not in_trees_block:
                fout.write(line)
                if stripped.lower().startswith("begin trees"):
                    in_trees_block = True
                continue

            if stripped.lower().startswith("translate"):
                in_translate = True
                fout.write(line)
                continue

            if in_translate:
                fout.write(line)
                if stripped.endswith(";"):
                    in_translate = False
                continue
            if stripped.lower().startswith("tree"):
                tree_counter += 1
                if tree_counter in indices:
                    fout.write(line)
                continue

            if stripped.lower().startswith("end"):
                fout.write(line)
                in_trees_block = False
                continue


def _dist_task(args):
    i, j, ti, tj, dist = args
    return i, j, dist(ti, tj)


def pairwise_distances_parallel(trees, dist, ncores=None):
    n = len(trees)
    D = np.zeros((n, n), dtype=int)

    tasks = (
        (i, j, trees[i], trees[j], dist)
        for i, j in combinations(range(n), 2)
    )

    with Pool(ncores or cpu_count()) as pool:
        for i, j, d in pool.imap_unordered(_dist_task, tasks, chunksize=10):
            D[i, j] = d
            D[j, i] = d

    return D


def mds_stuff():
    np.random.seed(42)
    random.seed(42)

    tree_file = "../data/subsample.trees"

    if not os.path.exists(tree_file):
        subsample_beast_tree_file(
            infile="../data/combined_chains.typed.node.trees",
            outputfile=tree_file,
            nsamples=750
        )

    trees, taxon_map = read_nexus_trees(tree_file, parse_taxon_map=True)
    print(len(trees))

    labels = ["posterior"] * len(trees)

    # Adding CCD sample
    logging.info("Adding CCD sample")

    from brokilon.ccd.domain.phylogeography import get_geo_map
    geo_ccd_map, _, clade_count_map = (
        get_geo_map(trees, geo_ann_str="type", ccd_type=1)
    )

    from brokilon.ccd.domain.phylogeography import sample_trees_from_geo_ccd
    ccd_sample = sample_trees_from_geo_ccd(len(trees), geo_ccd_map, "type", clade_count_map)

    ccd_labels = ["ccd-sample"] * len(trees)

    trees.extend(ccd_sample)
    labels.extend(ccd_labels)

    # adding special trees
    logging.info("Adding summary trees")

    extra_trees = (
        ("../data/combined_chains_mcc.typed.node.tree", "MCC"),
        ("../data/ext-ccd.noB.tree", "ext-CCD1"),
        ("../data/reg_ccd0.tree", "CCD0"),
        ("../data/reg_ccd1.tree", "CCD1"),
        ("../data/combined.hipstr.0.tree", "HIPSTR"),
    )

    for cur_file, label in extra_trees:
        cur_tree, cur_map = read_nexus_trees(cur_file, parse_taxon_map=True)
        if not cur_map == taxon_map:
            raise NotImplementedError("Maps are different!")
        trees.append(cur_tree[0])
        labels.append(label)

    # RF
    rf_file = "pwd_rf.npy"
    if os.path.exists(rf_file):
        logging.info(f"Loading RF distances from {rf_file}")
        pwd_rf = np.load(rf_file)
    else:
        logging.info("Computing RF distances")
        from brokilon.metrics import robinson_foulds
        pwd_rf = pairwise_distances_parallel(trees, dist=robinson_foulds)
        np.save(rf_file, pwd_rf)
        logging.info(f"Saved RF distances to {rf_file}")

    # e RF distances
    erf_file = "pwd_erf.npy"
    if os.path.exists(erf_file):
        logging.info(f"Loading ERF distances from {erf_file}")
        pwd_extended_rf = np.load(erf_file)
    else:
        logging.info("Computing extended RF distances")
        from brokilon.metrics import deme_robinson_foulds
        from functools import partial
        deme_rf = partial(deme_robinson_foulds, annotation_str="type")
        pwd_extended_rf = pairwise_distances_parallel(trees, dist=deme_rf)
        np.save(erf_file, pwd_extended_rf)
        logging.info(f"Saved extended RF distances to {erf_file}")

    # MDS coords
    logging.info("Computing MDS coordinates")
    from sklearn.manifold import MDS

    embedding = MDS(
        n_components=2,
        n_init=1,
        dissimilarity="precomputed"
    )
    rf_coords = embedding.fit_transform(pwd_rf)

    erf_coords = embedding.fit_transform(pwd_extended_rf)

    logging.info("Plotting coordinates...")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(
        1, 2,
        figsize=(10, 5),
    )

    style = {
        "posterior":         dict(s=20, marker="o", alpha=0.4, color="black"),
        "ccd-sample":        dict(s=20, marker="P", alpha=0.4, color="#0072B2"),  # blue
        "MCC":               dict(s=60, marker="v", color="#CC79A7"),  # pink/purple
        "ext-CCD1":          dict(s=60, marker="v", color="#E69F00"),  # orange
        "ext-CCD1-burnin":   dict(s=60, marker="v", color="#D55E00"),  # vermillion (darker orange-red)
        "CCD0":              dict(s=60, marker="v", color="#009E73"),  # bluish green
        "CCD1":              dict(s=60, marker="v", color="#F0E442"),  # yellow
        "HIPSTR":            dict(s=60, marker="v", color="#56B4E9"),  # sky blue
    }

    post_idx = [i for i, l in enumerate(labels) if l == "posterior"]
    ccd_idx = [i for i, l in enumerate(labels) if l == "ccd-sample"]

    ax[0].scatter(
        rf_coords[post_idx, 0],
        rf_coords[post_idx, 1],
        **style["posterior"]
    )
    ax[0].set_title("RF coordinates", fontsize=18)

    ax[1].scatter(
        erf_coords[post_idx, 0],
        erf_coords[post_idx, 1],
        **style["posterior"]
    )
    ax[1].set_title("Extended RF coordinates", fontsize=18)

    # Adding CCD sampled points
    ax[0].scatter(
        rf_coords[ccd_idx, 0],
        rf_coords[ccd_idx, 1],
        **style["ccd-sample"]
    )
    ax[1].scatter(
        erf_coords[ccd_idx, 0],
        erf_coords[ccd_idx, 1],
        **style["ccd-sample"]
    )

    legend_offset = 4

    texts_rf = []
    texts_erf = []

    for i, label in enumerate(labels):
        if label in ("posterior", "ccd-sample"):
            continue

        ax[0].scatter(
            rf_coords[i, 0],
            rf_coords[i, 1],
            **style[label]
        )

        ax[1].scatter(
            erf_coords[i, 0],
            erf_coords[i, 1],
            **style[label]
        )

        import matplotlib.patheffects as pe
        texts_rf.append(
            ax[0].text(
                rf_coords[i, 0],
                rf_coords[i, 1],  # + legend_offset,
                label,
                fontsize=9,
                ha="center",
                va="bottom",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )
        )

        texts_erf.append(
            ax[1].text(
                erf_coords[i, 0],
                erf_coords[i, 1],  # + legend_offset,
                label,
                fontsize=9,
                ha="center",
                va="bottom",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )
        )

    from adjustText import adjust_text
    adjust_text(
        texts_rf,
        x=rf_coords[:, 0],
        y=rf_coords[:, 1],
        ax=ax[0],
        expand_points=(3, 3),
        force_points=2.0,
        force_text=1,
        # arrowprops=dict(arrowstyle="-", lw=0.5)
    )
    adjust_text(
        texts_erf,
        x=erf_coords[-5:, 0],
        y=erf_coords[-5:, 1],
        ax=ax[1],
        expand_points=(3, 3),
        force_points=2.0,
        force_text=1,
        # arrowprops=dict(arrowstyle="-", lw=0.5)
    )

    for a in ax:
        a.set_aspect("equal", adjustable="box")
        a.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False
        )

    fig.tight_layout()
    # plt.show()
    plt.savefig("../plots/mds_comparison.pdf", bbox_inches='tight')


def distance_matrix_summary_trees():
    summary_trees = (
        ("combined_chains_mcc.typed.node.tree", "MCC"),
        # ("ann_ext_ccd.tree", "ext-CCD"),
        ("combined.hipstr.0.tree", "HIPSTR"),
        ("ext-ccd.noB.tree", "ext-CCD1"),
        ("ext-ccd.b10.tree", "ext-CCD1-burnin"),
        ("reg_ccd0.tree", "CCD0"),
        ("reg_ccd1.tree", "CCD1"),
    )

    sum_tree_list = {}

    for file, label in summary_trees:
        cur = read_nexus_trees(f"../data/{file}", parse_taxon_map=True)
        sum_tree_list[label] = cur

    from brokilon.metrics import robinson_foulds, deme_robinson_foulds

    for t in sum_tree_list:
        for t1 in sum_tree_list:
            print(
                t, t1,
                robinson_foulds(
                    sum_tree_list[t][0][0],
                    sum_tree_list[t1][0][0]
                ),
                deme_robinson_foulds(
                    sum_tree_list[t][0][0],
                    sum_tree_list[t1][0][0],
                    annotation_str="type"
                )
            )
        from brokilon.metrics.generalized_robinson_foulds import _get_deme_clades
        from brokilon.ccd.domain.topology.ccd import get_clades
        cd = _get_deme_clades(sum_tree_list[t][0][0], "type")
        c = get_clades(sum_tree_list[t][0][0])
        print(f"{t}: Nbr of clades: {len(c)}, deme clades: {len(cd)}")

    return None


def pwd_distribution():
    rf_file = "pwd_rf.npy"
    if not os.path.exists(rf_file):
        raise ValueError("Run mds plot before this to compute and save pwd matrix.")
    pwd_rf = np.load(rf_file)
    erf_file = "pwd_erf.npy"
    if not os.path.exists(erf_file):
        raise ValueError("Run mds plot before this to compute and save pwd matrix.")
    pwd_extended_rf = np.load(erf_file)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharey=True, sharex=True)

    n = pwd_rf.shape[0]
    iu = np.triu_indices(n, k=1)

    rf_vals = pwd_rf[iu]
    ext_rf_vals = pwd_extended_rf[iu]

    ax.scatter(rf_vals, ext_rf_vals, alpha=0.3, s=10, color="blue", zorder=2, rasterized=True)

    # lo = min(rf_vals.min(), ext_rf_vals.min())
    # hi = max(rf_vals.max(), ext_rf_vals.max())
    # ax.set_xlim(lo, hi)
    # ax.set_ylim(lo, hi)
    # ax.set_autoscale_on(False)   # lock limits before adding the infinite line

    ax.axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.5, zorder=0)

    ax.set_xlim(120, 250)
    ax.set_ylim(120, 250)

    ax.set_xlabel("RF", fontsize=25)
    ax.set_ylabel("extended RF", fontsize=25)

    # plt.rcParams['xtick.labelsize'] = 25

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.yticks([], [])

    plt.tight_layout()
    # plt.show()
    plt.savefig("../plots/distance_distribution.pdf", dpi=300)


if __name__ == '__main__':
    # mds_stuff()
    # distance_matrix_summary_trees()
    pwd_distribution()
