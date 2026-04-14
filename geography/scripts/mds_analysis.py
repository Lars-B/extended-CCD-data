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

    # todo use the new sample_trees_from_geo_ccd() to add ccd samples to the MDS plot in a different color...

    tree_file = "../data/testing_subsampling.trees"
    trees, taxon_map = read_nexus_trees(tree_file, parse_taxon_map=True)
    print(len(trees))

    labels = ["posterior"] * len(trees)

    # Adding CCD sample
    logging.info("Adding CCD sample")

    from brokilon.ccd.domain.phylogeography import get_geo_map
    geo_ccd_map, branch_lengths_map, clade_count_map = (
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
        ("../data/ann_ext_ccd.tree", "ext-CCD"),
        ("../data/reg_ccd0.tree", "CCD0"),
        ("../data/reg_ccd1.tree", "CCD1"),
    )

    for cur_file, label in extra_trees:
        cur_tree, cur_map = read_nexus_trees(cur_file, parse_taxon_map=True)
        if not cur_map == taxon_map:
            raise NotImplementedError("Maps are different!")
        trees.append(cur_tree[0])
        labels.append(label)

    # RF
    logging.info("Computing RF distances")
    from brokilon.metrics import robinson_foulds
    pwd_rf = pairwise_distances_parallel(trees, dist=robinson_foulds)

    # e RF distances
    logging.info("Computing extended RF distances")
    from brokilon.metrics import deme_robinson_foulds
    from functools import partial
    deme_rf = partial(deme_robinson_foulds, annotation_str="type")
    pwd_extended_rf = pairwise_distances_parallel(trees, dist=deme_rf)

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
        "posterior": dict(s=30, marker="o", alpha=0.6, color="black"),
        "ccd-sample": dict(s=30, marker="P", alpha=0.6, color="blue"),
        "MCC": dict(s=60, marker="v", color="purple"),
        "ext-CCD": dict(s=60, marker="v", color="orange"),
        "CCD0": dict(s=60, marker="v", color="red"),
        "CCD1": dict(s=60, marker="v", color="green"),
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

        # optional: annotate
        ax[0].text(
            rf_coords[i, 0],
            rf_coords[i, 1] + legend_offset,
            label,
            fontsize=9,
            ha="center",
            va="bottom",
        )

        ax[1].text(
            erf_coords[i, 0],
            erf_coords[i, 1] + legend_offset,
            label,
            fontsize=9,
            ha="center",
            va="bottom",
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
    plt.savefig("../plots/mds_initial.pdf", bbox_inches='tight')


def distance_matrix_summary_trees():
    # todo needs implementation etc....
    summary_trees = (
        ("combined_chains_mcc.typed.node.tree", "MCC"),
        ("ann_ext_ccd.tree", "ann-CCD"),
        ("reg_ccd0.tree", "CCD0"),
        ("reg_ccd1.tree", "CCD1"),
    )

    sum_tree_list = {}

    for file, label in summary_trees:
        cur = read_nexus_trees(file, parse_taxon_map=True)
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
    tree_file = "testing_subsampling.trees"
    trees, taxon_map = read_nexus_trees(tree_file, parse_taxon_map=True)

    extra_trees = (
        ("combined_chains_mcc.typed.node.tree", "MCC"),
        ("ann_ext_ccd.tree", "ann-CCD"),
        ("reg_ccd0.tree", "CCD0"),
        ("reg_ccd1.tree", "CCD1"),
    )

    for cur_file, _ in extra_trees:
        cur_tree, cur_map = read_nexus_trees(cur_file, parse_taxon_map=True)
        if not cur_map == taxon_map:
            raise NotImplementedError("Maps are different!")
        trees.append(cur_tree[0])

    # RF
    logging.info("Computing RF distances")
    from brokilon.metrics import robinson_foulds
    pwd_rf = pairwise_distances_parallel(trees, dist=robinson_foulds)

    # e RF distances
    logging.info("Computing extended RF distances")
    from brokilon.metrics import deme_robinson_foulds
    from functools import partial
    deme_rf = partial(deme_robinson_foulds, annotation_str="type")
    pwd_extended_rf = pairwise_distances_parallel(trees, dist=deme_rf)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharey=True, sharex=True)
    # ax = plt.figure()

    ax.hist(
        np.triu(pwd_rf, k=1).flatten(),
        bins=250,
        alpha=0.5,
        label="RF distances",
        density=True,
        color="blue",
    )
    ax.hist(
        np.triu(pwd_extended_rf, k=1).flatten(),
        bins=250,
        alpha=0.5,
        label="ext-RF distances",
        density=True,
        color="red"
    )

    plt.xlim([180, 250])
    plt.ylim([0, 0.08])

    plt.xlabel("Tree distances", fontsize=25)
    plt.ylabel("Density", fontsize=25)

    # plt.rcParams['xtick.labelsize'] = 25

    plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    plt.yticks([], [])

    plt.legend(fontsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig("distance_distribution.pdf")


if __name__ == '__main__':
    # subsample_beast_tree_file(
    #     infile="combined_chains.typed.node.trees",
    #     outputfile="testing_subsampling.trees",
    #     nsamples=100
    # )

    mds_stuff()
    # distance_matrix_summary_trees()
    # pwd_distribution()
