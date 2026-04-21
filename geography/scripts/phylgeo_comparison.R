library(ggtree)
library(treeio)
library(ggplot2)
require(gridExtra)
library(cowplot)

plot_two_trees_with_legend <- function(
    tree_left,
    tree_right,
    title_left,
    title_right
) {
  legend_height <- 1
  # Base plots (no legend)
  p_left <- ggtree(
    tree_left,
    branch.length = "none",
    aes(color = type)
  ) +
    ggtitle(title_left) +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5)
    )

  p_right <- ggtree(
    tree_right,
    branch.length = "none",
    aes(color = type)
  ) +
    scale_x_reverse() +
    ggtitle(title_right) +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5)
    )

  # Legend extracted from left plot
  legend <- get_legend(
    p_left +
      theme(legend.position = "right") +
      guides(color = guide_legend(nrow = 1, byrow = TRUE))
  )

  # Combine trees
  trees <- grid.arrange(p_left, p_right, ncol = 2)

  # Final layout
  final.plot <- grid.arrange(
    trees,
    legend,
    ncol = 1,
    heights = c(10, legend_height)
  )

  return(final.plot)
}


eccd.10.file <- "../data/ext-ccd.b10.tree"
eccd.10.tree <- read.beast(eccd.10.file)

eccd.noB.file <- "../data/ext-ccd.noB.tree"
eccd.noB.tree <- read.beast(eccd.noB.file)

mcc.file <- "../data/combined_chains_mcc.typed.node.tree"
mcc.tree <- read.beast(mcc.file)

mcc.10.file <- "../data/combined.mcc.10.tree"
mcc.10.tree <- read.beast(mcc.10.file)

mcc.noB.file <- "../data/combined.mcc.0.tree"
mcc.noB.tree <- read.beast(mcc.noB.file)


# eCCD.compare <- plot_two_trees_with_legend(eccd.noB.tree, eccd.10.tree, "eCCD no burnin", "eCCD 10% burnin")
# ggsave(filename="eccd_comparison.pdf", plot=eCCD.compare, width=10, height=6)

# plot_two_trees_with_legend(mcc.noB.tree, mcc.10.tree, "MCC no burnin", "MCC 10% burnin")

mcc.eccd.noB <- plot_two_trees_with_legend(mcc.tree, eccd.noB.tree, "MCC", "eCCD no burnin")
ggsave(filename="mcc_eccd-noB_comparison.pdf", plot=mcc.eccd.noB, width=10, height=6)
mcc.eccd.10 <- plot_two_trees_with_legend(mcc.tree, eccd.10.tree, "MCC", "eCCD 10% burnin")
ggsave(filename="mcc_eccd-10B_comparison.pdf", plot=mcc.eccd.10, width=10, height=6)


# Plotting the single MRC tree
mrc.tree.file <- "../data/extended.mrc.tree"
emrc.tree <- read.beast(mrc.tree.file)

emrc.plot <- ggtree(
  emrc.tree,
  branch.length = "none",
  aes(color = type)
) +
  ggtitle("Extended MRC tree") +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5)
  ) +
    theme(legend.position = "bottom") +
    guides(color = guide_legend(nrow = 1, byrow = TRUE))

emrc.plot
ggsave(filename="emrc.pdf", plot=emrc.plot, width=10, height=6)
