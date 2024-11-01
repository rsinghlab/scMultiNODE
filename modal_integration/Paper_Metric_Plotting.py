'''
Description:
    Plotting Fig. 3 in the paper.

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from plotting import *

# ===============================

def _addBarLabel(ax, value_list, thr, short_offset, long_offset, fmt="{:.2f}"):
    for index, value in enumerate(value_list):
        if value <= thr:  # Short bar
            ax.text(value + short_offset, index, fmt.format(value), va='center', fontsize=12, fontweight="bold")
        else:  # Long bar
            ax.text(value - long_offset, index, fmt.format(value), va='center', color='white', fontsize=12, fontweight="bold")

# ===============================

model_list = ["scMultiNODE", "SCOTv1", "SCOTv2", "Pamona", "UnionCom", "uniPort", "Seurat"]
n_models = len(model_list)

HC_batch_entropy = [0.667, 0.169, 0.531, 0.115, 0.138, 0.278, 0.243]
HO_batch_entropy = [0.521, 0.063, 0.020, 0.021, 0.366, 0.024, 0.431]
DR_batch_entropy = [0.614, 0.096, 0.183, 0.534, 0.145, 0.180, 0.188]
MN_batch_entropy = [0.500, 0.027, 0.161, 0.027, 0.001, 0.095, 0.075]

HC_time_correlation = [0.979, 0.560, 0.513, 0.505, 0.564, 0.125, 0.400]
HO_time_correlation = [0.974, 0.807, 0.748, 0.802, 0.931, 0.138, 0.604]
DR_time_correlation = [0.777, 0.443, 0.613, 0.303, 0.478, 0.081, 0.381]
MN_time_correlation = [0.856, 0.442, 0.243, 0.484, 0.474, 0.099, 0.436]

HC_LTA_type = [0.392, 0.561, 0.767, 0.214, 0.433, 0.308, 0.235]
HO_LTA_type = [0.955, 0.112, 0.571, 0.848, 0.946, 0.449, 0.327]
DR_LTA_type = [0.314, 0.297, 0.302, 0.279, 0.212, 0.238, 0.074]
MN_LTA_type = [0.148, 0.311, 0.285, 0.287, 0.167, 0.310, 0.231]

HC_LTA_time = [0.919, 0.258, 0.380, 0.145, 0.084, 0.165, 0.161]
HO_LTA_time = [0.895, 0.061, 0.198, 0.550, 0.773, 0.126, 0.116]
DR_LTA_time = [0.430, 0.094, 0.085, 0.116, 0.107, 0.033, 0.025]
MN_LTA_time = [0.989, 0.245, 0.336, 0.204, 0.245, 0.259, 0.356]


HC_FOSCTTM = [0.106, 0.238, 0.170, 0.421, 0.404, 0.418, 0.449]
HO_FOSCTTM = [0.097, 0.599, 0.337, 0.163, 0.080, 0.495, 0.440]

HC_NeighborOverlap = [0.203, 0.086, 0.224, 0.021, 0.045, 0.062, 0.025]
HO_NeighborOverlap = [0.054, 0.002, 0.007, 0.029, 0.094, 0.004, 0.006]

HC_SCC = [0.884, 0.771, 0.894, 0.859, 0.199, 0.098, 0.671]
HO_SCC = [0.986, 0.824, 0.409, 0.761, 0.881, 0.007, 0.900]

# ===============================

bar_height = 0.8
ms = 150
color_list = [model_color[m] for m in model_list]

# ===============================

def plotSCC():
    fig, ax_list = plt.subplots(1, 2, figsize=(4.5, 3))
    plt.subplots_adjust(wspace=0.05)

    ax_list[0].barh(np.arange(n_models), width=HC_SCC[::-1], height=bar_height, align="center", color=color_list[::-1])
    ax_list[1].barh(np.arange(n_models), width=HO_SCC[::-1], height=bar_height, align="center", color=color_list[::-1])

    _addBarLabel(ax_list[0], HC_SCC[::-1], thr=0.2, short_offset=0.05, long_offset=0.33)
    _addBarLabel(ax_list[1], HO_SCC[::-1], thr=0.2, short_offset=0.05, long_offset=0.34)

    ax_list[0].set_title("HC", fontweight="bold")
    ax_list[1].set_title("HO", fontweight="bold")

    ax_list[0].set_yticks(np.arange(n_models), model_list[::-1])
    ax_list[1].set_yticks(np.arange(n_models), ["" for _ in range(n_models)])
    fig.supxlabel(r"SCC ($\uparrow$)", y=0.1, x=0.7, fontsize=15)
    for ax in ax_list:
        removeTopRightBorders(ax)
    plt.tight_layout()
    # plt.savefig("../figs/SCC_bar.pdf", dpi=300)
    plt.show()


def plotNeighborOverlap():
    fig, ax_list = plt.subplots(1, 2, figsize=(3.5, 3))
    plt.subplots_adjust(wspace=0.05)

    ax_list[0].barh(np.arange(n_models), width=HC_NeighborOverlap[::-1], height=bar_height, align="center", color=color_list[::-1])
    ax_list[1].barh(np.arange(n_models), width=HO_NeighborOverlap[::-1], height=bar_height, align="center", color=color_list[::-1])

    _addBarLabel(ax_list[0], HC_NeighborOverlap[::-1], thr=0.05, short_offset=0.02, long_offset=0.06)
    _addBarLabel(ax_list[1], HO_NeighborOverlap[::-1], thr=0.01, short_offset=0.01, long_offset=0.035, fmt="{:.3f}")

    ax_list[0].set_title("HC", fontweight="bold")
    ax_list[1].set_title("HO", fontweight="bold")

    ax_list[0].set_yticks(np.arange(n_models), ["" for _ in range(n_models)])
    ax_list[1].set_yticks(np.arange(n_models), ["" for _ in range(n_models)])
    fig.supxlabel(r"Neighborhood Overlap ($\uparrow$)", y=0.1, fontsize=15)
    for ax in ax_list:
        removeTopRightBorders(ax)
    plt.tight_layout()
    # plt.savefig("../figs/neighbor_overlap_bar.pdf", dpi=300)
    plt.show()


def plotFOSCTTM():
    HC_FOSCTTM_inverse = 1 - np.asarray(HC_FOSCTTM)
    HO_FOSCTTM_inverse = 1 - np.asarray(HO_FOSCTTM)
    fig, ax_list = plt.subplots(1, 2, figsize=(3.5, 3))
    plt.subplots_adjust(wspace=0.05)

    ax_list[0].barh(np.arange(n_models), width=HC_FOSCTTM_inverse[::-1], height=bar_height, align="center", color=color_list[::-1])
    ax_list[1].barh(np.arange(n_models), width=HO_FOSCTTM_inverse[::-1], height=bar_height, align="center", color=color_list[::-1])

    _addBarLabel(ax_list[0], HC_FOSCTTM_inverse[::-1], thr=0.5, short_offset=0.1, long_offset=0.25)
    _addBarLabel(ax_list[1], HO_FOSCTTM_inverse[::-1], thr=0.5, short_offset=0.1, long_offset=0.25)

    ax_list[0].set_title("HC", fontweight="bold")
    ax_list[1].set_title("HO", fontweight="bold")

    # ax_list[0].set_yticks(np.arange(n_models), model_list[::-1])
    ax_list[0].set_yticks(np.arange(n_models), ["" for _ in range(n_models)])
    ax_list[1].set_yticks(np.arange(n_models), ["" for _ in range(n_models)])
    fig.supxlabel(r"1 - FOSCTTM ($\uparrow$)", y=0.1, fontsize=15)
    for ax in ax_list:
        removeTopRightBorders(ax)
    plt.tight_layout()
    # plt.savefig("../figs/FOSCTTM_bar.pdf", dpi=300)
    plt.show()


def plotBatchEntropy():
    # Plot for batch entropy
    fig, ax_list = plt.subplots(1, 4, figsize=(8, 3))
    plt.subplots_adjust(wspace=0.05)

    bar1 = ax_list[0].barh(np.arange(n_models), width=HC_batch_entropy[::-1], height=bar_height, align="center", color=color_list[::-1])
    ax_list[1].barh(np.arange(n_models), width=HO_batch_entropy[::-1], height=bar_height, align="center", color=color_list[::-1])
    ax_list[2].barh(np.arange(n_models), width=DR_batch_entropy[::-1], height=bar_height, align="center", color=color_list[::-1])
    ax_list[3].barh(np.arange(n_models), width=MN_batch_entropy[::-1], height=bar_height, align="center", color=color_list[::-1])

    _addBarLabel(ax_list[0], HC_batch_entropy[::-1], thr=0.2, short_offset=0.05, long_offset=0.2)
    _addBarLabel(ax_list[1], HO_batch_entropy[::-1], thr=0.2, short_offset=0.05, long_offset=0.2)
    _addBarLabel(ax_list[2], DR_batch_entropy[::-1], thr=0.2, short_offset=0.05, long_offset=0.2)
    _addBarLabel(ax_list[3], MN_batch_entropy[::-1], thr=0.2, short_offset=0.05, long_offset=0.2)

    ax_list[0].set_title("HC", fontweight="bold")
    ax_list[1].set_title("HO", fontweight="bold")
    ax_list[2].set_title("DR", fontweight="bold")
    ax_list[3].set_title("MN", fontweight="bold")

    ax_list[0].set_yticks(np.arange(n_models), model_list[::-1])
    ax_list[1].set_yticks(np.arange(n_models), ["" for _ in range(n_models)])
    ax_list[2].set_yticks(np.arange(n_models), ["" for _ in range(n_models)])
    ax_list[3].set_yticks(np.arange(n_models), ["" for _ in range(n_models)])
    fig.supxlabel(r"Batch Entropy ($\uparrow$)", y=0.1, fontsize=15)
    for ax in ax_list:
        removeTopRightBorders(ax)
    plt.tight_layout()
    # plt.savefig("../figs/batch_entropy_bar.pdf", dpi=300)
    plt.show()


def plotTimeCorrelation():
    # Plot for time correlation
    fig, ax_list = plt.subplots(1, 4, figsize=(8, 3))
    plt.subplots_adjust(wspace=0.05)

    bar1 = ax_list[0].barh(np.arange(n_models), width=HC_time_correlation[::-1], height=bar_height, align="center", color=color_list[::-1])
    ax_list[1].barh(np.arange(n_models), width=HO_time_correlation[::-1], height=bar_height, align="center", color=color_list[::-1])
    ax_list[2].barh(np.arange(n_models), width=DR_time_correlation[::-1], height=bar_height, align="center", color=color_list[::-1])
    ax_list[3].barh(np.arange(n_models), width=MN_time_correlation[::-1], height=bar_height, align="center", color=color_list[::-1])

    _addBarLabel(ax_list[0], HC_time_correlation[::-1], thr=0.2, short_offset=0.05, long_offset=0.25)
    _addBarLabel(ax_list[1], HO_time_correlation[::-1], thr=0.2, short_offset=0.05, long_offset=0.25)
    _addBarLabel(ax_list[2], DR_time_correlation[::-1], thr=0.2, short_offset=0.05, long_offset=0.25)
    _addBarLabel(ax_list[3], MN_time_correlation[::-1], thr=0.2, short_offset=0.05, long_offset=0.25)

    ax_list[0].set_title("HC", fontweight="bold")
    ax_list[1].set_title("HO", fontweight="bold")
    ax_list[2].set_title("DR", fontweight="bold")
    ax_list[3].set_title("MN", fontweight="bold")

    ax_list[0].set_yticks(np.arange(n_models), model_list[::-1])
    ax_list[1].set_yticks(np.arange(n_models), ["" for _ in range(n_models)])
    ax_list[2].set_yticks(np.arange(n_models), ["" for _ in range(n_models)])
    ax_list[3].set_yticks(np.arange(n_models), ["" for _ in range(n_models)])
    fig.supxlabel(r"Time Correlation ($\uparrow$)", y=0.1, fontsize=15)
    for ax in ax_list:
        removeTopRightBorders(ax)
    plt.tight_layout()
    # plt.savefig("../figs/time_correlation_bar.pdf", dpi=300)
    plt.show()


def plotLTAScatter():
    # Plot for LTA
    fig, ax_list = plt.subplots(1, 2, figsize=(5, 3))
    plt.subplots_adjust(wspace=0.1)

    ax_list[0].scatter(HC_LTA_type, HC_LTA_time, s=ms, c=color_list, edgecolors=white_color)
    ax_list[1].scatter(HO_LTA_type, HO_LTA_time, s=ms, c=color_list, edgecolors=white_color)

    ax_list[0].set_title("HC", fontweight="bold")
    ax_list[1].set_title("HO", fontweight="bold")
    ax_list[0].set_xlim(0.0, 1.0)
    ax_list[0].set_ylim(0.0, 1.0)
    ax_list[1].set_xlim(0.0, 1.0)
    ax_list[1].set_ylim(0.0, 1.0)

    fig.supxlabel(r"LTA-type ($\uparrow$)", y=0.1, fontsize=15)
    ax_list[0].set_ylabel(r"LTA-time ($\uparrow$)", fontsize=15)

    for ax in ax_list:
        removeTopRightBorders(ax)
    plt.tight_layout()
    plt.savefig("../figs/coassay_LTA_scatter.pdf", dpi=300)
    plt.show()
    # -----
    fig, ax_list = plt.subplots(1, 2, figsize=(5, 3))
    plt.subplots_adjust(wspace=0.05)

    ax_list[0].scatter(DR_LTA_type, DR_LTA_time, s=ms, c=color_list, edgecolors=white_color)
    ax_list[1].scatter(MN_LTA_type, MN_LTA_time, s=ms, c=color_list, edgecolors=white_color)

    ax_list[0].set_title("DR", fontweight="bold")
    ax_list[1].set_title("MN", fontweight="bold")
    ax_list[0].set_xlim(0.0, 0.5)
    ax_list[0].set_ylim(0.0, 0.5)
    ax_list[1].set_xlim(0.0, 0.6)
    ax_list[1].set_ylim(0.0, 1.2)

    fig.supxlabel(r"LTA-type ($\uparrow$)", y=0.1, fontsize=15)
    ax_list[0].set_ylabel(r"LTA-time ($\uparrow$)", fontsize=15)

    for ax in ax_list:
        removeTopRightBorders(ax)
    plt.tight_layout()
    # plt.savefig("../figs/unaligned_LTA_scatter.pdf", dpi=300)
    plt.show()


def plotScatterLegned():
    # -----
    plt.figure(figsize=(5, 8))
    for i, m in enumerate(model_list):
        plt.scatter([], [], c=color_list[i], s=ms, label=m)
    removeAllBorders(plt.gca())
    plt.xticks([], [])
    plt.yticks([], [])
    plt.legend(fontsize=20, labelspacing=1.5, handletextpad=0.01)
    plt.tight_layout()
    # plt.savefig("../figs/LTA_scatter_legend.pdf", dpi=300)
    plt.show()


# ===============================

if __name__== '__main__':
    # Fig. 3D (left)
    plotSCC()
    # Fig. 3D (mid)
    plotNeighborOverlap()
    # Fig. 3D (right)
    plotFOSCTTM()
    # Fig. 3A
    plotBatchEntropy()
    # Fig. 3B
    plotTimeCorrelation()
    # Fig. 3C
    plotLTAScatter()
    plotScatterLegned()