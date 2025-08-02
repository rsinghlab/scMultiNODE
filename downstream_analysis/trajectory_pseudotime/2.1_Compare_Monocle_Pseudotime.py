'''
Description:
    Compare Monocle pseudotime estimation from each method's integration/latent.

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os
import scipy.stats
import numpy as np
import pandas as pd
from plotting import removeTopRightBorders as _removeTopRightBorders, removeAllBorders as _removeAllBorders
from plotting import *
import pprint
from downstream_analysis.trajectory_pseudotime.Utils import MODEL_COLOR

# ======================================================

def loadModelPseudoTime(data_name, model_list):
    save_filename = "./res/data4Monocle/{}-{}-Monocle3_res_df.csv"
    model_res_dict = {}
    for m in model_list:
        m_dict = pd.read_csv(save_filename.format(data_name, m), index_col=None, header=0)
        model_res_dict[m] = m_dict[~np.isinf(m_dict.pseudotime)]
    return model_res_dict

# =======================================================

def comparePseudoTime(model_name_list, model_res_dict, lineage_list=[], lineage_id=""):
    save_filename = "./res/comparison/{}-Monocle-lineage-{}-comparison.npy".format(data_name, lineage_id)
    # -----
    if not os.path.isfile(save_filename):
        print("[ Extract Pseudo-Time ]")
        model_time_dict = {m: {"true": None, "pred": None, "df": None} for m in model_name_list}
        for m in model_name_list:
            model_time_dict[m]["df"] = model_res_dict[m]
            lineage_time = []
            true_time = []
            pseudotime = model_res_dict[m].pseudotime.values
            cell_type_list = model_res_dict[m].cell_types.values
            for c_i, c in enumerate(lineage_list):
                c_idx = np.where(cell_type_list == c)[0]
                lineage_time.append(pseudotime[c_idx])
                true_time.append(np.repeat(c_i, len(c_idx)))
            model_time_dict[m]["true"] = true_time
            model_time_dict[m]["pred"] = lineage_time
        np.save(save_filename, model_time_dict)
    else:
        model_time_dict = np.load(save_filename, allow_pickle=True).item()
    # -----
    # Rank correlation
    print("-" * 70)
    rank_corr_dict = {m: {"spearman": np.nan} for m in model_name_list}
    for m in model_name_list:
        print("*" * 70)
        print("[ Rank correlation ] {}".format(m))
        true_time = np.concatenate(model_time_dict[m]["true"])
        lineage_time = np.concatenate(model_time_dict[m]["pred"])
        print("Spearman rho...")
        spearman_corr = scipy.stats.spearmanr(true_time, lineage_time).correlation
        rank_corr_dict[m]["spearman"] = spearman_corr
    pprint.pprint(rank_corr_dict)
    return rank_corr_dict

# =======================================================

def _addBarLabel(ax, value_list, thr, short_offset, long_offset, fmt="{:.2f}"):
    for index, value in enumerate(value_list):
        if value <= thr:  # Short bar
            ax.text(value + short_offset, index, fmt.format(value), va='center', fontsize=12, fontweight="bold")
        else:  # Long bar
            ax.text(value - long_offset, index, fmt.format(value), va='center', color='white', fontsize=12, fontweight="bold")


def plotPseudoTimeCorr(model_name_list, lineage_metric_dict):
    n_models = len(model_name_list)
    bar_width = 0.95
    line_color = gray_color
    line_width = 0.5
    color_list = [MODEL_COLOR[m] for m in model_name_list]
    fig, ax_list = plt.subplots(1, len(lineage_metric_dict), figsize=(8, 4))
    for l_i, l in enumerate(list(lineage_metric_dict.keys())):
        m_spearman_corr = [lineage_metric_dict[l]["rank_corr_dict"][m]["spearman"] for m in model_name_list]
        # -----
        ax_list[l_i].set_title(lineage_metric_dict[l]["lineage_list"][-1])
        ax_list[l_i].barh(np.arange(n_models), m_spearman_corr[::-1], color=color_list[::-1], height=bar_width, edgecolor=line_color, linewidth=line_width)
        _addBarLabel(ax_list[l_i], m_spearman_corr[::-1], thr=0.25, short_offset=0.125, long_offset=0.25)

    for i in range(len(ax_list)):
        ax_list[i].set_yticks([], [])
    ax_list[0].set_yticks(np.arange(n_models), model_name_list[::-1], fontsize=12)
    ax_list[1].set_xlabel(r"Spearman Corr ($\uparrow$)")
    for i in range(len(ax_list)):
        _removeTopRightBorders(ax_list[i])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_name = "coassay_cortex"
    model_name_list = [
        "scMultiNODE", "SCOTv1", "SCOTv2", "Pamona", "UnionCom", "uniPort", "Seurat",
        "scNODE-RNA", "scNODE-ATAC", "Static_AE-RNA", "Static_AE-ATAC"
    ]
    # =====================================================
    # Loading Monocle results
    print("-" * 70)
    print("Loading Monocle results...")
    model_res_dict = loadModelPseudoTime(data_name, model_name_list)
    # =====================================================
    print("-" * 70)
    print("Plotting lineage pseudotime...")
    save_filename = "./res/comparison/{}-Monocle-metrics.npy".format(data_name)
    lineage_groups = [
        ["rg", "ipc", "en-fetal-early", "en-fetal-late", "en"],
        ["rg", "ipc", "in-fetal", "in-mge"],
        ["rg", "ipc", "in-fetal", "in-cge"],
        ["rg", "opc", "oligodendrocytes"],
    ]
    if not os.path.isfile(save_filename):
        lineage_metric_dict = {}
        for l_i, lineage_list in enumerate(lineage_groups):
            print("=" * 70)
            print("[ Lineage List ] {}".format(lineage_list))
            rank_corr_dict = comparePseudoTime(model_name_list, model_res_dict, lineage_list=lineage_list, lineage_id=lineage_list[-1])
            lineage_metric_dict[l_i] = {
                "lineage_list": lineage_list,
                "rank_corr_dict": rank_corr_dict,
            }
        np.save(save_filename, lineage_metric_dict)
    else:
        lineage_metric_dict = np.load(save_filename, allow_pickle=True).item()
    # -----
    plotPseudoTimeCorr(model_name_list, lineage_metric_dict)
