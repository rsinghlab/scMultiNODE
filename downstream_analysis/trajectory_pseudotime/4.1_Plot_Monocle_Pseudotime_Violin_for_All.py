'''
Description:
    Visualize Monocle pseudotime estimation with violin plots.

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
import pandas as pd
from plotting import removeTopRightBorders as _removeTopRightBorders, removeAllBorders as _removeAllBorders
from plotting import *


# =====================================================

def plotLineageTime_Monocle(data_name, res_df, lineage_list, lineage_name, model_name):
    lw = 2.5
    color_list = Kelly20[:len(lineage_list)]
    fig, ax_list = plt.subplots(1, 1, figsize=(6.5, 2))
    pseudotime = res_df.pseudotime.values
    pseudotime[np.isinf(pseudotime)] = np.nan
    cell_type_list = res_df.cell_types.values
    unique_cell_type = np.unique(cell_type_list)
    # -----
    lineage_time = []
    for c in lineage_list:
        c_idx = np.where((cell_type_list == c) & (~np.isnan(pseudotime)))[0]
        lineage_time.append(pseudotime[c_idx])
    # -----
    v = ax_list.violinplot(lineage_time, positions=np.arange(len(lineage_list)), showmeans=False, showmedians=True)
    for i, pc in enumerate(v['bodies']):
        pc.set_facecolor(color_list[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

    if 'cmedians' in v:
        v['cmedians'].set_color('red')
        v['cmedians'].set_linewidth(lw)

    for i, x in enumerate(lineage_time):
        median_val = np.median(x)
        ax_list.text(i, np.max(x) + 0.5, "med={:.2f}".format(median_val), ha='center', va='bottom', fontsize=12, color='red')

    ax_list.set_xticks(np.arange(len(lineage_list)), lineage_name, fontsize=13)
    ax_list.set_ylabel("Pseudotime")
    _removeTopRightBorders(ax_list)
    plt.tight_layout()
    plt.savefig("./res/figs/Monocle_pseudotime_each/lineage_pseudotime-{}-{}.pdf".format(model_name, lineage_name[-1]), dpi=600)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    data_name = "coassay_cortex"
    print("-" * 70)
    print("Loading Monocle results...")
    print("-" * 70)
    print("Plotting lineage pseudotime...")
    lineage_list = [
        ["rg", "ipc", "en-fetal-early", "en-fetal-late", "en"],
        ["rg", "ipc", "in-fetal", "in-mge"],
        ["rg", "ipc", "in-fetal", "in-cge"],
        ["rg", "opc", "oligodendrocytes"]
    ]
    lineage_name = [
        ["RG", "IPC", "EN-fetal-early", "EN-fetal-late", "EN"],
        ["RG", "IPC", "IN-fetal", "IN-mge"],
        ["RG", "IPC", "IN-fetal", "IN-cge"],
        ["RG", "OPC", "Oligo."],
    ]

    for i in range(4):
        print("=" * 70)
        print("Lineage {}".format(i))
        l_l = lineage_list[i]
        l_n = lineage_name[i]
        for m in [
            "scMultiNODE", "scNODE-RNA", "scNODE-ATAC", "SCOTv1", "SCOTv2", "Pamona", "UnionCom",
            "uniPort", "Seurat", "Static_AE-RNA", "Static_AE-ATAC"
        ]:
            print("*" * 70)
            print("Model: {}".format(m))
            res_df = pd.read_csv("./res/data4Monocle/{}-{}-Monocle3_res_df.csv".format(data_name, m),
                                 index_col=None, header=0)
            plotLineageTime_Monocle(data_name, res_df, l_l, l_n, model_name=m)


