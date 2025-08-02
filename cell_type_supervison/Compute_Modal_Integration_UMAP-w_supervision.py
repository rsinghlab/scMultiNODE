'''
Description:
    Visualize integration with 2D UMAP.

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from plotting.PlottingUtils import umapWithoutPCA
from plotting import *
from utils.FileUtils import loadSCData, tpSplitInd
from modal_integration import DATA_DIR_DICT
from cell_type_supervison.FileUtils import loadscMultiNODELatent_withSupervision

# ================================================

def _plotF(rna_integrated, atac_integrated, ax1, ax2, ax3, title=""):
    if data_name == "mouse_neocortex":
        umap_neighbors = 50 # default 100
        umap_dist = 0.9  # default 0.5
    elif data_name == "human_organoid":
        umap_neighbors = 100
        umap_dist = 0.5
    elif data_name == "drosophila":
        umap_neighbors = 100
        umap_dist = 0.9
    else:
        umap_neighbors = 100
        umap_dist = 0.5
    marker_s = 10
    marker_alpha = 0.7
    color_list = Kelly20
    # -----
    mod_list = np.asarray(
        ["rna" for _ in range(all_rna_data.shape[0])] + ["atac" for _ in range(all_atac_data.shape[0])])
    cell_type_list = np.concatenate([rna_cell_types, atac_cell_types], axis=0)
    cell_tp_list = np.concatenate([rna_tps, atac_tps], axis=0)
    concat_latent_sample = np.concatenate([rna_integrated, atac_integrated], axis=0)
    n_tps = len(np.unique(cell_tp_list))
    latent_umap, _ = umapWithoutPCA(concat_latent_sample, n_neighbors=umap_neighbors, min_dist=umap_dist)

    # colored by timepoint
    ax1.set_title(title)
    for i, t in enumerate(range(n_tps)):
        t_idx = np.where(cell_tp_list == t)[0]
        ax1.scatter(latent_umap[t_idx, 0], latent_umap[t_idx, 1], label=t, color=color_list[i], s=marker_s,
                    alpha=marker_alpha)

    # colored by cell type
    if len(rna_cell_types) == len(atac_cell_types):
        cell_type_num = [(n, len(np.where(cell_type_list == n)[0])) for n in np.unique(cell_type_list)]
        cell_type_num.sort(reverse=True, key=lambda x: x[1])
        select_cell_typs = [x[0] for x in cell_type_num[:10]]
        cell_type_list = np.asarray([x if x in select_cell_typs else "other" for x in cell_type_list])
    else:
        cell_type_num = [
            (n, len(np.where(cell_type_list == n)[0]))
            for n in np.unique(cell_type_list)
            if n in np.unique(np.intersect1d(rna_cell_types, atac_cell_types))
        ]
        cell_type_num.sort(reverse=True, key=lambda x: x[1])
        select_cell_typs = [x[0] for x in cell_type_num[:10]]
        cell_type_list = np.asarray([x if x in select_cell_typs else "other" for x in cell_type_list])
    for i, n in enumerate(np.unique(cell_type_list)):  #
        n_idx = np.where(cell_type_list == n)[0]
        if n in select_cell_typs:
            c = color_list[select_cell_typs.index(n)]
        else:
            c = gray_color
        ax2.scatter(latent_umap[n_idx, 0], latent_umap[n_idx, 1], label=n.split(" ")[0], color=c, s=marker_s,
                    alpha=marker_alpha if n != "other" else 0.4)

    # colored by modality
    for i, m in enumerate(["rna", "atac"]):
        m_idx = np.where(mod_list == m)[0]
        ax3.scatter(latent_umap[m_idx, 0], latent_umap[m_idx, 1], label=m, color=color_list[i], s=marker_s, alpha=0.25)



def plotAllModelLatent(integrated_dict, model_list, plot_umap=True):
    # -----
    if plot_umap:
        m = model_list[0]
        fig, ax_list = plt.subplots(3, 1, figsize=(5.5, 8))
        ax1, ax2, ax3 = ax_list[0], ax_list[1], ax_list[2]
        rna_integrated, atac_integrated = integrated_dict[m]["rna"], integrated_dict[m]["atac"]
        _plotF(rna_integrated, atac_integrated, ax1, ax2, ax3, title=m)

        for i in range(3):
            ax_list[i].set_xticks([])
            ax_list[i].set_yticks([])
            removeAllBorders(ax_list[i])
            ax_list[i].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        plt.savefig(
            "./supervision_UMAP_{}_{}_{}.png".format(data_name, data_type, split_type),
            dpi=600
        )
        plt.show()
        plt.close()


if __name__ == '__main__':
    # Loading data
    data_name = "zebrahub"  # coassay_cortex, human_organoid, drosophila, mouse_neocortex, zebrahub, amphioxus
    split_type = "all"
    data_type = "reduce"
    # Load processed data to obtain cell type and cell timepoint labels
    (
        ann_rna_data, ann_atac_data, rna_cell_tps, atac_cell_tps,
        rna_n_tps, atac_n_tps, n_genes, n_peaks
    ) = loadSCData(data_name=data_name, data_type=data_type, split_type=split_type, data_dir=DATA_DIR_DICT[data_name])
    rna_train_tps, atac_train_tps, rna_test_tps, atac_test_tps = tpSplitInd(data_name, split_type)
    rna_cnt = ann_rna_data.X
    atac_cnt = ann_atac_data.X
    # cell type
    rna_cell_types = np.asarray([str(x).lower() for x in ann_rna_data.obs["cell_type"].values])
    atac_cell_types = np.asarray([str(x).lower() for x in ann_atac_data.obs["cell_type"].values])
    rna_traj_cell_type = [rna_cell_types[np.where(rna_cell_tps == t)[0]] for t in range(1, rna_n_tps + 1)]
    atac_traj_cell_type = [atac_cell_types[np.where(atac_cell_tps == t)[0]] for t in range(1, atac_n_tps + 1)]
    # Convert to torch project
    rna_traj_data = [rna_cnt[np.where(rna_cell_tps == t)[0], :] for t in
                     range(1, rna_n_tps + 1)]  # (# tps, # cells, # genes)
    atac_traj_data = [atac_cnt[np.where(atac_cell_tps == t)[0], :] for t in
                      range(1, atac_n_tps + 1)]  # (# tps, # cells, # peaks)
    all_rna_data = np.concatenate(rna_traj_data)
    all_atac_data = np.concatenate(atac_traj_data)
    n_tps = len(rna_traj_data)
    all_tps = list(range(n_tps))
    print("RNA shape: ", ann_rna_data.shape)
    print("ATAC shape: ", ann_atac_data.shape)
    print("# genes={}, # peaks={}".format(n_genes, n_peaks))
    # ================================================
    rna_cell_types, atac_cell_types = np.concatenate(rna_traj_cell_type), np.concatenate(atac_traj_cell_type)
    rna_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(rna_traj_data)])
    atac_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(atac_traj_data)])
    # ================================================
    # Load model integrations and plot 2D UMAP
    latent_dim = 50
    output_dim = latent_dim
    integrated_dict = loadscMultiNODELatent_withSupervision(data_name, data_type, split_type, latent_dim)
    plotAllModelLatent(integrated_dict, list(integrated_dict.keys()), plot_umap=True)
