'''
Description:
    Visualize integration with PCA.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from plotting.PlottingUtils import onlyPCA
from plotting import *
from utils.FileUtils import loadSCData, tpSplitInd, loadIntegratedLatent
from modal_integration import DATA_DIR_DICT

# ================================================

def _plotF(rna_integrated, atac_integrated, ax1, ax2, ax3, title=""):
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
    latent_umap, _ = onlyPCA(concat_latent_sample, pca_pcs=2)

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


def _plotLegend():
    color_list = Kelly20
    # -----
    mod_list = np.asarray(
        ["rna" for _ in range(all_rna_data.shape[0])] +
        ["atac" for _ in range(all_atac_data.shape[0])]
    )
    cell_type_list = np.concatenate([rna_cell_types, atac_cell_types], axis=0)
    cell_tp_list = np.concatenate([rna_tps, atac_tps], axis=0)
    n_tps = len(np.unique(cell_tp_list))
    # -----
    if data_name == "coassay_cortex":
        cell_type_list[np.where(cell_type_list == "astrocyte")[0]] = "Astrocyte"
        cell_type_list[np.where(cell_type_list == "caudal ganglionic eminence derived interneuron")[0]] = "CGE"
        cell_type_list[np.where(cell_type_list == "endothelial cell")[0]] = "EC"
        cell_type_list[np.where(cell_type_list == "glutamatergic neuron")[0]] = "Gutamatergic"
        cell_type_list[np.where(cell_type_list == "inhibitory interneuron")[0]] = "Inhibitory"
        cell_type_list[np.where(cell_type_list == "medial ganglionic eminence derived interneuron")[0]] = "MGE"
        cell_type_list[np.where(cell_type_list == "microglial cell")[0]] = "Microglial"
        cell_type_list[np.where(cell_type_list == "neural progenitor cell")[0]] = "NPC"
        cell_type_list[np.where(cell_type_list == "oligodendrocyte")[0]] = "Oligodendrocyte"
        cell_type_list[np.where(cell_type_list == "oligodendrocyte precursor cell")[0]] = "OPC"
        cell_type_list[np.where(cell_type_list == "pericyte")[0]] = "Pericyte"
        cell_type_list[np.where(cell_type_list == "radial glial cell")[0]] = "RGC"
        cell_type_list[np.where(cell_type_list == "vascular associated smooth muscle cell")[0]] = "VSMC"
    if data_name == "human_organoid":
        cell_type_list[np.where(cell_type_list == "eb")[0]] = "EB"
        cell_type_list[np.where(cell_type_list == "nect")[0]] = "Neuroectoderm"
        cell_type_list[np.where(cell_type_list == "nepi")[0]] = "Neuroepithelium"
        cell_type_list[np.where(cell_type_list == "organoid")[0]] = "Organoid"
    if data_name == "drosophila":
        pass
    if data_name == "mouse_neocortex":
        for tmp_x in [rna_cell_types, atac_cell_types]:
            tmp_x[np.where(tmp_x == "astro")[0]] = "Astrocytes"
            tmp_x[np.where(tmp_x == "cpn")[0]] = "CPN"
            tmp_x[np.where(tmp_x == "cthpn")[0]] = "CThPN"
            tmp_x[np.where(tmp_x == "inh_cge")[0]] = "IN-CGE"
            tmp_x[np.where(tmp_x == "inh_mge")[0]] = "IN-MGE"
            tmp_x[np.where(tmp_x == "inh_npy")[0]] = "IN-Npy"
            tmp_x[np.where(tmp_x == "inh_sst")[0]] = "IN-Sst"
            tmp_x[np.where(tmp_x == "layer iv")[0]] = "Layer IV"
            tmp_x[np.where(tmp_x == "opc")[0]] = "OPC"
            tmp_x[np.where(tmp_x == "scpn")[0]] = "SCPN"
            tmp_x[np.where(tmp_x == "other")[0]] = "other"
        cell_type_list = np.concatenate([rna_cell_types, atac_cell_types], axis=0)
    # -----
    ms = 50
    title_fontsize = 18
    legend_fontsize = 15

    fig, ax_list = plt.subplots(1, 3, figsize=(15, 10))
    ax1, ax2, ax3 = ax_list

    for i, t in enumerate(range(n_tps)):
        ax1.scatter([], [], label=t, color=color_list[i], s=ms, alpha=1.0)
    ax1.legend(
        loc="center", title="TP", title_fontsize=title_fontsize, fontsize=legend_fontsize,
        ncol=1 if data_name == "mouse_neocortex" else 2,
        columnspacing=0.3, handletextpad=0.1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    removeAllBorders(ax1)

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
        if n in select_cell_typs:
            c = color_list[select_cell_typs.index(n)]
        else:
            c = gray_color
        ax2.scatter([], [], label=n, color=c, s=ms, alpha=1.0)
    ax2.legend(
        loc="center", title="Cell Type", title_fontsize=title_fontsize, fontsize=legend_fontsize,
        ncol=2 if data_name not in ["human_organoid"] else 1,
        columnspacing=0.3, handletextpad=0.1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    removeAllBorders(ax2)

    for i, m in enumerate(["RNA", "ATAC"]):
        ax3.scatter([], [], label=m, color=color_list[i], s=ms, alpha=1.0)
    ax3.legend(loc="center", title="Mod.", title_fontsize=title_fontsize, fontsize=legend_fontsize, handletextpad=0.1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    removeAllBorders(ax3)
    plt.tight_layout()
    plt.savefig(
        "./compare_all_PCA_{}_{}_{}.png".format(data_name, data_type, split_type),
        dpi=600
    )
    plt.show()


def plotAllModelLatent(integrated_dict, model_list, plot_legend=True, plot_pca=True):
    if plot_legend:
        _plotLegend()
    # -----
    if plot_pca:
        fig, ax_list = plt.subplots(3, len(model_list), figsize=(16, 7))
        for m_idx, m in enumerate(model_list):
            ax1, ax2, ax3 = ax_list[0, m_idx], ax_list[1, m_idx], ax_list[2, m_idx]
            rna_integrated, atac_integrated = integrated_dict[m]["rna"], integrated_dict[m]["atac"]
            _plotF(rna_integrated, atac_integrated, ax1, ax2, ax3, title=m)
        for i in range(3):
            for j in range(len(model_list)):
                ax_list[i, j].set_xticks([])
                ax_list[i, j].set_yticks([])
                removeAllBorders(ax_list[i, j])
        plt.tight_layout()
        plt.savefig(
            "./compare_all_PCA_{}_{}_{}-legend.png".format(data_name, data_type, split_type),
            dpi=600
        )
        plt.show()



if __name__ == '__main__':
    # Loading data
    data_name = "coassay_cortex"  # coassay_cortex, human_organoid, drosophila, mouse_neocortex, zebrahub, amphioxus
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
    # Load model integrations and plot PCA (first two principal components)
    latent_dim = 50
    output_dim = latent_dim
    model_list = ["scMultiNODE", "SCOTv2", "SCOTv1", "Pamona", "UnionCom", "uniPort", "Seurat"]
    integrated_dict = loadIntegratedLatent(data_name, data_type, split_type, model_list, latent_dim)
    plotAllModelLatent(integrated_dict, model_list, plot_legend=True, plot_pca=True)
