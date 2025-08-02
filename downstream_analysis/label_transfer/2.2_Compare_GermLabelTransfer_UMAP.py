'''
Description:
    Visualize germ layer label transfer with UMAP.

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from plotting.PlottingUtils import umapWithoutPCA
from plotting import removeTopRightBorders as _removeTopRightBorders, removeAllBorders as _removeAllBorders
from plotting import *

# ======================================================

def loadTransferRes(model_name):
    label_res = np.load(
        "./res/germ_layer/drosophila-reduce-all-{}-label_transfer.npy".format(model_name),
        allow_pickle=True
    ).item()
    concat_cell_mod = label_res["concat_cell_mod"]
    concat_germ_layer = label_res["concat_germ_layer"]
    umap_latent_data = label_res["umap_latent_data"]
    concat_cell_tps = label_res["concat_cell_tps"]
    concat_latent_seq = label_res["concat_latent_seq"]
    return concat_cell_mod ,concat_germ_layer, concat_cell_tps, umap_latent_data, concat_latent_seq

# =====================================================

ALL_GERM_LABELS = ['blastoderm / pole', 'ectoderm', 'endoderm', 'extra-embryonic', 'germ cell', 'maternal', 'mesoderm', 'neuroectoderm']
GERM_COLOR_DICT = {g: Kelly20[g_i] for g_i, g in enumerate(ALL_GERM_LABELS)}


def plotLatentGermLayer(latent_dict, model_list):
    latent_list = [latent_dict[m]["latent"][np.where(latent_dict[m]["mod"] == "atac")[0]] for m in model_list]
    cell_label_list = [latent_dict[m]["germ_layer"][np.where(latent_dict[m]["mod"] == "atac")[0]] for m in model_list]
    cell_tps_list = [latent_dict[m]["tps"][np.where(latent_dict[m]["mod"] == "atac")[0]] for m in model_list]
    selected_labels = ['ectoderm', 'endoderm', 'mesoderm', 'neuroectoderm']
    # -----
    color_list = Kelly20
    marker_s = 5
    legned_ms = 20
    marker_alpha = 0.8
    # -----
    for model_i, model_n in enumerate(model_list):
        print("-" * 70)
        print("Model : {}".format(model_n))
        latent_seq = latent_list[model_i]
        cell_labels = cell_label_list[model_i]
        cell_tps = cell_tps_list[model_i]
        select_ind = np.isin(cell_labels, selected_labels)
        latent_seq = latent_seq[select_ind]
        cell_labels = cell_labels[select_ind]
        cell_tps = cell_tps[select_ind]
        # -----
        print("UMAP...")
        n_neighbors = 100
        min_dist = 1.0
        umap_latent, umap_model = umapWithoutPCA(
            latent_seq,
            n_neighbors=n_neighbors, min_dist=min_dist
        )
        print("umap_latent shape: ", umap_latent.shape)
        # -----
        germ_labels = cell_labels
        n_tps = len(np.unique(cell_tps))

        # -----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for i, t in enumerate(range(n_tps)):
            t_idx = np.where(cell_tps == t)[0]
            ax1.scatter(umap_latent[t_idx, 0], umap_latent[t_idx, 1], color=color_list[i], s=marker_s,
                        alpha=marker_alpha)
            ax1.scatter([], [], label=t, color=color_list[i], s=legned_ms)
        ax1.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="TPs", title_fontsize=14, fontsize=13)
        # -----
        unique_germ_layer = np.unique(germ_labels)
        for i, m in enumerate(unique_germ_layer):
            if m == "unknown" or m == "NA":
                continue
            m_idx = np.where(germ_labels == m)[0]
            ax2.scatter(umap_latent[m_idx, 0], umap_latent[m_idx, 1],
                        color=GERM_COLOR_DICT[m] if m != "NA" else gray_color, s=marker_s, alpha=marker_alpha)
            ax2.scatter([], [], label=m, color=GERM_COLOR_DICT[m] if m != "NA" else gray_color, s=legned_ms)
        ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Germ Layer", title_fontsize=14, fontsize=13)
        # -----
        for a in (ax1, ax2):
            _removeAllBorders(a)
            a.set_xticks([], [])
            a.set_yticks([], [])
        plt.tight_layout()
        # plt.savefig(".res//figs/germ_label_ATAC_umap-{}.pdf".format(model_n), dpi=600)
        plt.show()
        plt.close()


if __name__ == '__main__':
    # Loading data
    data_name = "drosophila"
    split_type = "all"
    data_type = "reduce"
    # -----
    model_list = ["scMultiNODE", "Pamona", "SCOTv1", "SCOTv2", "UnionCom", "uniPort", "Seurat"]
    latent_dict = {}
    for model in model_list:
        print("*" * 70)
        print(model)
        concat_cell_mod ,concat_germ_layer, concat_cell_tps, umap_latent_data, concat_latent_seq = loadTransferRes(model_name=model)
        print("concat_cell_mod shape: ", concat_cell_mod.shape)
        print("concat_germ_layer shape: ", concat_germ_layer.shape)
        print("concat_cell_tps shape: ", concat_cell_tps.shape)
        print("umap_latent_data shape: ", umap_latent_data.shape)
        print("concat_latent_seq shape: ", concat_latent_seq.shape)
        latent_dict[model] = {
            "germ_layer": concat_germ_layer,
            "tps": concat_cell_tps,
            "mod": concat_cell_mod,
            "umap": umap_latent_data,
            "latent": concat_latent_seq,
        }
    plotLatentGermLayer(latent_dict, model_list)