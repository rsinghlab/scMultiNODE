'''
Description:
    Evaluate germ layer label transfer with marker gene expression.

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
from utils.FileUtils import loadSCData, tpSplitInd
from plotting.PlottingUtils import umapWithoutPCA
from plotting import removeTopRightBorders as _removeTopRightBorders, removeAllBorders as _removeAllBorders
from plotting import *

# ======================================================


def loadLabelList(model_name):
    label_res = np.load(
        "./res/germ_layer/drosophila-reduce-all-{}-label_transfer.npy".format(model_name),
        allow_pickle=True
    ).item()
    concat_cell_mod = label_res["concat_cell_mod"]
    concat_germ_layer = label_res["concat_germ_layer"]
    atac_germ_layer = concat_germ_layer[np.where(concat_cell_mod=="atac")[0]]
    # -----
    umap_latent_data = label_res["umap_latent_data"]
    atac_umap_latent_data = umap_latent_data[np.where(concat_cell_mod == "atac")[0]]
    # -----
    concat_cell_tps = label_res["concat_cell_tps"]
    atac_cell_tps = concat_cell_tps[np.where(concat_cell_mod == "atac")[0]]
    return atac_germ_layer ,atac_umap_latent_data, atac_cell_tps

# ======================================================

def findMarkerFromRNA():
    data_name = "drosophila"
    split_type = "all"
    data_type = "reduce"
    # -----
    print("-" * 70)
    save_filename = "./res/germ_layer/{}-rna_marker_genes.csv".format(data_name)
    print("Preparing data for analysis...")
    if not os.path.isfile(save_filename):
        (
            ann_rna_data, ann_atac_data, rna_cell_tps, atac_cell_tps,
            rna_n_tps, atac_n_tps, n_genes, n_peaks
        ) = loadSCData(data_name=data_name, data_type=data_type, split_type=split_type)

        ann_rna_data = ann_rna_data[(ann_rna_data.obs.germ_layer != "NA") & (ann_rna_data.obs.germ_layer != "unknown")]
        scanpy.tl.rank_genes_groups(ann_rna_data, 'germ_layer', method="wilcoxon")  # logreg, wilcoxon
        scanpy.pl.rank_genes_groups(ann_rna_data, n_genes=25, sharey=False, fontsize=12)
        group_id = ann_rna_data.uns["rank_genes_groups"]["names"].dtype.names
        gene_names = ann_rna_data.uns["rank_genes_groups"]["names"]
        print(group_id)
        print(gene_names[:10])
        marker_gene_dict = {}
        for i, g_name in enumerate(group_id):
            g_gene = [x[i] for x in gene_names]
            marker_gene_dict[g_name] = g_gene
        pd.DataFrame(marker_gene_dict).to_csv(save_filename, header=True, index=True)
    else:
        marker_gene_dict = pd.read_csv(save_filename, header=0, index_col=0)
        print(marker_gene_dict)


# ======================================================

def _preprocess(ann_data):
    scanpy.pp.normalize_total(ann_data, target_sum=1e4)
    scanpy.pp.log1p(ann_data)
    return ann_data


# ======================================================

def _plotMarkerGene_Manual(ann_data, ax, colorbar=True):
    marker_genes_dict = {
        "ectoderm": ['mgl'],
        "endoderm": ["grn"],
        "mesoderm": ["Mhc"],
        "neuroectoderm": ["CadN"],
    }
    ann_data = ann_data[np.isin(ann_data.obs.germ_layer, ["ectoderm", "endoderm", "mesoderm", "neuroectoderm"])]
    # -----
    data_germ_layers = np.unique(ann_data.obs["germ_layer"])
    marker_genes_dict = {x: marker_genes_dict[x] for x in data_germ_layers}
    # -----
    desired_order = list(marker_genes_dict.keys())
    ann_data.obs["germ_layer"] = ann_data.obs["germ_layer"].astype("category")
    ann_data.obs["germ_layer"].cat.reorder_categories(desired_order, inplace=True)
    # -----
    germ_layer_list = ["ectoderm", "endoderm", "mesoderm", "neuroectoderm"]
    germ_layer_gene_list = ["mgl", "grn", "Mhc", "CadN"]

    expr = ann_data[:, germ_layer_gene_list].to_df()

    expr['germ_layer'] = ann_data.obs["germ_layer"].astype("category")
    expr["germ_layer"].cat.reorder_categories(desired_order, inplace=True)
    gene_signal_matrix = expr.groupby('germ_layer').mean()
    gene_signal_matrix = gene_signal_matrix.reindex(index=germ_layer_list, columns=germ_layer_gene_list)

    gene_signal_matrix -= gene_signal_matrix.min(0)
    gene_signal_matrix = (gene_signal_matrix / gene_signal_matrix.max(0))

    p = sbn.heatmap(
        gene_signal_matrix,
        cmap='Blues',
        linewidths=0.01,
        linecolor='grey',
        square=True,
        mask=gene_signal_matrix.isna(),
        ax=ax,
        xticklabels=germ_layer_gene_list,
        yticklabels=["Ecto", "Endo", "Meso", "Neuro Ecto"],
        cbar=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    mask = gene_signal_matrix.isna()
    for i in range(gene_signal_matrix.shape[0]):
        for j in range(gene_signal_matrix.shape[1]):
            if mask.iloc[i, j]:
                ax.plot([j, j + 1], [i, i + 1], color='gray', linewidth=1.0)
                ax.plot([j, j + 1], [i + 1, i], color='gray', linewidth=1.0)
    return p



def plotGeneSignal(
        gene_activity, cell_labels, cell_tps, cell_names, atac_umap_latent, model_name
):
    print("-" * 70)
    print("Constructing ann data...")
    cell_meta = pd.DataFrame({"germ_layer": cell_labels, "tp": cell_tps, "id": cell_names})
    cell_meta = cell_meta.set_index("id")
    gene_act_ann = scanpy.AnnData(gene_activity, obs=cell_meta)
    print(gene_act_ann)

    # -----
    print("-" * 70)
    print("Preprocessing...")
    gene_act_ann = _preprocess(gene_act_ann)
    print(gene_act_ann)
    # -----
    print("-" * 70)
    print("Marker genes...")

    fig, ax_list = plt.subplots(1, 1, figsize=(6, 6), sharey=True)
    plt.title(model_name)
    mp = _plotMarkerGene_Manual(gene_act_ann, ax_list, colorbar=True)
    plt.yticks(rotation=0)
    plt.tight_layout()
    # plt.savefig("./res/figs/germ_marker-{}.pdf".format(model_name), dpi=600)
    plt.show()
    plt.close()


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
    # # -----
    # # Option: find marker genes from scRNA-seq data
    # findMarkerFromRNA()
    # -----
    print("-" * 70)
    save_filename = "./res/germ_layer/{}-reorder_gene_activity.csv".format(data_name)
    reorder_gene_activity = pd.read_csv(save_filename, header=0, index_col=0)
    print("reorder_gene_activity shape: ", reorder_gene_activity.shape)
    atac_cell_name_list = reorder_gene_activity.index.values
    # -----
    for model in ["scMultiNODE", "Pamona", "SCOTv1", "SCOTv2", "UnionCom", "uniPort", "Seurat"]:
        print("*" * 70)
        print(model)
        atac_germ_layer, atac_umap_latent_data, atac_cell_tps = loadLabelList(model_name=model)
        print("atac_germ_layer shape: ", atac_germ_layer.shape)
        print("atac_umap_latent_data shape: ", atac_umap_latent_data.shape)
        print("atac_cell_tps shape: ", atac_cell_tps.shape)
        # -----
        plotGeneSignal(
            reorder_gene_activity, atac_germ_layer, atac_cell_tps, atac_cell_name_list, atac_umap_latent_data, model
        )