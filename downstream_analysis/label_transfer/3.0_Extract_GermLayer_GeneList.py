'''
Description:
    Preparation for GO enrichment analysis: extract 100 highly variable genes for each predicted germ layer group.
    Data will be saved to ./res/GO, if they do not exist therein.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import scanpy
import numpy as np
import pandas as pd

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

def _preprocess(ann_data):
    scanpy.pp.normalize_total(ann_data, target_sum=1e4)
    scanpy.pp.log1p(ann_data)
    return ann_data


def buildGOGeneFile(gene_activity, cell_labels, cell_tps, cell_names, model_name):
    print("-" * 70)
    print("Constructing ann data...")
    cell_meta = pd.DataFrame({"germ_layer": cell_labels, "tp": cell_tps, "id": cell_names})
    cell_meta = cell_meta.set_index("id")
    gene_act_ann = scanpy.AnnData(gene_activity, obs=cell_meta)
    selected_labels = ['ectoderm', 'endoderm', 'mesoderm', 'neuroectoderm']
    gene_act_ann = gene_act_ann[np.isin(gene_act_ann.obs.germ_layer, selected_labels)]
    print(gene_act_ann)
    # -----
    print("-" * 70)
    print("Preprocessing...")
    gene_act_ann = _preprocess(gene_act_ann)
    print(gene_act_ann)
    print("-" * 70)
    # save_filename = "./germ_layer/{}-pred_atac_marker_genes-{}.csv".format(data_name, model_name)
    scanpy.tl.rank_genes_groups(gene_act_ann, 'germ_layer', method="wilcoxon")  # logreg, wilcoxon
    scanpy.pl.rank_genes_groups(gene_act_ann, n_genes=25, sharey=False, fontsize=12)
    group_id = gene_act_ann.uns["rank_genes_groups"]["names"].dtype.names
    gene_names = gene_act_ann.uns["rank_genes_groups"]["names"]
    print(group_id)
    print(gene_names[:10])
    marker_gene_dict = {}
    n_genes = 100
    for i, g_name in enumerate(group_id):
        g_gene = [x[i] for x in gene_names]
        marker_gene_dict[g_name] = g_gene[:n_genes]
    # -----
    gene_list = []
    label_list = []
    for c in marker_gene_dict:
        g = marker_gene_dict[c]
        gene_list.append(g)
        label_list.append(np.repeat(c, len(g)))
    df = pd.DataFrame({"gene": np.concatenate(gene_list), "label": np.concatenate(label_list)})
    df.to_csv("./res/GO/all-genes.csv".format(model_name))



if __name__ == '__main__':
    # Loading data
    data_name = "drosophila"
    split_type = "all"
    data_type = "reduce"
    model_name = "scMultiNODE"
    # -----
    print("-" * 70)
    save_filename = "./res/germ_layer/{}-reorder_gene_activity.csv".format(data_name)
    reorder_gene_activity = pd.read_csv(save_filename, header=0, index_col=0)
    print("reorder_gene_activity shape: ", reorder_gene_activity.shape)
    atac_cell_name_list = reorder_gene_activity.index.values
    # -----
    atac_germ_layer, atac_umap_latent_data, atac_cell_tps = loadLabelList(model_name)
    print("atac_germ_layer shape: ", atac_germ_layer.shape)
    print("atac_umap_latent_data shape: ", atac_umap_latent_data.shape)
    print("atac_cell_tps shape: ", atac_cell_tps.shape)
    # -----
    buildGOGeneFile(reorder_gene_activity, atac_germ_layer, atac_cell_tps, atac_cell_name_list, model_name)

