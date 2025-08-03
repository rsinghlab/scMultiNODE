'''
Description:
    Preprocessing for zebrahub data.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>

Reference:
    [1] Raw data link: https://drive.google.com/drive/folders/1SIkI6FAOtNp3hCeJYxfi19EgnXCDCgOu
'''

import scanpy
import pandas as pd
import numpy as np
from scipy.io import mmread
import scipy.sparse
from data.ATAC_HVF import select_var_feature


# Load data
print("-" * 70)
print("Loading ATAC...")
atac_ann = scanpy.read_h5ad("./raw/sub_atac_ann.h5ad")
atac_ann.obs["cell_type"] = atac_ann.obs["zebrafish_anatomy_ontology_class"]
atac_ann.obs["age"] = atac_ann.obs["timepoint"]
print(atac_ann)

print("-" * 70)
print("Loading RNA...")
rna_ann = scanpy.read_h5ad("./raw/sub_rna_ann.h5ad")
rna_ann.obs["cell_type"] = rna_ann.obs["zebrafish_anatomy_ontology_class"]
rna_ann.obs["age"] = rna_ann.obs["timepoint"]
print(rna_ann)

print("-" * 70)
print("RNA tps: {}".format(rna_ann.obs.age.unique()))
print("ATAC tps: {}".format(atac_ann.obs.age.unique()))

print("-" * 70)
print("RNA cell types: {}".format(rna_ann.obs.cell_type.unique()))
print("ATAC cell types: {}".format(atac_ann.obs.cell_type.unique()))

# -----
print("-" * 70)
print("Relabel time points...")
tp_map = {
    "10hpf":0,
    "12hpf":1,
    "14hpf":2,
    "16hpf":3,
    "19hpf":4,
    "24hpf":5,
}
rna_ann.obs["tp"] = rna_ann.obs.age.apply(lambda x: tp_map[x])
atac_ann.obs["tp"] = atac_ann.obs.age.apply(lambda x: tp_map[x])
print("RNA tps: {}".format(rna_ann.obs.tp.unique()))
print("ATAC tps: {}".format(atac_ann.obs.tp.unique()))
# -----
print("-" * 70)
print("Split timepoints...")
split_type = "all"
if split_type == "all":
    rna_train_tps = [0, 1, 2, 3, 4, 5]
    rna_test_tps = []
    atac_train_tps = [0, 1, 2, 3, 4, 5]
    atac_test_tps = []
print("RNA Train tps: ", rna_train_tps)
print("RNA Test tps: ", rna_test_tps)
print("ATAC Train tps: ", atac_train_tps)
print("ATAC Test tps: ", atac_test_tps)
# -----
print("-" * 70)
print("Pre-processing...")
rna_train_adata = rna_ann[np.where(rna_ann.obs['tp'].apply(lambda x: x in rna_train_tps))]
atac_train_adata = atac_ann[np.where(atac_ann.obs['tp'].apply(lambda x: x in atac_train_tps))]
print("RNA train data shape: ", rna_train_adata.shape)
print("ATAC train data shape: ", atac_train_adata.shape)

rna_hvgs_summary = scanpy.pp.highly_variable_genes(scanpy.pp.log1p(rna_train_adata, copy=True), n_top_genes=2000, inplace=False)
rna_hvgs = rna_train_adata.var.index.values[rna_hvgs_summary.highly_variable]
rna_ann = rna_ann[:, rna_hvgs]

atac_ann = select_var_feature(atac_train_adata, nb_features=2000, copy=True)
atac_hvgs = atac_ann.var.index.values

print("RNA HVG data shape: ", rna_ann.shape)
print("ATAC HVG data shape: ", atac_ann.shape)

# -----
print("-" * 70)
print("Save data...")
rna_ann.to_df().to_csv("./reduce_processed/{}-RNA-data-hvg.csv".format(split_type))
pd.DataFrame(rna_hvgs).to_csv("./reduce_processed/{}-RNA-var_genes_list.csv".format(split_type))
rna_ann.obs.to_csv("./reduce_processed/{}-RNA-cell_meta.csv".format(split_type))

atac_ann.to_df().to_csv("./reduce_processed/{}-ATAC-data-hvg.csv".format(split_type))
pd.DataFrame(atac_hvgs).to_csv("./reduce_processed/{}-ATAC-var_genes_list.csv".format(split_type))
atac_ann.obs.to_csv("./reduce_processed/{}-ATAC-cell_meta.csv".format(split_type))