'''
Description:
    Preprocessing of the human cortex dataset.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import scanpy
import pandas as pd
import numpy as np

# Load data
rna_ann = scanpy.read_h5ad("./raw/10x_scRNA.h5ad")
atac_ann = scanpy.read_h5ad("./raw/10x_scATAC.h5ad")
# # remove non-expressed cells
# scanpy.pp.filter_cells(rna_ann, min_genes=1)
# scanpy.pp.filter_cells(atac_ann, min_genes=1)

print("-" * 70)
print("RNA")
print(rna_ann)
print("-" * 70)
print("ATAC")
print(atac_ann)
print("-" * 70)
print("RNA tps: {}".format(rna_ann.obs.development_stage.unique()))
print("ATAC tps: {}".format(atac_ann.obs.development_stage.unique()))
print("-" * 70)
print("RNA cell types: {}".format(rna_ann.obs.cell_type.unique()))
print("ATAC cell types: {}".format(atac_ann.obs.cell_type.unique()))

# -----
print("-" * 70)
print("Relabel time points...")
tp_map = {
    '18th week post-fertilization human stage':0,
    '19th week post-fertilization human stage':1,
    '23rd week post-fertilization human stage':2,
    '24th week post-fertilization human stage':3,
    'immature stage':4,
    'under-1-year-old human stage':4,
    '4-year-old human stage':5,
    '6-year-old human stage':6,
    '14-year-old human stage':7,
    '20-year-old human stage':8,
    '39-year-old human stage':9,
}
rna_ann.obs["tp"] = rna_ann.obs.development_stage.apply(lambda x: tp_map[x])
atac_ann.obs["tp"] = atac_ann.obs.development_stage.apply(lambda x: tp_map[x])
print("RNA tps: {}".format(rna_ann.obs.tp.unique()))
print("ATAC tps: {}".format(atac_ann.obs.tp.unique()))
# -----
print("-" * 70)
print("Split timepoints...")
split_type = "all"
if split_type == "all":
    rna_train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rna_test_tps = []
    atac_train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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

atac_hvgs_summary = scanpy.pp.highly_variable_genes(scanpy.pp.log1p(atac_train_adata, copy=True), n_top_genes=2000, inplace=False) # cell-by-gene activity matrix
atac_hvgs = atac_train_adata.var.index.values[atac_hvgs_summary.highly_variable]
atac_ann = atac_ann[:, atac_hvgs]

print("RNA HVG data shape: ", rna_ann.shape)
print("ATAC HVG data shape: ", atac_ann.shape)

# -----
print("-" * 70)
print("Save data...")
rna_ann.to_df().to_csv("./processed/{}-RNA-data-hvg.csv".format(split_type))
pd.DataFrame(rna_hvgs).to_csv("./processed/{}-RNA-var_genes_list.csv".format(split_type))
rna_ann.obs.to_csv("./processed/{}-RNA-cell_meta.csv".format(split_type))

atac_ann.to_df().to_csv("./processed/{}-ATAC-data-hvg.csv".format(split_type))
pd.DataFrame(atac_hvgs).to_csv("./processed/{}-ATAC-var_genes_list.csv".format(split_type))
atac_ann.obs.to_csv("./processed/{}-ATAC-cell_meta.csv".format(split_type))