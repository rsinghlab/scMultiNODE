'''
Description:
    Preprocessing of the mouse neocortex dataset.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import scanpy
import pandas as pd
import numpy as np
from scipy.io import mmread
from scipy.sparse import csc_matrix
from data.ATAC_HVF import select_var_feature


# Load data
print("Loading RNA data...")
# rna_cnt = pd.read_csv("./raw/sample_rna_cnt.csv", header=0, index_col=0)
rna_cnt = mmread("./raw/sample_rna_cnt.mtx")
rna_meta = pd.read_csv("./raw/sample_rna_meta.csv", header=0, index_col=0)

print("Loading ATAC data...")
# atac_cnt = pd.read_csv("./raw/sample_atac_cnt.csv", header=0, index_col=0)
atac_cnt = mmread("./raw/sample_atac_cnt.mtx")
atac_meta = pd.read_csv("./raw/sample_atac_meta.csv", header=0, index_col=0)
atac_meta["CellType"] = atac_meta["cellType"]

print("Constructing AnnData...")
rna_ann = scanpy.AnnData(X=csc_matrix(rna_cnt), obs=rna_meta)
atac_ann = scanpy.AnnData(X=csc_matrix(atac_cnt), obs=atac_meta)


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
print("RNA tps: {}".format(rna_ann.obs.timePoint.unique()))
print("ATAC tps: {}".format(atac_ann.obs.timePoint.unique()))
print("-" * 70)
print("RNA cell types: {}".format(rna_ann.obs.CellType.unique()))
print("ATAC cell types: {}".format(atac_ann.obs.CellType.unique()))

# -----
print("-" * 70)
print("Relabel time points...")
tp_map = {
    'P1':0,
    'P7':1,
    'P21':2,
}
rna_ann.obs["tp"] = rna_ann.obs.timePoint.apply(lambda x: tp_map[x])
atac_ann.obs["tp"] = atac_ann.obs.timePoint.apply(lambda x: tp_map[x])
print("RNA tps: {}".format(rna_ann.obs.tp.unique()))
print("ATAC tps: {}".format(atac_ann.obs.tp.unique()))
# -----
print("-" * 70)
print("Split timepoints...")
split_type = "all" # all
if split_type == "all":
    rna_train_tps = [0, 1, 2]
    rna_test_tps = []
    atac_train_tps = [0, 1, 2]
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