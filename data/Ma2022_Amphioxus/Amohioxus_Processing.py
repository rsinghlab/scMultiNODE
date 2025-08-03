'''
Description:
    Preprocessing for zebrahub data.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import scanpy
import pandas as pd
import numpy as np
from scipy.io import mmread
import scipy.sparse
from data.ATAC_HVF import select_var_feature


# Load data
# print("Downsampling...")
# n_cells = 45549
# ratio = 0.25
# rand_idx = np.random.choice(np.arange(n_cells), int(n_cells * ratio), replace=False)
# print(rand_idx.shape)

print("-" * 70)
print("Loading RNA meta...")
rna_meta_df = pd.read_csv("./combined/rna_cell_meta.csv", index_col=0, header=0)
print(rna_meta_df)

print("-" * 70)
print("Loading ATAC meta...")
atac_meta_df =  pd.read_csv("./combined/atac_cell_meta.csv", index_col=0, header=0)
print(atac_meta_df)

print("-" * 70)
print("Loading RNA...")
rna_mat = mmread("./combined/rna_mat.mtx").tocsr().astype("float32")
print(rna_mat.shape)
rna_cell_name = pd.read_csv("./combined/rna_cell_names.csv").cell.values
rna_gene_name = pd.read_csv("./combined/rna_gene_names.csv").gene.values
rna_ann = scanpy.AnnData(X=rna_mat, obs=rna_meta_df) # or use pandas.DataFrame.sparse.from_spmatrix
rna_ann.obs_names = rna_cell_name
rna_ann.var_names = rna_gene_name
print(rna_ann)

print("-" * 70)
print("Loading ATAC...")
atac_mat = mmread("./combined/atac_mat.mtx").tocsr().astype("float32")
print(atac_mat.shape)
atac_cell_name = pd.read_csv("./combined/atac_cell_names.csv").cell.values

unique_values, counts = np.unique(atac_cell_name, return_counts=True)
non_duplicate_values = unique_values[counts == 1]
indices = np.where(np.isin(atac_cell_name, non_duplicate_values))[0]

atac_cell_name = atac_cell_name[indices]
atac_gene_name = pd.read_csv("./combined/atac_gene_names.csv").gene.values
atac_ann = scanpy.AnnData(X=atac_mat[indices, :], obs=atac_meta_df.loc[atac_cell_name, :]) # or use pandas.DataFrame.sparse.from_spmatrix
atac_ann.obs_names = atac_cell_name
atac_ann.var_names = atac_gene_name
print(atac_ann)

print("-" * 70)
print("Subsampling...")
sample_ratio = 0.1
rna_idx = np.random.choice(np.arange(rna_ann.shape[0]), int(rna_ann.shape[0] * sample_ratio), replace=False)
atac_idx = np.random.choice(np.arange(atac_ann.shape[0]), int(atac_ann.shape[0] * sample_ratio), replace=False)
sub_rna_ann = rna_ann[rna_idx, :]
sub_atac_ann = atac_ann[atac_idx, :]
print(sub_rna_ann)
print(sub_atac_ann)

rna_ann = sub_rna_ann
atac_ann = sub_atac_ann

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
    "B":0,
    "G3":1,
    "G6":2,
    "N1":3,
    "N3":4,
    "L0":5,
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