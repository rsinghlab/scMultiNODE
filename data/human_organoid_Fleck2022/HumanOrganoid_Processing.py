'''
Description:
    Preprocessing of the human organoid dataset.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import scanpy
import pandas as pd
import numpy as np
from scipy.io import mmread


# Load data
print("-" * 70)
print("Loading meta...")
meta_df = pd.read_csv("./raw/sample_meta_data.csv")
meta_df["cell_type"] = meta_df.stage
meta_df = meta_df.set_index("Unnamed: 0")
print(meta_df)

print("-" * 70)
print("Loading RNA...")
rna_mat = mmread("./raw/sample_rna_data_count.mtx").todense()
rna_cell_name = pd.read_csv("./raw/sample_rna_cell_name").x.values
rna_gene_name = pd.read_csv("./raw/sample_rna_gene_name").x.values
rna_ann = scanpy.AnnData(X=pd.DataFrame(data=rna_mat.T, index=rna_cell_name, columns=rna_gene_name), obs=meta_df)
print(rna_ann)

print("-" * 70)
print("Loading ATAC...")
atac_mat = mmread("./raw/sample_atac_data_activity.mtx").todense()
atac_cell_name = pd.read_csv("./raw/sample_atac_cell_name").x.values
atac_gene_name = pd.read_csv("./raw/sample_atac_gene_name").x.values
atac_ann = scanpy.AnnData(X=pd.DataFrame(data=atac_mat.T, index=atac_cell_name, columns=atac_gene_name), obs=meta_df)
print(atac_ann)

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
    4:0,
    7:1,
    9:2,
    11:3,
    12:4,
    16:5,
    18:6,
    21:7,
    26:8,
    31:9,
    61:10,
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
    rna_train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rna_test_tps = []
    atac_train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
rna_ann.to_df().to_csv("./reduce_processed/{}-RNA-data-hvg.csv".format(split_type))
pd.DataFrame(rna_hvgs).to_csv("./reduce_processed/{}-RNA-var_genes_list.csv".format(split_type))
rna_ann.obs.to_csv("./reduce_processed/{}-RNA-cell_meta.csv".format(split_type))

atac_ann.to_df().to_csv("./reduce_processed/{}-ATAC-data-hvg.csv".format(split_type))
pd.DataFrame(atac_hvgs).to_csv("./reduce_processed/{}-ATAC-var_genes_list.csv".format(split_type))
atac_ann.obs.to_csv("./reduce_processed/{}-ATAC-cell_meta.csv".format(split_type))