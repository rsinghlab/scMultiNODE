import scanpy
import pandas as pd
import numpy as np
import anndata


split_type = "hard" # all, easy, medium, hard
rna_ann = anndata.AnnData(
    X=pd.read_csv("./processed/{}-RNA-data-hvg.csv".format(split_type), header=0, index_col=0),
    obs=pd.read_csv("./processed/{}-RNA-cell_meta.csv".format(split_type), header=0, index_col=0)
)
atac_ann = anndata.AnnData(
    X=pd.read_csv("./processed/{}-ATAC-data-hvg.csv".format(split_type), header=0, index_col=0),
    obs=pd.read_csv("./processed/{}-ATAC-cell_meta.csv".format(split_type), header=0, index_col=0)
)

sampling_ratio = 0.05
cell_idx = np.random.choice(np.arange(rna_ann.shape[0]), int(rna_ann.shape[0]*sampling_ratio), replace=False)
rna_ann = rna_ann[cell_idx, :]
atac_ann = atac_ann[cell_idx, :]

print("RNA subsample data shape: ", rna_ann.shape)
print("ATAC subsample data shape: ", atac_ann.shape)

print("Save data...")
rna_ann.to_df().to_csv("./reduce_processed/{}-RNA-data-hvg.csv".format(split_type))
rna_ann.obs.to_csv("./reduce_processed/{}-RNA-cell_meta.csv".format(split_type))

atac_ann.to_df().to_csv("./reduce_processed/{}-ATAC-data-hvg.csv".format(split_type))
atac_ann.obs.to_csv("./reduce_processed/{}-ATAC-cell_meta.csv".format(split_type))