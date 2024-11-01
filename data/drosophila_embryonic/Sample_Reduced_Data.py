'''
Description:
    Sampling cells in the DR dataset.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
import scanpy
import pandas as pd
import natsort


def loadDrosophilaData(data_dir, split_type):
    # Load RNA data
    cnt_data = pd.read_csv("{}/{}-RNA_count_data.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/subsample_RNA_meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["time"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_rna_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    # Add cell labels
    cell_meta = pd.read_csv("{}/rna_meta.csv".format(data_dir), header=0, index_col=0)
    cell_meta = cell_meta.set_index("cell")
    cell_meta = cell_meta.loc[ann_rna_data.obs_names.values, :]
    ann_rna_data.obs = pd.concat([ann_rna_data.obs, cell_meta[["manual_annot", "germ_layer"]]], axis=1)
    ann_rna_data.obs["cell_type"] = ann_rna_data.obs["manual_annot"]
    # -----
    # Load ATAC data
    cnt_data = pd.read_csv("{}/{}-ATAC_count_data.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/subsample_ATAC_meta_data.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.loc[cnt_data.index, :]
    cell_stage = meta_data["time"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage),))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_atac_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    # Add cell labels
    cell_meta = pd.read_csv("{}/atac_meta.csv".format(data_dir), header=0, index_col=0)
    cell_meta = cell_meta.set_index("cell")
    cell_meta = cell_meta.loc[ann_atac_data.obs_names.values, :]
    ann_atac_data.obs = pd.concat([ann_atac_data.obs, cell_meta[["seurat_clusters", "refined_annotation"]]], axis=1)
    ann_atac_data.obs["cell_type"] = ann_atac_data.obs["refined_annotation"]
    return ann_rna_data, ann_atac_data


if __name__ == '__main__':
    drosophila_data_dir = "./processed/"
    split_type = "hard" # all, easy, medium, hard
    rna_ann, atac_ann = loadDrosophilaData(drosophila_data_dir, split_type)
    sampling_ratio = 0.05

    rna_ann = rna_ann[np.random.choice(np.arange(rna_ann.shape[0]), int(rna_ann.shape[0] * sampling_ratio), replace=False), :]
    atac_ann = atac_ann[np.random.choice(np.arange(atac_ann.shape[0]), int(atac_ann.shape[0] * sampling_ratio), replace=False),:]

    print("-" * 70)
    print("RNA")
    print(rna_ann)
    print("-" * 70)
    print("ATAC")
    print(atac_ann)

    rna_ann.to_df().to_csv("reduce_processed/{}-RNA_count_data.csv".format(split_type))
    rna_ann.obs.to_csv("reduce_processed/{}_RNA_meta_data.csv".format(split_type))

    atac_ann.to_df().to_csv("reduce_processed/{}-ATAC_count_data.csv".format(split_type))
    atac_ann.obs.to_csv("reduce_processed/{}_ATAC_meta_data.csv".format(split_type))
