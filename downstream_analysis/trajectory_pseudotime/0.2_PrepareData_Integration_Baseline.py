'''
Description:
    Prepare data for pseudotime estimation with Monocle3 or PAGA.

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os
import scipy.interpolate
import scipy.stats
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
from plotting import *
from plotting.PlottingUtils import umapWithoutPCA
from utils.FileUtils import loadSCData, tpSplitInd

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# ======================================================
# Load integrations
# ======================================================

def loadModelLatent(data_name, data_type, split_type, latent_type, model_name, latent_dim, output_dim):
    save_filename = "../../modal_integration/res/model_latent/{}-{}-{}-{}-{}dim.npy"
    seurat_save_filename = "../../modal_integration/res/model_latent/{}-{}-{}-{}-{}dim-{}.csv"
    if model_name == "Seurat":
        rna_integrated = pd.read_csv(
            seurat_save_filename.format(
                data_name, data_type, split_type, model_name, latent_dim, output_dim,"rna"
            ), header=0, index_col=None).values
        atac_integrated = pd.read_csv(
            seurat_save_filename.format(
                data_name, data_type, split_type, model_name, latent_dim, output_dim,"atac"
            ), header=0, index_col=None).values
    else:
        res = np.load(
            save_filename.format(latent_type, data_name, data_type, split_type, model_name, latent_dim, output_dim),
            allow_pickle=True).item()
        rna_integrated = res["rna_integrated"]
        atac_integrated = res["atac_integrated"]
    return rna_integrated, atac_integrated


if __name__ == '__main__':
    # Loading data
    data_name = "coassay_cortex"
    split_type = "all"
    data_type = "reduce"
    data_dir = "../../data/human_prefrontal_cortex_multiomic/reduce_processed/"
    (
        ann_rna_data, ann_atac_data, rna_cell_tps, atac_cell_tps,
        rna_n_tps, atac_n_tps, n_genes, n_peaks
    ) = loadSCData(data_name=data_name, data_type=data_type, split_type=split_type, data_dir=data_dir)
    rna_train_tps, atac_train_tps, rna_test_tps, atac_test_tps = tpSplitInd(data_name, split_type)
    rna_cnt = ann_rna_data.X
    atac_cnt = ann_atac_data.X
    # cell type
    rna_cell_types = np.asarray([x.lower() for x in ann_rna_data.obs["author_cell_type"].values])
    atac_cell_types = np.asarray([x.lower() for x in ann_atac_data.obs["author_cell_type"].values])
    rna_traj_cell_type = [rna_cell_types[np.where(rna_cell_tps == t)[0]] for t in range(1, rna_n_tps + 1)]
    atac_traj_cell_type = [atac_cell_types[np.where(atac_cell_tps == t)[0]] for t in range(1, atac_n_tps + 1)]
    # Convert to torch project
    rna_traj_data = [rna_cnt[np.where(rna_cell_tps == t)[0], :] for t in
                     range(1, rna_n_tps + 1)]  # (# tps, # cells, # genes)
    atac_traj_data = [atac_cnt[np.where(atac_cell_tps == t)[0], :] for t in
                      range(1, atac_n_tps + 1)]  # (# tps, # cells, # peaks)
    all_rna_data = np.concatenate(rna_traj_data)
    all_atac_data = np.concatenate(atac_traj_data)
    n_tps = len(rna_traj_data)
    all_tps = list(range(n_tps))
    print("RNA shape: ", ann_rna_data.shape)
    print("ATAC shape: ", ann_atac_data.shape)
    print("# genes={}, # peaks={}".format(n_genes, n_peaks))
    # -----

    model_list = ["SCOTv1", "SCOTv2", "Pamona", "UnionCom", "uniPort", "Seurat"]
    for m in model_list:
        print("-" * 70)
        save_filename = "./res/aux_data/{}-reduce-all-{}-aux_data.npy".format(data_name, m)
        print("Preparing data for analysis...")
        n_neighbors = 100
        min_dist = 0.5
        if not os.path.isfile(save_filename):
            rna_integrate, atac_integrate = loadModelLatent(data_name, data_type, split_type, "ae", m, 50, 50)
            rna_cell_types = np.concatenate(rna_traj_cell_type)
            atac_cell_types = np.concatenate(atac_traj_cell_type)
            rna_cell_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(rna_traj_data)])
            atac_cell_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(atac_traj_data)])
            print("-" * 70)
            print("UMAP...")
            umap_latent_data, umap_model = umapWithoutPCA(
                np.concatenate([rna_integrate, atac_integrate], axis=0),
                n_neighbors=n_neighbors, min_dist=min_dist
            )
            np.save(save_filename, {
                "rna_integrate": rna_integrate,
                "atac_integrate": atac_integrate,
                "rna_cell_types": rna_cell_types,
                "atac_cell_types": atac_cell_types,
                "rna_cell_tps": rna_cell_tps,
                "atac_cell_tps": atac_cell_tps,
                "umap_latent_data": umap_latent_data,
                "umap_model": umap_model,
            })
        else:
            data_res = np.load(save_filename, allow_pickle=True).item()
            rna_integrate = data_res["rna_integrate"]
            atac_integrate = data_res["atac_integrate"]
            rna_cell_types = data_res["rna_cell_types"]
            atac_cell_types = data_res["atac_cell_types"]
            rna_cell_tps = data_res["rna_cell_tps"]
            atac_cell_tps = data_res["atac_cell_tps"]
            umap_latent_data = data_res["umap_latent_data"]
            umap_model = data_res["umap_model"]  # Note: The UMAP is for the model latent space
        # -----
        concat_cell_types = np.concatenate([rna_cell_types, atac_cell_types])
        concat_cell_tps = np.concatenate([rna_cell_tps, atac_cell_tps])
        concat_cell_mod = np.concatenate([
            ["rna" for _ in range(rna_integrate.shape[0])],
            ["atac" for _ in range(rna_integrate.shape[0])]
        ])
        concat_latent_seq = np.concatenate([rna_integrate, atac_integrate], axis=0)
        print(concat_cell_types.shape)
        print(concat_cell_tps.shape)
        print(concat_cell_mod.shape)
        print(concat_latent_seq.shape)
        # -----
        print("Saving data for Monocle...")
        df_savefile = "./res/data4Monocle/{}-{}-concat_meta_df.csv".format(data_name, m)
        if not os.path.isfile(df_savefile):
            obs_df = pd.DataFrame({
                "tps": concat_cell_tps,
                "cell_types": concat_cell_types,
                "modality": concat_cell_mod
            })
            print(obs_df)
            obs_df.to_csv(df_savefile)
            np.savetxt("./res/data4Monocle/{}-{}-concat_integrate.csv".format(data_name, m), concat_latent_seq, delimiter=",")
            np.savetxt("./res/data4Monocle/{}-{}-umap_latent_data.csv".format(data_name, m), umap_latent_data, delimiter=",")
        else:
            print("Files already exist in the ./res/data4Monocle directory.")


