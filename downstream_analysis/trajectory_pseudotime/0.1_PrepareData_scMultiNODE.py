'''
Description:
    Prepare scMultiNODE integration for pseudotime estimation with Monocle3 or PAGA.
    Data will be saved to ./res/aux_data/ and ./res/data4Monocle, if they do not exist therein.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os
import torch
import numpy as np
import pandas as pd
from utils.FileUtils import loadSCData, tpSplitInd
from optim.running import constructscMultiNODEModel
from plotting.PlottingUtils import umapWithoutPCA

# ======================================================
# Load Integrations
# ======================================================

def loadModel():
    dict_filename = "./res/trained_model/coassay_cortex-reduce-all-scMultiNODE-50dim-state_dict.pt"
    n_genes = 2000
    n_peaks = 2000
    latent_dim=50
    # Construct scMulti model
    anchor_mod = "rna"
    rna_enc_latent = [50]
    rna_dec_latent = [50]
    atac_enc_latent = [50]
    atac_dec_latent = [50]
    fusion_latent = [50]
    drift_latent = [50]
    dynamic_model = constructscMultiNODEModel(
        n_genes, n_peaks, latent_dim, anchor_mod,
        rna_enc_latent=rna_enc_latent, rna_dec_latent=rna_dec_latent,
        atac_enc_latent=atac_enc_latent, atac_dec_latent=atac_dec_latent,
        fusion_latent=fusion_latent, drift_latent=drift_latent,
        act_name="relu", ode_method="euler"
    )
    dynamic_model.load_state_dict(torch.load(dict_filename))
    dynamic_model.eval()
    return dynamic_model


def loadLatent():
    res = np.load(
        "./res/trained_model/coassay_cortex-reduce-all-scMultiNODE-50dim.npy",
        allow_pickle=True
    ).item()
    rna_integrate = res["rna_integrated"]
    atac_integrate = res["atac_integrated"]
    return rna_integrate, atac_integrate


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
    rna_cell_types = np.asarray([x.lower() for x in ann_rna_data.obs["cell_type"].values])
    atac_cell_types = np.asarray([x.lower() for x in ann_atac_data.obs["cell_type"].values])
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
    print("=" * 70)
    print("Loading model...")
    dynamic_model = loadModel()
    print(dynamic_model)
    # -----
    print("-" * 70)
    save_filename = "./res/aux_data/coassay_cortex-reduce-all-scMultiNODE-aux_data.npy"
    print("Preparing data for analysis...")
    if not os.path.isfile(save_filename):
        # Load scMultiNODE integration
        rna_integrate, atac_integrate = loadLatent()
        rna_cell_types = np.concatenate(rna_traj_cell_type)
        atac_cell_types = np.concatenate(atac_traj_cell_type)
        rna_cell_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(rna_traj_data)])
        atac_cell_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(atac_traj_data)])
        print("-" * 70)
        print("UMAP...")
        umap_latent_data, umap_model = umapWithoutPCA(
            np.concatenate([rna_integrate, atac_integrate], axis=0),
            n_neighbors=100, min_dist=0.5
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
    # ----
    df_savefile = "./res/data4Monocle/{}-scMultiNODE-concat_meta_df.csv".format(data_name)
    if not os.path.isfile(df_savefile):
        obs_df = pd.DataFrame({
            "tps": concat_cell_tps,
            "cell_types": concat_cell_types,
            "modality": concat_cell_mod
        })
        print(obs_df)
        obs_df.to_csv(df_savefile)
        np.savetxt("./res/data4Monocle/{}-scMultiNODE-concat_integrate.csv".format(data_name), concat_latent_seq, delimiter=",")
        np.savetxt("./res/data4Monocle/{}-scMultiNODE-umap_latent_data.csv".format(data_name), umap_latent_data, delimiter=",")
    else:
        print("Files already exist in the ./res/data4Monocle directory.")


