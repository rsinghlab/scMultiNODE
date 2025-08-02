'''
Description:
    Germ layer label transfer with baseline models' integration.

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os
import numpy as np
import pandas as pd
from utils.FileUtils import loadSCData, tpSplitInd
from plotting.PlottingUtils import umapWithoutPCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ======================================================

def loadModelLatent(data_name, data_type, split_type, model_name, latent_dim):
    save_filename = "../../modal_integration/res/model_latent/{}-{}-{}-{}-{}dim.npy"
    seurat_save_filename = "../../modal_integration/res/model_latent/{}-{}-{}-{}-{}dim-{}.csv"
    if model_name == "Seurat":
        rna_integrated = pd.read_csv(
            seurat_save_filename.format(data_name, data_type, split_type, m, latent_dim, "rna"), header=0, index_col=None).values
        atac_integrated = pd.read_csv(
            seurat_save_filename.format(data_name, data_type, split_type, m, latent_dim,
                                        "atac"), header=0, index_col=None).values
    else:
        res = np.load(save_filename.format(data_name, data_type, split_type, m, latent_dim),
                      allow_pickle=True).item()
        rna_integrated = res["rna_integrated"]
        atac_integrated = res["atac_integrated"]
    return rna_integrated, atac_integrated


if __name__ == '__main__':
    # Loading data
    data_name = "drosophila"
    split_type = "all"
    data_type = "reduce"
    data_dir = "../../data/drosophila_embryonic/reduce_processed/"
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
    # RNA germ germ_layer
    rna_germ_layer = np.asarray([x if isinstance(x, str) else "NA" for x in ann_rna_data.obs["germ_layer"].values])
    rna_traj_germ_layer = [rna_germ_layer[np.where(rna_cell_tps == t)[0]] for t in range(1, rna_n_tps + 1)]
    # Convert to torch project
    rna_traj_data = [rna_cnt[np.where(rna_cell_tps == t)[0], :] for t in range(1, rna_n_tps + 1)]  # (# tps, # cells, # genes)
    atac_traj_data = [atac_cnt[np.where(atac_cell_tps == t)[0], :] for t in range(1, atac_n_tps + 1)]  # (# tps, # cells, # peaks)
    all_rna_data = np.concatenate(rna_traj_data)
    all_atac_data = np.concatenate(atac_traj_data)
    n_tps = len(rna_traj_data)
    all_tps = list(range(n_tps))
    print("RNA shape: ", ann_rna_data.shape)
    print("ATAC shape: ", ann_atac_data.shape)
    print("# genes={}, # peaks={}".format(n_genes, n_peaks))
    # -----
    import importlib.util
    module_name = "1.1_scMultiNODE_GermLabelTransfer"
    file_path = "1.1_scMultiNODE_GermLabelTransfer.py"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    scMultiNODE_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scMultiNODE_module)
    # -----
    model_list = ["SCOTv1", "SCOTv2", "Pamona", "UnionCom", "uniPort", "Seurat"]
    for m in model_list:
        print("-" * 70)
        save_filename = "./res/germ_layer/{}-reduce-all-{}-label_transfer.npy".format(data_name, m)
        print("Preparing data for analysis [{}] ...".format(m))
        n_neighbors = 100
        min_dist = 0.3
        if not os.path.isfile(save_filename):
            rna_integrate, atac_integrate = loadModelLatent(
                data_name, data_type, split_type, model_name=m, latent_dim=50)
            rna_cell_types = np.concatenate(rna_traj_cell_type)
            atac_cell_types = np.concatenate(atac_traj_cell_type)
            rna_cell_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(rna_traj_data)])
            atac_cell_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(atac_traj_data)])
            rna_germ_layer = np.concatenate(rna_traj_germ_layer)
            # -----
            select_idx = np.where((rna_germ_layer != "NA") & (rna_germ_layer != "unknown"))[0]
            rna_integrate = rna_integrate[select_idx, :]
            rna_cell_types = rna_cell_types[select_idx]
            rna_cell_tps = rna_cell_tps[select_idx]
            rna_germ_layer = rna_germ_layer[select_idx]
            print("-" * 70)
            print("Label transfer...")
            atac_germ_layer, class_model = scMultiNODE_module.germLabelTransfer(rna_integrate, rna_germ_layer, atac_integrate)
            # -----
            concat_cell_types = np.concatenate([rna_cell_types, atac_cell_types])
            concat_cell_tps = np.concatenate([rna_cell_tps, atac_cell_tps])
            concat_cell_mod = np.concatenate([
                ["rna" for _ in range(rna_integrate.shape[0])],
                ["atac" for _ in range(atac_integrate.shape[0])]
            ])
            concat_germ_layer = np.concatenate([rna_germ_layer, atac_germ_layer])
            concat_latent_seq = np.concatenate([rna_integrate, atac_integrate], axis=0)
            print("-" * 70)
            print("UMAP...")
            umap_latent_data, umap_model = umapWithoutPCA(
                concat_latent_seq,
                n_neighbors=n_neighbors, min_dist=min_dist
            )

            np.save(save_filename, {
                "concat_latent_seq": concat_latent_seq,
                "concat_cell_types": concat_cell_types,
                "concat_cell_tps": concat_cell_tps,
                "concat_cell_mod": concat_cell_mod,
                "concat_germ_layer": concat_germ_layer,

                "class_model": class_model,

                "select_idx": select_idx,
                "umap_latent_data": umap_latent_data,
                "umap_model": umap_model,
            })
        else:
            data_res = np.load(save_filename, allow_pickle=True).item()
            concat_latent_seq = data_res["concat_latent_seq"]
            concat_cell_types = data_res["concat_cell_types"]
            concat_cell_tps = data_res["concat_cell_tps"]
            concat_cell_mod = data_res["concat_cell_mod"]
            concat_germ_layer = data_res["concat_germ_layer"]
            class_model = data_res["class_model"]
            select_idx = data_res["select_idx"]
            umap_latent_data = data_res["umap_latent_data"]
            umap_model = data_res["umap_model"]  # Note: The UMAP is for the model latent space
        # -----
        print(select_idx.shape)
        print(concat_cell_types.shape)
        print(concat_cell_tps.shape)
        print(concat_cell_mod.shape)
        print(concat_germ_layer.shape)
        print(concat_latent_seq.shape)
