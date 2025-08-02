'''
Description:
    Prepare data for pseudotime estimation with Monocle3 or PAGA.

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os
import numpy as np
import pandas as pd
from plotting import *
from utils.FileUtils import loadSCData, tpSplitInd, loadAELatent
from plotting.PlottingUtils import umapWithoutPCA

# ======================================================

def _plotAELatent(latent_umap, all_tps, all_cell_types, modal_name):
    n_tps = len(np.unique(all_tps))
    # -----
    marker_s = 20
    marker_alpha = 0.7
    color_list = Kelly20
    # -----
    # AE latent visualization (RNA)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("{} latent (by time)".format(modal_name), fontsize=15)
    ax2.set_title("{} latent (by type)".format(modal_name), fontsize=15)
    for i, t in enumerate(range(n_tps)):
        t_idx = np.where(all_tps == t)[0]
        ax1.scatter(latent_umap[t_idx, 0], latent_umap[t_idx, 1], label=t, color=color_list[i], s=marker_s,
                    alpha=marker_alpha)
    ax1.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="TPs", title_fontsize=14, fontsize=13)

    cell_type_num = [(n, len(np.where(all_cell_types == n)[0])) for n in np.unique(all_cell_types)]
    n_c = len(cell_type_num)
    cell_type_num.sort(reverse=True, key=lambda x: x[1])
    select_cell_typs = [x[0] for x in cell_type_num[:n_c]]
    all_cell_types = np.asarray([x if x in select_cell_typs else "other" for x in all_cell_types])
    for i, n in enumerate(np.unique(all_cell_types)):
        n_idx = np.where(all_cell_types == n)[0]
        if n in select_cell_typs:
            c = color_list[select_cell_typs.index(n)]
        else:
            c = gray_color
        ax2.scatter(latent_umap[n_idx, 0], latent_umap[n_idx, 1], label=n, color=c, s=marker_s,
                    alpha=marker_alpha if n != "other" else 0.4)
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Cell Types", title_fontsize=14, fontsize=13)
    plt.tight_layout()
    plt.show()


def _perModalityAnalysis(modal_latent, modal_cell_types, modal_cell_tps, modal_name):
    # -----
    print("-" * 70)
    print("UMAP ({})...".format(modal_name))
    save_filename = "./res/aux_data/{}-reduce-all-Static_AE-{}_aux_data.npy".format(data_name, modal_name)
    print("Preparing data for analysis...")
    n_neighbors = 50
    min_dist = 0.5
    if not os.path.isfile(save_filename):
        umap_latent_data, umap_model = umapWithoutPCA(modal_latent, n_neighbors=n_neighbors, min_dist=min_dist)
        np.save(save_filename, {
            "modal_latent": modal_latent,
            "modal_cell_types": modal_cell_types,
            "modal_cell_tps": modal_cell_tps,
            "umap_latent_data": umap_latent_data,
            "umap_model": umap_model,
        })
    else:
        data_res = np.load(save_filename, allow_pickle=True).item()
        modal_latent = data_res["modal_latent"]
        modal_cell_types = data_res["modal_cell_types"]
        modal_cell_tps = data_res["modal_cell_tps"]
        umap_latent_data = data_res["umap_latent_data"]
        umap_model = data_res["umap_model"]  # Note: The UMAP is for the model latent space
    # -----
    df_savefile = "./res/data4Monocle/{}-Static_AE-{}-concat_meta_df.csv".format(data_name, modal_name)
    if not os.path.isfile(df_savefile):
        obs_df = pd.DataFrame({
            "tps": modal_cell_tps,
            "cell_types": modal_cell_types,
        })
        print(obs_df)
        obs_df.to_csv()
        np.savetxt("./res/data4Monocle/{}-Static_AE-{}-integrate.csv".format(data_name, modal_name), modal_latent, delimiter=",")
        np.savetxt("./res/data4Monocle/{}-Static_AE-{}-umap_latent_data.csv".format(data_name, modal_name), umap_latent_data, delimiter=",")
    else:
        print("Files already exist in the ./res/data4Monocle directory.")
    # -----
    # Plot AE latent
    _plotAELatent(umap_latent_data, modal_cell_tps, modal_cell_types, modal_name)



def AELatentAnalysis(rna_tps, atac_tps, rna_cell_types, atac_cell_types, latent_dim=50):
    print("-" * 70)
    print("Loading modality-specific AE latent...")
    (rna_data, atac_data, _, _, rna_latent, atac_latent,
     _, _, _, _) = loadAELatent(data_name, data_type, split_type, latent_dim, file_dir="../../modal_integration/res/preprocess_latent")
    print("rna_latent shape: ", rna_latent.shape)
    print("atac_latent shape: ", atac_latent.shape)
    # -----
    _perModalityAnalysis(rna_latent, rna_cell_types, rna_tps, "RNA")
    _perModalityAnalysis(atac_latent, atac_cell_types, atac_tps, "ATAC")



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
    rna_cell_types = np.concatenate(rna_traj_cell_type)
    atac_cell_types = np.concatenate(atac_traj_cell_type)
    rna_cell_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(rna_traj_data)])
    atac_cell_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(atac_traj_data)])
    # -----
    # modality-specific static latent
    AELatentAnalysis(rna_cell_tps, atac_cell_tps, rna_cell_types, atac_cell_types, latent_dim=50)


