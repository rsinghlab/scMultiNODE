'''
Description:
    Utility functions for figure plotting.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
from matplotlib import pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA

# ======================================
from plotting import Kelly20, gray_color


def computeVisEmbedding(true_data, model_pred_data, embed_name):
    if embed_name == "umap":
        true_umap_traj, umap_model = umapWithoutPCA(np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.2)
        model_pred_umap_traj = [umap_model.transform(np.concatenate(m_pred, axis=0)) for m_pred in model_pred_data]
    elif embed_name == "pca_umap":
        true_umap_traj, umap_model, pca_model = umapWithPCA(
            np.concatenate(true_data, axis=0), n_neighbors=50, min_dist=0.1, pca_pcs=50
        )
        model_pred_umap_traj = [
            umap_model.transform(pca_model.transform(np.concatenate(m_pred, axis=0)))
            for m_pred in model_pred_data
        ]
    else:
        raise ValueError("Unknown embedding type {}!".format(embed_name))
    return true_umap_traj, model_pred_umap_traj


def umapWithPCA(traj_data, n_neighbors, min_dist, pca_pcs):
    pca_model = PCA(n_components=pca_pcs, svd_solver="arpack")
    umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_traj_data = umap_model.fit_transform(pca_model.fit_transform(traj_data))
    return umap_traj_data, umap_model, pca_model


def umapWithoutPCA(traj_data, n_neighbors, min_dist):
    umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_traj_data = umap_model.fit_transform(traj_data)
    return umap_traj_data, umap_model


def onlyPCA(traj_data, pca_pcs):
    pca_model = PCA(n_components=pca_pcs, svd_solver="arpack")
    pca_traj_data = pca_model.fit_transform(traj_data)
    return pca_traj_data, pca_model

# ======================================

def computeLatentEmbedding(latent_seq, next_seq, n_neighbors, min_dist):
    latent_tp_list = np.concatenate([np.repeat(t, each.shape[0]) for t, each in enumerate(latent_seq)])
    umap_latent_data, umap_model = umapWithoutPCA(np.concatenate(latent_seq, axis=0), n_neighbors=n_neighbors, min_dist=min_dist)
    # umap_latent_data, umap_model = onlyPCA(np.concatenate(latent_seq, axis=0), pca_pcs=2)
    umap_latent_data = [umap_latent_data[np.where(latent_tp_list == t)[0], :] for t in range(len(latent_seq))]
    umap_next_data = [umap_model.transform(each) for each in next_seq]
    return umap_latent_data, umap_next_data, umap_model, latent_tp_list

# ======================================

def plotIntegration(rna_integrated, atac_integrated, rna_cell_types, atac_cell_types, rna_tps, atac_tps, model_name, data_name = ""):
    color_list = Kelly20
    # -----
    umap_neighbors = 50
    umap_dist = 0.9
    marker_s = 20
    marker_alpha = 0.7
    # -----
    # Plotting for fusion latent (jointly)
    mod_list = np.asarray(["rna" for _ in range(rna_integrated.shape[0])] + ["atac" for _ in range(atac_integrated.shape[0])])
    cell_type_list = np.concatenate([rna_cell_types, atac_cell_types], axis=0)
    cell_tp_list = np.concatenate([rna_tps, atac_tps], axis=0)
    concat_latent_sample = np.concatenate([rna_integrated, atac_integrated], axis=0)
    n_tps = len(np.unique(cell_tp_list))
    latent_umap, _ = umapWithoutPCA(concat_latent_sample, n_neighbors=umap_neighbors, min_dist=umap_dist)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    ax1.set_title(model_name)
    for i, t in enumerate(range(n_tps)):
        t_idx = np.where(cell_tp_list == t)[0]
        ax1.scatter(latent_umap[t_idx, 0], latent_umap[t_idx, 1], label=t, color=color_list[i], s=marker_s,
                    alpha=marker_alpha)
    ax1.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="TPs", title_fontsize=14, fontsize=13)

    if len(rna_cell_types) == len(atac_cell_types):
        cell_type_num = [(n, len(np.where(cell_type_list == n)[0])) for n in np.unique(cell_type_list)]
        cell_type_num.sort(reverse=True, key=lambda x: x[1])
        select_cell_typs = [x[0] for x in cell_type_num[:10]]
        cell_type_list = np.asarray([x if x in select_cell_typs else "other" for x in cell_type_list])
    else:
        cell_type_num = [
            (n, len(np.where(cell_type_list == n)[0]))
            for n in np.unique(cell_type_list)
            if n in np.unique(np.intersect1d(rna_cell_types, atac_cell_types))
        ]
        cell_type_num.sort(reverse=True, key=lambda x: x[1])
        select_cell_typs = [x[0] for x in cell_type_num[:10]]
        cell_type_list = np.asarray([x if x in select_cell_typs else "other" for x in cell_type_list])
    if data_name == "zebrahub":
        n_idx = np.where(cell_type_list == "other")[0]
        ax2.scatter(latent_umap[n_idx, 0], latent_umap[n_idx, 1], label="other", color=gray_color, s=marker_s, alpha=0.4)
        for i, n in enumerate(np.unique(cell_type_list)):
            if n == "other":
                continue
            n_idx = np.where(cell_type_list == n)[0]
            if n in select_cell_typs:
                c = color_list[select_cell_typs.index(n)]
            else:
                c = gray_color
            ax2.scatter(
                latent_umap[n_idx, 0], latent_umap[n_idx, 1], label=n.split(" ")[0], color=c, s=marker_s,
                alpha=marker_alpha if n != "other" else 0.4
            )
    for i, n in enumerate(np.unique(cell_type_list)):  #
        n_idx = np.where(cell_type_list == n)[0]
        if n in select_cell_typs:
            c = color_list[select_cell_typs.index(n)]
        else:
            c = gray_color
        ax2.scatter(latent_umap[n_idx, 0], latent_umap[n_idx, 1], label=n.split(" ")[0], color=c, s=marker_s,
                    alpha=marker_alpha if n != "other" else 0.4)
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Cell Types", title_fontsize=14, fontsize=13)

    for i, m in enumerate(["rna", "atac"]):
        m_idx = np.where(mod_list == m)[0]
        ax3.scatter(latent_umap[m_idx, 0], latent_umap[m_idx, 1], label=m, color=color_list[i], s=marker_s, alpha=0.25)
    ax3.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="Modality", title_fontsize=14, fontsize=13)
    plt.tight_layout()
    plt.show()
    # --------------------------
    # Plot by cell type
    x_min, x_max = latent_umap[:, 0].min() - 1.0, latent_umap[:, 0].max() + 1.0
    y_min, y_max = latent_umap[:, 1].min() - 1.0, latent_umap[:, 1].max() + 1.0
    fig, ax_list = plt.subplots(2, 5, figsize=(15, 8))
    ax_list[0, 0].set_title(model_name)
    rna_latent_umap = latent_umap[np.where(mod_list == "rna")[0]]
    atac_latent_umap = latent_umap[np.where(mod_list == "atac")[0]]
    rna_cell_type_list = cell_type_list[np.where(mod_list == "rna")[0]]
    atac_cell_type_list = cell_type_list[np.where(mod_list == "atac")[0]]
    cell_type_num = [(n, len(np.where(cell_type_list == n)[0])) for n in
                     np.unique(np.intersect1d(rna_cell_type_list, atac_cell_type_list)) if n != "other"]
    cell_type_num.sort(reverse=True, key=lambda x: x[1])
    select_cell_types = np.asarray([x[0] for x in cell_type_num[:5]])
    for i, n in enumerate(select_cell_types):  #
        rna_idx = np.where(rna_cell_type_list == n)[0]
        atac_idx = np.where(atac_cell_type_list == n)[0]
        ax_list[0, i].set_title(n)
        ax_list[0, i].scatter(
            rna_latent_umap[rna_idx, 0], rna_latent_umap[rna_idx, 1],
            color=color_list[0], s=marker_s, alpha=marker_alpha
        )
        ax_list[1, i].scatter(
            atac_latent_umap[atac_idx, 0], atac_latent_umap[atac_idx, 1],
            color=color_list[1], s=marker_s, alpha=marker_alpha
        )
    for i in range(2):
        for j in range(5):
            ax_list[i, j].set_xlim(x_min, x_max)
            ax_list[i, j].set_ylim(y_min, y_max)
    ax_list[0, 0].set_ylabel("RNA")
    ax_list[1, 0].set_ylabel("ATAC")
    plt.tight_layout()
    plt.show()