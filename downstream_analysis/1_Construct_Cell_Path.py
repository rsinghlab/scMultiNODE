'''
Description:
    Construct cell path in the joint latent space.

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
from utils.FileUtils import loadSCData, tpSplitInd
from optim.running import constructscMultiNODEModel
from plotting.PlottingUtils import umapWithoutPCA
import scanpy


pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# ======================================================
# Load Trained Model and Pre-Computed Latent
# ======================================================

def loadModel():
    dict_filename = "res/trained_model/coassay_cortex-reduce-all-scMultiNODE-50dim-state_dict.pt"
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
        "res/trained_model/coassay_cortex-reduce-all-scMultiNODE-50dim.npy",
        allow_pickle=True
    ).item()
    rna_integrate = res["rna_integrated"]
    atac_integrate = res["atac_integrated"]
    return rna_integrate, atac_integrate


# ======================================================
# LAP Algorithm
# ======================================================

def _discreteVelo(cur_x, last_x, dt):
    # compute discretized tangential velocity
    velo = (cur_x - last_x) / dt
    return velo


def _netVelo(cur_x, last_x, vec_field):
    # compute vector field velocity
    mid_point = (cur_x + last_x) / 2
    velo = vec_field(torch.FloatTensor(mid_point)).detach().numpy()
    return velo


def _action(P, dt, vec_field, D, latent_dim):
    if len(P.shape) == 1:
        P = P.reshape(-1, latent_dim)
    cur_x = P[1:, :]
    last_x = P[:-1, :]
    v = _discreteVelo(cur_x, last_x, dt)
    f = _netVelo(cur_x, last_x, vec_field)
    s = 0.5 * np.square(np.linalg.norm(v-f, ord="fro")) * dt / D
    return s.item()


def leastActionPath(x_0, x_T, path_length, vec_field, D, iters):
    dt = 1
    P = np.linspace(x_0, x_T, num=path_length, endpoint=True, axis=0)
    iter_pbar = tqdm(range(iters), desc="[ LAP ]")
    K = P.shape[1]
    action_list = [_action(P, dt, vec_field, D, K)]
    dt_list = [dt]
    P_list = [P]
    best_dt = dt
    best_P = P
    best_s = action_list[-1]
    for _ in iter_pbar:
        # Step 1: minimize step dt
        dt_res = minimize(
            lambda t: _action(P, dt=t, vec_field=vec_field, D=D, latent_dim=K),
            dt,
            bounds=((1e-5, None), )
        )
        dt = dt_res["x"].item()
        dt_list.append(dt)
        # Step 2: minimize path
        path_res = minimize(
            lambda p: _action(P=p, dt=dt, vec_field=vec_field, D=D, latent_dim=K),
            P[1:-1, :].reshape(-1),
            method="SLSQP",
            tol=1e-5, options={'disp': False ,'eps' : 1e-2},
        )
        inter_P = path_res["x"].reshape(-1, K)
        P = np.concatenate([x_0[np.newaxis, :], inter_P, x_T[np.newaxis, :]], axis=0)
        P_list.append(P)
        # Compute action
        s = _action(P, dt, vec_field, D, K)
        action_list.append(s)
        iter_pbar.set_postfix({"Action": "{:.3f}".format(s)})
        if s < best_s:
            best_dt = dt
            best_P = P
    return best_dt, best_P, action_list, dt_list, P_list


def _interpSpline(x, y):
    x_idx = np.argsort(x)
    sort_x = x[x_idx]
    sort_y = y[x_idx]
    cs = scipy.interpolate.CubicSpline(sort_x, sort_y)
    new_x = np.linspace(sort_x[0], sort_x[-1], 100)
    new_y = cs(new_x)
    return new_x, new_y


def plotLAP(
        umap_latent_data, umap_OL_path, umap_GN_path,
        OL_idx, GN_idx, start_idx
):
    concat_umap_latent = umap_latent_data
    color_list = Bold_10.mpl_colors
    plt.figure(figsize=(6.5, 4))
    plt.scatter(concat_umap_latent[:, 0], concat_umap_latent[:, 1], color=gray_color, s=5, alpha=0.4)
    plt.scatter(
        concat_umap_latent[OL_idx, 0], concat_umap_latent[OL_idx, 1],
        color=color_list[0], s=10, alpha=0.4
    )
    plt.scatter(
        concat_umap_latent[GN_idx, 0], concat_umap_latent[GN_idx, 1],
        color=color_list[1], s=10, alpha=0.4
    )
    plt.scatter(
        concat_umap_latent[start_idx, 0], concat_umap_latent[start_idx, 1],
        color=color_list[2], s=10, alpha=0.4
    )

    spline_OL_x, spline_OL_y = _interpSpline(umap_OL_path[:, 0], umap_OL_path[:, 1])
    spline_GN_x, spline_GN_y = _interpSpline(umap_GN_path[:, 0], umap_GN_path[:, 1])
    plt.plot(
        spline_OL_x, spline_OL_y, "--", lw=3,
        color=color_list[0],
    )
    plt.plot(
        spline_GN_x, spline_GN_y, "--", lw=3,
        color=color_list[1],
    )

    plt.scatter(umap_OL_path[: ,0], umap_OL_path[:, 1], c=color_list[0], s=200, marker="o", edgecolors= "black")
    plt.scatter(umap_GN_path[: ,0], umap_GN_path[:, 1], c=color_list[1], s=200, marker="o", edgecolors= "black")

    plt.scatter([], [], color=color_list[0], s=150, alpha=1.0, label="Oligodendrocyte")
    plt.scatter([], [], color=color_list[1], s=150, alpha=1.0, label="Glutamatergic")
    plt.scatter([], [], color=color_list[2], s=150, alpha=1.0, label="t=0")

    plt.xticks([], [])
    plt.yticks([], [])
    removeAllBorders()
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=11)
    plt.tight_layout()
    plt.show()


# ======================================================
# Detect DE Genes
# ======================================================

def pathKNNGenes(path_data, latent_data, K):
    nn = NearestNeighbors(n_neighbors=K, n_jobs=-1)
    nn.fit(latent_data)
    dists, neighs = nn.kneighbors(path_data)
    return neighs


# ======================================================
# Plot Gene Expression Curves
# ======================================================

from scipy.interpolate import interp1d
def _plotSmoothCurve(ax, x, y, style, color, alpha, lw, n_inter=100):
    # Create a B-spline representation of the data
    spl = interp1d(x, y, kind='cubic')
    # Generate a denser set of x values for a smoother curve
    x_smooth = np.linspace(x.min(), x.max(), n_inter)
    # Evaluate the spline at the denser x values
    y_smooth = spl(x_smooth)
    # Plot the original data and the smooth spline
    # ax.scatter(x, y)
    ax.plot(x_smooth, y_smooth, style, c=color, alpha=alpha, lw=lw)


def plotGeneExpressionAlongPath(ann_data, cell_tps, gene_list, gene_name_list, path_idx, path_name=""):
    other_idx = [x for x in np.arange(ann_data.shape[0]) if x not in path_idx]
    path_cell_tps = cell_tps[path_idx]
    other_cell_tps = cell_tps[other_idx]

    ann_data.X = (ann_data.X - np.mean(ann_data.X, axis=0)[np.newaxis,:]) / np.std(ann_data.X, axis=0)[np.newaxis,:] # z-score

    path_gene_expr = ann_rna_data[ann_data.obs_names[path_idx], gene_list].X
    other_gene_expr = ann_rna_data[ann_data.obs_names[other_idx], gene_list].X

    n_random = 5
    np.random.seed(666 if path_name=="OL" else 333)
    other_gene_list = [x for x in ann_data.var_names.values if x not in gene_list]
    random_gene_list = np.random.choice(other_gene_list, n_random, replace=False)
    path_random_gene_expr = ann_rna_data[ann_data.obs_names[path_idx], random_gene_list].X
    other_random_gene_expr = ann_rna_data[ann_data.obs_names[other_idx], random_gene_list].X

    n_genes = len(gene_list)
    tp_list = np.unique(cell_tps)

    expression_list = [[], []]
    # -----
    lw = 1.5
    alpha1 = 1.0
    alpha2 = 0.4
    color1 = Bold_10.mpl_colors[0] if path_name=="OL" else Bold_10.mpl_colors[1]
    color2 = dark_gray_color
    ms = 15
    x = np.arange(n_tps)
    fig, ax_list = plt.subplots(1, n_genes, figsize=(6, 4))
    for i, g in enumerate(gene_list):
        print("-" * 70)
        print("Gene {}".format(gene_name_list[i]))
        ax_list[i].set_title(gene_name_list[i])
        ax_list[i].set_xticks(np.arange(len(tp_list)), tp_list)
        removeTopRightBorders(ax_list[i])
        # -----
        # DE gene on path
        g_expr = path_gene_expr[:, i]
        t_expr = [g_expr[np.where(path_cell_tps == t)[0]] for t in tp_list]
        avg_t_expr = [np.mean(x) for x in t_expr]
        ax_list[i].scatter(x, avg_t_expr, c=color1, alpha=alpha1, s=ms)
        _plotSmoothCurve(ax_list[i], x, avg_t_expr, style="-", color=color1, alpha=alpha1, lw=lw)
        expression_list[i].append(avg_t_expr.copy())
        # -----
        # DE gene off path
        other_g_expr = other_gene_expr[:, i]
        t_expr = [other_g_expr[np.where(other_cell_tps == t)[0]] for t in tp_list]
        avg_t_expr = [np.mean(x) for x in t_expr]
        ax_list[i].scatter(x, avg_t_expr, c=color1, alpha=alpha2, s=ms)
        _plotSmoothCurve(ax_list[i], x, avg_t_expr, style="--", color=color1, alpha=alpha2, lw=lw)
        expression_list[i].append(avg_t_expr.copy())
        # ----
        # Random gene on path
        r_g_expr = np.nanmean(path_random_gene_expr, axis=1)
        t_expr = [r_g_expr[np.where(path_cell_tps == t)[0]] for t in tp_list]
        avg_t_expr = [np.nanmean(x).item() for x in t_expr]
        ax_list[i].scatter(x, avg_t_expr, c=color2, alpha=alpha1, s=ms)
        _plotSmoothCurve(ax_list[i], x, avg_t_expr, style="-", color=color2, alpha=alpha1, lw=lw)
        expression_list[i].append(avg_t_expr.copy())
        # ----
        # Random gene off path
        r_g_expr = np.nanmean(other_random_gene_expr, axis=1)
        t_expr = [r_g_expr[np.where(other_cell_tps == t)[0]] for t in tp_list]
        avg_t_expr = [np.nanmean(x).item() for x in t_expr]
        ax_list[i].scatter(x, avg_t_expr, c=color2, alpha=alpha2, s=ms)
        _plotSmoothCurve(ax_list[i], x, avg_t_expr, style="--", color=color2, alpha=alpha2, lw=lw)
        expression_list[i].append(avg_t_expr.copy())
        # -----
        de_on_out_ks = scipy.stats.ks_2samp(expression_list[i][0], expression_list[i][1])
        de_random_on_ks = scipy.stats.ks_2samp(expression_list[i][0], expression_list[i][2])
        de_on_random_out_ks = scipy.stats.ks_2samp(expression_list[i][0], expression_list[i][3])
        print("DE (on) - DE (out) KS p-value = {}".format(de_on_out_ks.pvalue))
        print("DE (on) - random (on) KS -value = {}".format(de_random_on_ks.pvalue))
        print("DE (on) - random (out) KS p-value = {}".format(de_on_random_out_ks.pvalue))

        de_on_out_tt = scipy.stats.ttest_rel(expression_list[i][0], expression_list[i][1])
        de_random_on_tt = scipy.stats.ttest_rel(expression_list[i][0], expression_list[i][2])
        de_on_random_out_tt = scipy.stats.ttest_rel(expression_list[i][0], expression_list[i][3])
        print("DE (on) - DE (out) tt p-value = {}".format(de_on_out_tt.pvalue))
        print("DE (on) - random (on) tt p-value = {}".format(de_random_on_tt.pvalue))
        print("DE (on) - random (out) tt p-value = {}".format(de_on_random_out_tt.pvalue))
        # -----

    ax_list[0].set_ylabel("z-score", fontsize=20)
    ax_list[0].set_xlabel("timepoint", fontsize=20)
    ax_list[1].set_xlabel("timepoint", fontsize=20)
    ax_list[-1].plot([], [], "-", c=color1, alpha=alpha1, lw=2.5, label="DE (on)")
    ax_list[-1].plot([], [], "--", c=color1, alpha=alpha1, lw=2.5, label="DE (out)")
    ax_list[-1].plot([], [], "-", c=color2, alpha=alpha1, lw=2.5, label="random (on)")
    ax_list[-1].plot([], [], "--", c=color2, alpha=alpha1, lw=2.5, label="random (out)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot([], [], "-", c=color1, alpha=alpha1, lw=2.5, label="DE (on)")
    plt.plot([], [], "--", c=color1, alpha=alpha1, lw=2.5, label="DE (out)")
    plt.plot([], [], "-", c=color2, alpha=alpha1, lw=2.5, label="random (on)")
    plt.plot([], [], "--", c=color2, alpha=alpha1, lw=2.5, label="random (out)")
    plt.legend(loc='center', fontsize=20, ncol=4, handletextpad=0.01)
    removeAllBorders(plt.gca())
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Loading data
    data_name = "coassay_cortex"
    split_type = "all"
    data_type = "reduce"
    data_dir = "../data/human_prefrontal_cortex_multiomic/reduce_processed/"
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
    print("=" * 70)
    print("Loading model...")
    dynamic_model = loadModel()
    print(dynamic_model)
    # -----
    print("-" * 70)
    save_filename = "./res/aux_data/coassay_cortex-reduce-all-scMultiNODE-aux_data.npy"
    print("Preparing data for analysis...")
    if not os.path.isfile(save_filename):
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
    # =====================================================
    # Construct cell path
    print("=" * 70)
    print("Construction LAP...")
    path_filename = "./res/aux_data/coassay_cortex-reduce-all-scMultiNODE-path_data.npy"
    if not os.path.isfile(path_filename):
        OL_cell_idx = np.intersect1d(
            np.where(concat_cell_types == "oligodendrocyte")[0],
            np.where(concat_cell_tps == n_tps - 1)[0],
        )
        GN_cell_idx = np.intersect1d(
            np.where(concat_cell_types == "glutamatergic neuron")[0],
            np.where(concat_cell_tps == n_tps - 1)[0],
        )
        start_cell_idx = np.where(concat_cell_tps == 0)[0]
        print("OL={} | OL={} | Starting={}".format(len(OL_cell_idx), len(GN_cell_idx), len(start_cell_idx)))

        start_cell_mean = np.mean(concat_latent_seq[start_cell_idx, :], axis=0)
        OL_cell_mean = np.mean(concat_latent_seq[OL_cell_idx, :], axis=0)
        GN_cell_mean = np.mean(concat_latent_seq[GN_cell_idx, :], axis=0)
        path_length = 8
        OL_dt, OL_P, OL_action_list, OL_dt_list, OL_P_list = leastActionPath(
            x_0=start_cell_mean, x_T=OL_cell_mean,
            path_length=path_length, vec_field=dynamic_model.diffeq_decoder.net, D=1, iters=10
        )
        GN_dt, GN_P, GN_action_list, GN_dt_list, GN_P_list = leastActionPath(
            x_0=start_cell_mean, x_T=GN_cell_mean,
            path_length=path_length, vec_field=dynamic_model.diffeq_decoder.net, D=1, iters=10
        )
        np.save(path_filename,
                {
                    "start_cell_idx": start_cell_idx,
                    "OL_cell_idx": OL_cell_idx,
                    "GN_cell_idx": GN_cell_idx,

                    "start_cell_mean": start_cell_mean,
                    "OL_cell_mean": OL_cell_mean,
                    "GN_cell_mean": GN_cell_mean,

                    "OL_dt": OL_dt,
                    "OL_P": OL_P,
                    "OL_action_list": OL_action_list,
                    "OL_dt_list": OL_dt_list,
                    "OL_P_list": OL_P_list,

                    "GN_dt": GN_dt,
                    "GN_P": GN_P,
                    "GN_action_list": GN_action_list,
                    "GN_dt_list": GN_dt_list,
                    "GN_P_list": GN_P_list,
                })
    else:
        path_data = np.load(path_filename, allow_pickle=True).item()
        start_cell_idx = path_data["start_cell_idx"]
        OL_cell_idx = path_data["OL_cell_idx"]
        GN_cell_idx = path_data["GN_cell_idx"]

        start_cell_mean = path_data["start_cell_mean"]
        OL_cell_mean = path_data["OL_cell_mean"]
        GN_cell_mean = path_data["GN_cell_mean"]

        OL_dt = path_data["OL_dt"]
        OL_P = path_data["OL_P"]
        OL_action_list = path_data["OL_action_list"]
        OL_dt_list = path_data["OL_dt_list"]
        OL_P_list = path_data["OL_P_list"]

        GN_dt = path_data["GN_dt"]
        GN_P = path_data["GN_P"]
        GN_action_list = path_data["GN_action_list"]
        GN_dt_list = path_data["GN_dt_list"]
        GN_P_list = path_data["GN_P_list"]

    # -----
    # Plot path
    umap_OL_P = umap_model.transform(OL_P)
    umap_GN_P = umap_model.transform(GN_P)
    plotLAP(
        umap_latent_data, umap_OL_P, umap_GN_P,
        OL_cell_idx, GN_cell_idx, start_cell_idx
    )
    # =====================================================
    print("=" * 70)
    print("Detecting DE genes from cell path...")
    concat_rna_latent_seq = rna_integrate
    # Detect differentially expressed genes
    # To augment cell sets, we find KNN neighbors of path nodes and then detect genes in the gene space
    OL_KNN_idx = pathKNNGenes(OL_P, concat_rna_latent_seq, K=10)
    GN_KNN_idx = pathKNNGenes(GN_P, concat_rna_latent_seq, K=10)
    de_filename = "./res/path_DE/path_DE_genes_wilcoxon.csv"
    if not os.path.isfile(de_filename):
        concat_traj_data = np.concatenate([each for each in rna_traj_data], axis=0)
        OL_gene = np.concatenate([concat_traj_data[idx, :] for idx in OL_KNN_idx], axis=0)
        GN_gene = np.concatenate([concat_traj_data[idx, :] for idx in GN_KNN_idx], axis=0)
        expr_mat = np.concatenate([OL_gene, GN_gene], axis=0)
        cell_idx = ["cell_{}".format(i) for i in range(expr_mat.shape[0])]
        cell_types = ["OL" for t in range(OL_gene.shape[0])] + ["GN" for t in range(GN_gene.shape[0])]
        cell_df = pd.DataFrame(data=np.zeros((expr_mat.shape[0], 2)), index=cell_idx, columns=["TP", "TYPE"])
        cell_df.TYPE = cell_types
        expr_df = pd.DataFrame(data=expr_mat, index=cell_idx, columns=ann_rna_data.var_names.values)
        path_ann = scanpy.AnnData(X=expr_df, obs=cell_df)
        scanpy.tl.rank_genes_groups(path_ann, 'TYPE', method="wilcoxon")
        scanpy.pl.rank_genes_groups(path_ann, n_genes=25, sharey=False, fontsize=12)
        group_id = path_ann.uns["rank_genes_groups"]["names"].dtype.names
        gene_names = path_ann.uns["rank_genes_groups"]["names"]
        print(group_id)
        print(gene_names[:10])
        marker_gene_dict = {}
        for i, g_name in enumerate(group_id):
            g_gene = [x[i] for x in gene_names]
            marker_gene_dict[g_name] = g_gene
        pd.DataFrame(marker_gene_dict).to_csv(de_filename)
    else:
        marker_gene_dict = pd.read_csv(de_filename)
        print(marker_gene_dict.head())
    # =====================================================
    # Plot gene expression (z-score) curve
    print("=" * 70)
    print("Plot gene expression curve...")
    top_GN_genes = ['ENSG00000185518', "ENSG00000134343"]
    top_GN_genes_name = ['SV2B', "ANO3"]
    plotGeneExpressionAlongPath(
        ann_rna_data, rna_cell_tps, top_GN_genes, top_GN_genes_name, path_idx=np.concatenate(GN_KNN_idx), path_name="GN"
    )

    top_OL_genes = ['ENSG00000110693', 'ENSG00000079215']
    top_OL_genes_name = ['SOX6', 'SLC1A3']
    plotGeneExpressionAlongPath(
        ann_rna_data, rna_cell_tps, top_OL_genes, top_OL_genes_name, path_idx=np.concatenate(OL_KNN_idx), path_name="OL"
    )
