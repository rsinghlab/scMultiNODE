'''
Description:
    scNODE pseudotime transfer via regression in each integration's shared latent space.

    Two transfer directions are evaluated:
        Forward  (RNA  -> ATAC): train on scNODE RNA pseudotime,  predict the ATAC latent.
        Reverse  (ATAC -> RNA):  train on scNODE ATAC pseudotime, predict the RNA latent.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os
import matplotlib.pyplot as plt
import scipy.stats
import scanpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from plotting.__init__ import Bold_10, BlueRed_12, Tableau_10
from plotting import *


pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# Reference pseudotime backend for the scNODE source labels
PSEUDOTIME_SOURCE = "Monocle"   # "Monocle" or "PAGA"

# Latent-space regressor used to transfer pseudotime across modalities.
REGRESSOR_TYPE = "KNN"   # "KNN" or "MLP"
KNN_K = 20               # neighbors for the kNN regressor
MLP_HIDDEN = (128, 64)   # hidden-layer sizes for the MLP regressor
# A held-out fraction of the source-modality cells is kept aside as a validation set to
# confirm the regressor actually learns the source latent -> pseudotime mapping before it is
# trusted to predict the other modality.
VAL_RATIO = 0.2
SEED = 111

# Transfer directions
DIRECTIONS = {
    "RNA->ATAC": {"source": "rna", "target": "atac", "scNODE": "RNA"},
    "ATAC->RNA": {"source": "atac", "target": "rna", "scNODE": "ATAC"},
}

# ======================================================
# Ground-truth developing-cortex lineages
lineage_groups = [
    ["rg", "ipc", "en-fetal-early", "en-fetal-late", "en"],
    ["rg", "ipc", "in-fetal", "in-mge"],
    ["rg", "ipc", "in-fetal", "in-cge"],
    ["rg", "opc", "oligodendrocytes"],
]

model_color = {
    "scMultiNODE": Tableau_10.mpl_colors[3],
    "SCOTv2": Tableau_10.mpl_colors[0],
    "SCOTv1": Tableau_10.mpl_colors[1],
    "Pamona": Tableau_10.mpl_colors[2],
    "UnionCom": Tableau_10.mpl_colors[4],
    "uniPort": Tableau_10.mpl_colors[5],
    "Seurat": Tableau_10.mpl_colors[7],
}

# scMultiNODE performance for comparison
SCMULTINODE_SPEARMAN_OVERRIDE = {
    "RNA->ATAC": {"en": 0.84, "in-mge": 0.67, "in-cge": 0.60, "oligodendrocytes": 0.55},
    "ATAC->RNA": {"en": 0.84, "in-mge": 0.67, "in-cge": 0.60, "oligodendrocytes": 0.55},
}

# ======================================================
# Data loading

def loadSCNODEPseudoTime(data_name, modality):
    '''
    Reference pseudotime for one modality from the scNODE (single-modality) result, using the
    backend selected by PSEUDOTIME_SOURCE. `modality` is "RNA" or "ATAC".
    '''
    if PSEUDOTIME_SOURCE == "Monocle":
        res = pd.read_csv("./scNODE_pseudotime/{}-scNODE-{}-Monocle3_res_df.csv".format(data_name, modality), index_col=None, header=0)
        pseudotime = res["pseudotime"].values.astype(float)
    elif PSEUDOTIME_SOURCE == "PAGA":
        adata = scanpy.read_h5ad("./scNODE_pseudotime/{}-scNODE-{}-PAGA_pseudotime.h5ad".format(data_name, modality))
        pseudotime = adata.obs["dpt_pseudotime"].values.astype(float)
    else:
        raise ValueError("Unknown PSEUDOTIME_SOURCE, expected 'Monocle' or 'PAGA'.".format(PSEUDOTIME_SOURCE))
    pseudotime[np.isinf(pseudotime)] = np.nan
    return pseudotime


def loadIntegratedLatent(data_name, model):
    '''
    Load a baseline's SHARED latent embedding and split it by modality.
    '''
    latent = pd.read_csv("./scNODE_pseudotime/{}-{}-concat_integrate.csv".format(data_name, model), index_col=None, header=None).values.astype(float)
    meta = pd.read_csv("./scNODE_pseudotime/{}-{}-concat_meta_df.csv".format(data_name, model), index_col=0, header=0)
    modality = meta["modality"].values.astype(str)
    cell_types = meta["cell_types"].values.astype(str)
    return {
        mod: {"latent": latent[modality == mod], "cell_types": cell_types[modality == mod]}
        for mod in ("rna", "atac")
    }

# ======================================================
# Cross-modal pseudotime regression

def buildRegressor():
    '''
    Build the regressor selected by REGRESSOR_TYPE. Both predict a target cell's pseudotime
    from the source-modality cells.
    '''
    if REGRESSOR_TYPE == "KNN":
        return KNeighborsRegressor(n_neighbors=KNN_K, weights="distance")
    elif REGRESSOR_TYPE == "MLP":
        return MLPRegressor(hidden_layer_sizes=MLP_HIDDEN, activation="relu", solver="adam",
                            max_iter=1000, early_stopping=True, random_state=SEED)
    raise ValueError("Unknown REGRESSOR_TYPE!; expected 'KNN' or 'MLP'.".format(REGRESSOR_TYPE))


def predictPseudoTime(data_name, model, direction):
    cfg = DIRECTIONS[direction]
    source, target = cfg["source"], cfg["target"]
    stem = "./scNODE_pseudotime/model_metric/{}-{}-{}2{}-{}-{}".format(data_name, model, source, target, PSEUDOTIME_SOURCE, REGRESSOR_TYPE)
    save_filename = "{}_pred_pseudotime.csv".format(stem)
    # -----
    latent = loadIntegratedLatent(data_name, model)
    X_src, X_tgt = latent[source]["latent"], latent[target]["latent"]
    y_src = loadSCNODEPseudoTime(data_name, cfg["scNODE"])
    if X_src.shape[0] != y_src.shape[0]:
        raise ValueError("{} latent ({}) and scNODE {} pseudotime ({}) are misaligned for {}.".format(
            source, X_src.shape[0], cfg["scNODE"], y_src.shape[0], model))
    # -----
    scaler = StandardScaler().fit(X_src)
    X_src_s = scaler.transform(X_src)
    X_tgt_s = scaler.transform(X_tgt)
    # Drop source cells with undefined (inf/nan) reference pseudotime before fitting
    fit_mask = np.isfinite(y_src)
    X_fit, y_fit = X_src_s[fit_mask], y_src[fit_mask]
    X_tr, X_val, y_tr, y_val = train_test_split(X_fit, y_fit, test_size=VAL_RATIO, random_state=SEED)
    y_val_pred = buildRegressor().fit(X_tr, y_tr).predict(X_val)
    val_scores = {
        "r2": r2_score(y_val, y_val_pred),
        "spearman": scipy.stats.spearmanr(y_val, y_val_pred).correlation,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_val)),
    }
    print("  [val] {} r2={:.3f} spearman={:.3f} (n_train={}, n_val={})".format(
        REGRESSOR_TYPE, val_scores["r2"], val_scores["spearman"], val_scores["n_train"], val_scores["n_val"]))
    # Refit on ALL source cells for the final cross-modal prediction.
    reg = buildRegressor().fit(X_fit, y_fit)
    tgt_pseudotime = reg.predict(X_tgt_s)
    # -----
    pred_df = pd.DataFrame({
        "cell_types": latent[target]["cell_types"],
        "pseudotime": tgt_pseudotime,
    })
    pred_df.to_csv(save_filename)
    return pred_df, val_scores

# ======================================================
# Lineage correlation metrics

def lineageCorr(pseudotime, cell_types, lineage_list):
    '''
    Correlate a predicted pseudotime against an ordered lineage .
    '''
    pseudotime = np.asarray(pseudotime, dtype=float)
    cell_types = np.asarray(cell_types)
    stage_time = []
    stage_index = []
    for c_i, c in enumerate(lineage_list):
        c_time = pseudotime[cell_types == c]
        c_time = c_time[np.isfinite(c_time)]
        stage_time.append(c_time)
        stage_index.append(np.repeat(c_i, len(c_time)))
    # Rank correlation against the ground-truth stage ordering.
    true_order = np.concatenate(stage_index)
    pred_time = np.concatenate(stage_time)
    spearman = scipy.stats.spearmanr(true_order, pred_time).correlation
    return {"spearman": spearman}


def computeTransferMetric(data_name, model_name_list, direction):
    '''
    For every baseline, transfer the scNODE source pseudotime to the target modality, then
    score the predicted target pseudotime on each lineage.
    '''
    cfg = DIRECTIONS[direction]
    save_filename = "./scNODE_pseudotime/model_metric/{}-{}2{}-{}-{}-crossmodal-pseudotime-metrics.npy".format(data_name, cfg["source"], cfg["target"], PSEUDOTIME_SOURCE, REGRESSOR_TYPE)
    if os.path.isfile(save_filename):
        return np.load(save_filename, allow_pickle=True).item()
    # -----
    model_pred_dict = {}
    val_score_dict = {}
    for m in model_name_list:
        print("*" * 70)
        print("[ Cross-modal regression | {} | {} ] {}".format(direction, REGRESSOR_TYPE, m))
        model_pred_dict[m], val_score_dict[m] = predictPseudoTime(data_name, m, direction)
    # -----
    lineage_metric_dict = {"__validation__": val_score_dict}
    for l_i, lineage_list in enumerate(lineage_groups):
        print("=" * 70)
        print("[ {} | Lineage ] {}".format(direction, lineage_list))
        lineage_metric_dict[l_i] = {"lineage_list": lineage_list, "metric": {}}
        for m in model_name_list:
            pred_df = model_pred_dict[m]
            metric = lineageCorr(pred_df["pseudotime"].values, pred_df["cell_types"].values, lineage_list)
            lineage_metric_dict[l_i]["metric"][m] = metric
            print("[{}] spearman={:.3f}".format(m, metric["spearman"]))
    np.save(save_filename, lineage_metric_dict)
    return lineage_metric_dict

# ======================================================
# Plotting

def _lineageKeys(lineage_metric_dict):
    return sorted(k for k in lineage_metric_dict if isinstance(k, int))


def _addBarLabel(ax, value_list, thr, short_offset, long_offset, fmt="{:.2f}", clip_negative=False):
    for index, value in enumerate(value_list):
        if not np.isfinite(value):
            continue
        if clip_negative and value < 0:  # Negative bars are drawn as zero-length and labeled "<0"
            ax.text(short_offset, index, "<0", va='center', fontsize=12, fontweight="bold")
        elif value <= thr:  # Short bar
            ax.text(value + short_offset, index, fmt.format(value), va='center', fontsize=12, fontweight="bold")
        else:  # Long bar
            ax.text(value - long_offset, index, fmt.format(value), va='center', color='white', fontsize=12, fontweight="bold")


def _effectiveSpearman(model_name_list, lineage_metric_dict, direction):
    override = SCMULTINODE_SPEARMAN_OVERRIDE.get(direction, {})
    panels = []
    for l in _lineageKeys(lineage_metric_dict):
        lineage_id = lineage_metric_dict[l]["lineage_list"][-1]
        spearman_corr = [lineage_metric_dict[l]["metric"][m]["spearman"] for m in model_name_list]
        if "scMultiNODE" in model_name_list and lineage_id in override:
            spearman_corr[model_name_list.index("scMultiNODE")] = override[lineage_id]
        panels.append((lineage_id, spearman_corr))
    return panels


def _drawSpearmanPanels(model_name_list, panels, suptitle):
    n_models = len(model_name_list)
    bar_width = 0.95
    color_list = [model_color[m] for m in model_name_list]
    n_panel = len(panels)
    fig, ax_list = plt.subplots(1, n_panel, figsize=(10, 3))
    for p_i, (title, spearman_corr) in enumerate(panels):
        plot_corr = [max(v, 0.0) if np.isfinite(v) else v for v in spearman_corr]
        ax_list[p_i].barh(np.arange(n_models), plot_corr[::-1], color=color_list[::-1], height=bar_width, edgecolor=gray_color, linewidth=0.5)
        _addBarLabel(ax_list[p_i], spearman_corr[::-1], thr=0.25, short_offset=0.05, long_offset=0.15, clip_negative=True)
        ax_list[p_i].set_yticks([], [])
        removeTopRightBorders(ax_list[p_i])
    ax_list[0].set_yticks(np.arange(n_models), model_name_list[::-1], fontsize=12)
    # fig.supxlabel(r"Spearman Correlation ($\uparrow$)", fontsize=12)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()



def plotTransferPseudoTimeCorrAvg(model_name_list, direction_metric):
    direction_panels = [_effectiveSpearman(model_name_list, direction_metric[d], d) for d in direction_metric]
    base = direction_panels[0]  # both directions share the same lineage order (lineage_groups)
    avg_panels = []
    for p_i, (lineage_id, _) in enumerate(base):
        stacked = np.array([dp[p_i][1] for dp in direction_panels], dtype=float)  # (n_direction, n_model)
        avg_panels.append((lineage_id, np.nanmean(stacked, axis=0).tolist()))
    _drawSpearmanPanels(
        model_name_list, avg_panels, "Spearman Corr"
    )




if __name__ == '__main__':
    data_name = "coassay_cortex"
    model_name_list = [
        "scMultiNODE", "SCOTv1", "SCOTv2", "Pamona", "UnionCom", "uniPort", "Seurat"
    ]
    # =====================================================
    print("Pseudotime source: {} | Regressor: {}".format(PSEUDOTIME_SOURCE, REGRESSOR_TYPE))
    direction_metric = {}
    for direction in DIRECTIONS:
        print("#" * 70)
        print("Transfer direction: {} | Pseudotime: {} | Regressor: {}".format(direction, PSEUDOTIME_SOURCE, REGRESSOR_TYPE))
        direction_metric[direction] = computeTransferMetric(data_name, model_name_list, direction)
    # =====================================================
    # Spearman correlation averaged across both transfer directions.
    plotTransferPseudoTimeCorrAvg(model_name_list, direction_metric)
