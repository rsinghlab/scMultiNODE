'''
Description:
    Evaluate single-modal latent representation from scNODE.

    Loads the pre-computed per-modality scNODE latents from disk and scores each with the same
    metric functions used by the alignment scripts Compare_Modal_Alignment_Metric.py /
    Compare_Modal_Alignment_Clustering.py.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os
import sys

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

# Put the scMultiNODE repo root and this directory on sys.path so the local modules
# (optim / utils / modal_integration and the sibling Compare_Modal_Alignment_* scripts) resolve.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_HERE = os.path.abspath(os.path.dirname(__file__))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from optim.evaluation import labelCorr             # tp_corr (same source the alignment scripts use)
from utils.FileUtils import loadSCData
from modal_integration import DATA_DIR_DICT
import Compare_Modal_Alignment_Clustering as cac    # provides _louvain (metrics via the alignment script)

# =============================================
# Configuration
# =============================================
DATA_NAME = "coassay_cortex"    # coassay_cortex, human_organoid, drosophila, mouse_neocortex
SPACE_NAME = "scNODE"           # label for the latent space being evaluated
DATA_TYPE = "reduce"
SPLIT_TYPE = "all"
LATENT_DIM = 50

EXCLUDE_LABELS = ("unknown", "nan", "none", "")  # dropped from the cell-type clustering

# Precomputed scNODE latent directory
SCNODE_LATENT_DIR = "./res/scNODE_latent"

SUMMARY_METRICS = ["tp_corr", "nmi_global"]
HIGHER_BETTER = {m: True for m in SUMMARY_METRICS}


# =============================================
# Helpers
# =============================================

def _remapUnknown(labels, data_name):
    '''Match Compare_Modal_Alignment_Clustering.py: treat mouse_neocortex "undetermined" as "unknown".'''
    labels = np.asarray(labels).copy()
    if data_name == "mouse_neocortex":
        labels[labels == "undetermined"] = "unknown"
    return labels


def _safeNanmean(values):
    '''np.nanmean that returns NaN (without a RuntimeWarning) for an empty / all-NaN input.'''
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _printMetricTable(metric_dict, title="", higher_better=None):
    '''Pretty-print a {row_name: {metric: value}} table (pandas + tabulate).'''
    import pandas as pd
    df = pd.DataFrame(metric_dict).T
    if higher_better is not None:
        df = df.rename(columns={
            m: "{} {}".format(m, "↑" if higher_better[m] else "↓")
            for m in df.columns if m in higher_better
        })
    if title:
        print("\n" + "=" * 70)
        print(title)
    try:
        import tabulate
        print(tabulate.tabulate(df, headers=["row"] + list(df.columns), tablefmt="grid",
                                floatfmt=".4f"))
    except ImportError:
        print(df.to_string())
    return df


def globalClustering(feature, label, data_name, exclude=EXCLUDE_LABELS):
    '''
    Louvain clustering of the whole latent (reusing Compare_Modal_Alignment_Clustering.py's
    `_louvain`), scored against the cell-type labels with NMI. Well-defined whenever the
    modality has >=2 cell types overall. Returns nmi.
    '''
    feature = np.asarray(feature)
    label = _remapUnknown(label, data_name)
    keep = ~np.isin(label, list(exclude))
    feature, label = feature[keep], label[keep]
    if len(feature) < 3 or len(np.unique(label)) < 2:
        return float("nan")
    pred = cac._louvain(feature)
    return float(normalized_mutual_info_score(label, pred))


def evaluateSingleModality(latent, cell_tp):
    '''Single-modal metric bundle: timepoint distance correlation (labelCorr).'''
    latent = np.asarray(latent)
    cell_tp = np.asarray(cell_tp)
    return {"tp_corr": float(labelCorr(latent, cell_tp))}


# =============================================
# Data / latent loading
# =============================================

def loadData(data_name, data_type=DATA_TYPE, split_type=SPLIT_TYPE):
    (
        ann_rna_data, ann_atac_data, rna_cell_tps, atac_cell_tps,
        rna_n_tps, atac_n_tps, n_genes, n_peaks
    ) = loadSCData(data_name=data_name, data_type=data_type, split_type=split_type,
                   data_dir=DATA_DIR_DICT[data_name])
    rna_cell_types = np.asarray([str(x).lower() for x in ann_rna_data.obs["cell_type"].values])
    atac_cell_types = np.asarray([str(x).lower() for x in ann_atac_data.obs["cell_type"].values])
    rna_traj_cell_type = [rna_cell_types[np.where(rna_cell_tps == t)[0]] for t in range(1, rna_n_tps + 1)]
    atac_traj_cell_type = [atac_cell_types[np.where(atac_cell_tps == t)[0]] for t in range(1, atac_n_tps + 1)]
    rna_traj_len = [np.sum(rna_cell_tps == t) for t in range(1, rna_n_tps + 1)]
    atac_traj_len = [np.sum(atac_cell_tps == t) for t in range(1, atac_n_tps + 1)]
    return {
        "rna_cell_types": np.concatenate(rna_traj_cell_type),
        "atac_cell_types": np.concatenate(atac_traj_cell_type),
        "rna_cell_tps": np.concatenate([np.repeat(t, n) for t, n in enumerate(rna_traj_len)]),
        "atac_cell_tps": np.concatenate([np.repeat(t, n) for t, n in enumerate(atac_traj_len)]),
    }


def loadscNODELatent(data_name, file_dir=SCNODE_LATENT_DIR):
    '''
    Load the per-modality scNODE latents.
    '''
    rna_res = np.load("{}/{}-RNA-scNODE-res.npy".format(file_dir, data_name), allow_pickle=True).item()
    atac_res = np.load("{}/{}-ATAC-scNODE-res.npy".format(file_dir, data_name), allow_pickle=True).item()
    rna_latent = np.concatenate(rna_res["latent_seq"], axis=0)
    atac_latent = np.concatenate(atac_res["latent_seq"], axis=0)
    return np.asarray(rna_latent), np.asarray(atac_latent)


# =============================================
if __name__ == '__main__':
    print("=" * 70)
    print("[ scNODE per-modality latent evaluation | {} ]".format(DATA_NAME).center(70))

    # ---- Data + scNODE latent (per modality, aligned to the loadData cell ordering) ---------
    data = loadData(DATA_NAME)
    scnode_rna, scnode_atac = loadscNODELatent(DATA_NAME)
    modalities = {
        "rna": (np.asarray(scnode_rna), np.asarray(data["rna_cell_types"]), np.asarray(data["rna_cell_tps"])),
        "atac": (np.asarray(scnode_atac), np.asarray(data["atac_cell_types"]), np.asarray(data["atac_cell_tps"])),
    }
    for modality, (latent, ctype, _ctp) in modalities.items():
        if latent.shape[0] != len(ctype):
            raise ValueError(
                "scNODE {} latent has {} rows but the data has {} cells; the scNODE latent must be "
                "row-aligned with loadData(split_type='all').".format(
                    modality.upper(), latent.shape[0], len(ctype)))

    # ---- Evaluate each modality independently ----------------------------------------------
    per_mod = {}
    for modality, (latent, ctype, ctp) in modalities.items():
        print("\n" + "#" * 70)
        print("# [{}] single-modal metrics".format(modality.upper()))
        bundle = evaluateSingleModality(latent, ctp)
        bundle["nmi_global"] = globalClustering(latent, ctype, DATA_NAME)
        per_mod[modality] = bundle

    # ---- Two-modality average ---------------------------------------------------------------
    avg_bundle = {
        m: _safeNanmean([per_mod["rna"][m], per_mod["atac"][m]]) for m in SUMMARY_METRICS}

    # ---- Report: RNA, ATAC, and the two-modality average ------------------------------------
    summary = {
        "{} (RNA)".format(SPACE_NAME): {m: per_mod["rna"][m] for m in SUMMARY_METRICS},
        "{} (ATAC)".format(SPACE_NAME): {m: per_mod["atac"][m] for m in SUMMARY_METRICS},
        "{} (avg)".format(SPACE_NAME): avg_bundle,
    }
    _printMetricTable(
        summary, title="scNODE per-modality single-modal metrics ({})".format(DATA_NAME),
        higher_better=HIGHER_BETTER)
