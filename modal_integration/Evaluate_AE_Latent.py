'''
Description:
    Evaluate single-modal latent representation from AE.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os
import sys
import itertools

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score

# # Put the scMultiNODE repo root and this directory on sys.path so the local modules
# # (optim / utils / modal_integration and the sibling Compare_Modal_Alignment_* scripts) resolve.
# _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# _HERE = os.path.abspath(os.path.dirname(__file__))
# for _p in (_ROOT, _HERE):
#     if _p not in sys.path:
#         sys.path.insert(0, _p)

from optim.evaluation import labelCorr             # tp_corr (same source the alignment scripts use)
from optim.loss_func import MSELoss                 # AE reconstruction loss
from model.layer import LinearNet                   # AE encoder / decoder
from utils.FileUtils import loadSCData
from modal_integration import DATA_DIR_DICT
import Compare_Modal_Alignment_Clustering as cac    # provides _louvain (metrics via the alignment script)

# =============================================
# Configuration
# =============================================
DATA_NAME = "coassay_cortex"    # coassay_cortex, human_organoid, drosophila, mouse_neocortex
SPACE_NAME = "AE"
DATA_TYPE = "reduce"
SPLIT_TYPE = "all"
LATENT_DIM = 50

EXCLUDE_LABELS = ("unknown", "nan", "none", "")  # dropped from the cell-type clustering

# Evaluation metrics
SUMMARY_METRICS = ["tp_corr", "nmi_global"]
HIGHER_BETTER = {m: True for m in SUMMARY_METRICS}

# Where the evaluation result is written (existing house directory).
RES_DIR = "./res/comparison"


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
# Data loading
# =============================================

def loadData(data_name, data_type=DATA_TYPE, split_type=SPLIT_TYPE):
    (
        ann_rna_data, ann_atac_data, rna_cell_tps, atac_cell_tps,
        rna_n_tps, atac_n_tps, n_genes, n_peaks
    ) = loadSCData(data_name=data_name, data_type=data_type, split_type=split_type,
                   data_dir=DATA_DIR_DICT[data_name])
    rna_cnt = ann_rna_data.X
    atac_cnt = ann_atac_data.X
    # Cell types (lower-cased, following the house convention).
    rna_cell_types = np.asarray([str(x).lower() for x in ann_rna_data.obs["cell_type"].values])
    atac_cell_types = np.asarray([str(x).lower() for x in ann_atac_data.obs["cell_type"].values])
    rna_traj_cell_type = [rna_cell_types[np.where(rna_cell_tps == t)[0]] for t in range(1, rna_n_tps + 1)]
    atac_traj_cell_type = [atac_cell_types[np.where(atac_cell_tps == t)[0]] for t in range(1, atac_n_tps + 1)]
    # Per-timepoint count matrices.
    rna_traj_data = [rna_cnt[np.where(rna_cell_tps == t)[0], :] for t in range(1, rna_n_tps + 1)]
    atac_traj_data = [atac_cnt[np.where(atac_cell_tps == t)[0], :] for t in range(1, atac_n_tps + 1)]
    # Flat arrays aligned with the concatenated-over-timepoints latent ordering.
    return {
        "rna_traj_data": rna_traj_data, "atac_traj_data": atac_traj_data,
        "rna_cell_types": np.concatenate(rna_traj_cell_type),
        "atac_cell_types": np.concatenate(atac_traj_cell_type),
        "rna_cell_tps": np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(rna_traj_data)]),
        "atac_cell_tps": np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(atac_traj_data)]),
        "n_genes": n_genes, "n_peaks": n_peaks,
    }


# =============================================
# Single-modal AE
# =============================================

def trainModalityEncoder(traj_data, input_dim, latent_dim=LATENT_DIM,
                         enc_latent=(50,), dec_latent=(50,), act_name="relu",
                         ae_iters=1000, ae_lr=1e-3, ae_batch_size=128, name="modality", seed=0):
    '''
    Train a single-modality AE.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    all_data = torch.cat([torch.FloatTensor(np.asarray(x)) for x in traj_data], dim=0)
    enc = LinearNet(input_dim=input_dim, latent_size_list=list(enc_latent), output_dim=latent_dim, act_name=act_name)
    dec = LinearNet(input_dim=latent_dim, latent_size_list=list(dec_latent), output_dim=input_dim, act_name=act_name)
    optimizer = torch.optim.Adam(
        params=itertools.chain(enc.parameters(), dec.parameters()), lr=ae_lr, betas=(0.95, 0.99))
    enc.train(); dec.train()
    pbar = tqdm(range(ae_iters), desc="[ Single-Modal AE / {} ]".format(name))
    for _ in pbar:
        optimizer.zero_grad()
        batch_idx = np.random.choice(all_data.shape[0], min(ae_batch_size, all_data.shape[0]), replace=False)
        batch = all_data[batch_idx, :]
        recon = dec(enc(batch))
        recon_loss = MSELoss(batch, recon)
        pbar.set_postfix({"Loss": "{:.4f}".format(recon_loss)})
        recon_loss.backward()
        optimizer.step()
    enc.eval()
    return enc


def encodeSingleModal(enc, data):
    enc.eval()
    with torch.no_grad():
        return enc(torch.FloatTensor(np.asarray(data))).numpy()


# =============================================
if __name__ == '__main__':
    print("=" * 70)
    print("[ AE per-modality latent evaluation | {} ]".format(DATA_NAME).center(70))

    # ---- Data + AE latent (per modality, aligned to the loadData cell ordering) --------------
    data = loadData(DATA_NAME)
    ae_specs = {
        "rna": (data["rna_traj_data"], data["n_genes"]),
        "atac": (data["atac_traj_data"], data["n_peaks"]),
    }
    ae_latent = {}
    for modality, (traj_data, input_dim) in ae_specs.items():
        enc = trainModalityEncoder(traj_data, input_dim, name=modality)
        all_data = np.concatenate([np.asarray(x) for x in traj_data], axis=0)
        ae_latent[modality] = encodeSingleModal(enc, all_data)
    # -----
    modalities = {
        "rna": (np.asarray(ae_latent["rna"]), np.asarray(data["rna_cell_types"]), np.asarray(data["rna_cell_tps"])),
        "atac": (np.asarray(ae_latent["atac"]), np.asarray(data["atac_cell_types"]), np.asarray(data["atac_cell_tps"])),
    }

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
        summary, title="AE per-modality single-modal metrics ({})".format(DATA_NAME),
        higher_better=HIGHER_BETTER)
