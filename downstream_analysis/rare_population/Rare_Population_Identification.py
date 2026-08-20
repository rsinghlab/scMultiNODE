'''
Description:
    Rare population identification.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os
import sys
import types
import itertools

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import silhouette_samples

from utils.FileUtils import loadSCData, loadIntegratedLatent
from modal_integration import DATA_DIR_DICT
from model.layer import LinearNet
from optim.loss_func import MSELoss
from plotting.__init__ import *

import baseline.Pamona.Pamona as Pamona
from baseline.UnionCom import UnionCom

# =============================================
# Inferred global constants (previously imported from eval_utils)
# =============================================
DATA_TYPE = "reduce"
SPLIT_TYPE = "all"
JOINT_MODEL = "scMultiNODE"
LATENT_DIM = 50

EXCLUDE_LABELS = ("unknown", "nan", "none", "")   # cell-type labels dropped from separability
SWEEP_METRICS = ["silhouette"]

# input directories
MODEL_LATENT_DIR = "../../modal_integration/res/model_latent"
SCNODE_LATENT_DIR = "../../modal_integration/res/scNODE_latent"
# output directory
RESULT_DIR = "./res"

# =============================================
# Configuration
# =============================================
BASELINES = ["SCOTv2", "SCOTv1", "Pamona", "UnionCom", "uniPort", "Seurat"]
DATASETS = ["coassay_cortex", "human_organoid", "drosophila", "mouse_neocortex"]
MODALITIES = ["rna", "atac"]

DATASET_DISPLAY = {
    "coassay_cortex": "HC",
    "human_organoid": "HO",
    "drosophila": "DR",
    "mouse_neocortex": "MN",
    "zebrahub": "ZB",
    "amphioxus": "AM",
}

# The strongest integration baseline per dataset
BEST_BASELINE = {
    "coassay_cortex": "SCOTv2",
    "human_organoid": "UnionCom",
    "drosophila": "Pamona",
    "mouse_neocortex": "SCOTv2",
    "zebrahub": "SCOTv1",
    "amphioxus": "Seurat",
}

# Number of rare cell types
DEFAULT_N_RARE = 3
N_RARE = {"human_organoid": 1}
XLIM = None

_GRAY = (173 / 255, 181 / 255, 189 / 255)


# =============================================
# Data / latent loading
# =============================================
def _remapUnknown(labels, data_name):
    '''Match Compare_Modal_Alignment_Clustering.py: treat mouse_neocortex "undetermined" as "unknown".'''
    labels = np.asarray(labels).copy()
    if data_name == "mouse_neocortex":
        labels[labels == "undetermined"] = "unknown"
    return labels


def loadEvalData(data_name):
    (
        ann_rna_data, ann_atac_data, rna_cell_tps, atac_cell_tps,
        rna_n_tps, atac_n_tps, n_genes, n_peaks
    ) = loadSCData(data_name=data_name, data_type=DATA_TYPE, split_type=SPLIT_TYPE, data_dir="../" + DATA_DIR_DICT[data_name])
    rna_cnt = ann_rna_data.X
    atac_cnt = ann_atac_data.X
    # Cell types (lower-cased, following the house convention).
    rna_cell_types = _remapUnknown([str(x).lower() for x in ann_rna_data.obs["cell_type"].values], data_name)
    atac_cell_types = _remapUnknown([str(x).lower() for x in ann_atac_data.obs["cell_type"].values], data_name)
    rna_traj_cell_type = [rna_cell_types[np.where(rna_cell_tps == t)[0]] for t in range(1, rna_n_tps + 1)]
    atac_traj_cell_type = [atac_cell_types[np.where(atac_cell_tps == t)[0]] for t in range(1, atac_n_tps + 1)]
    # Per-timepoint count matrices.
    rna_traj_data = [rna_cnt[np.where(rna_cell_tps == t)[0], :] for t in range(1, rna_n_tps + 1)]
    atac_traj_data = [atac_cnt[np.where(atac_cell_tps == t)[0], :] for t in range(1, atac_n_tps + 1)]
    return {
        "rna_traj_data": rna_traj_data, "atac_traj_data": atac_traj_data,
        "rna_cell_types": np.concatenate(rna_traj_cell_type),
        "atac_cell_types": np.concatenate(atac_traj_cell_type),
        "rna_cell_tps": np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(rna_traj_data)]),
        "atac_cell_tps": np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(atac_traj_data)]),
        "n_genes": n_genes, "n_peaks": n_peaks,
    }


# The Pamona / UnionCom integrated-latent .npy files pickled their fitted model object.
_STALE_MODEL_ALIASES = [
    ("modal_integration.Pamona.Pamona", Pamona),
    ("modal_integration.UnionCom.UnionCom", UnionCom),
]
def _aliasBaselineModules():
    for modname, realmod in _STALE_MODEL_ALIASES:
        if modname in sys.modules:
            continue
        # Ensure every intermediate package (e.g. modal_integration.Pamona) exists.
        parts = modname.split(".")
        for i in range(1, len(parts)):
            pkgname = ".".join(parts[:i])
            if pkgname not in sys.modules:
                pkg = types.ModuleType(pkgname)
                pkg.__path__ = []
                sys.modules[pkgname] = pkg
        sys.modules[modname] = realmod
        setattr(sys.modules[modname.rsplit(".", 1)[0]], parts[-1], realmod)



_MODAL_DIR = os.path.dirname(os.path.dirname(MODEL_LATENT_DIR))
def _loadIntegratedLatent(data_name):
    '''
    Load the joint (scMultiNODE) and baseline integrated latents via
    utils.FileUtils.loadIntegratedLatent. That helper reads from a CWD-relative ./res/model_latent,
    so the call runs with the working directory switched to the modal_integration directory.
    Returns {model: {"rna": latent, "atac": latent}}.
    '''
    _aliasBaselineModules()
    model_list = [JOINT_MODEL] + BASELINES
    cwd = os.getcwd()
    try:
        os.chdir(_MODAL_DIR)
        return loadIntegratedLatent(data_name, DATA_TYPE, SPLIT_TYPE, model_list, LATENT_DIM)
    finally:
        os.chdir(cwd)


def _loadscNODELatent(data_name):
    rna_res = np.load(os.path.join(SCNODE_LATENT_DIR, "{}-RNA-scNODE-res.npy".format(data_name)),
                      allow_pickle=True).item()
    atac_res = np.load(os.path.join(SCNODE_LATENT_DIR, "{}-ATAC-scNODE-res.npy".format(data_name)),
                       allow_pickle=True).item()
    rna_latent = np.concatenate(rna_res["latent_seq"], axis=0)
    atac_latent = np.concatenate(atac_res["latent_seq"], axis=0)
    return np.asarray(rna_latent), np.asarray(atac_latent)


# =============================================
# Single-modal AE
# =============================================
def trainModalityEncoder(traj_data, input_dim, latent_dim=LATENT_DIM,
                         enc_latent=(50,), dec_latent=(50,), act_name="relu",
                         ae_iters=1000, ae_lr=1e-3, ae_batch_size=128, name="modality", seed=0):
    '''Train a single-modality AE and return its encoder.'''
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


def buildLatentSpaces(data, data_name, include_scnode=True):
    integrated = _loadIntegratedLatent(data_name)
    # Single-modal AE latent (trained per modality, encoded on the concatenated data).
    ae_specs = {
        "rna": (data["rna_traj_data"], data["n_genes"]),
        "atac": (data["atac_traj_data"], data["n_peaks"]),
    }
    ae_latent = {}
    for modality, (traj_data, input_dim) in ae_specs.items():
        enc = trainModalityEncoder(traj_data, input_dim, name=modality)
        all_data = np.concatenate([np.asarray(x) for x in traj_data], axis=0)
        ae_latent[modality] = encodeSingleModal(enc, all_data)
    # Single-modal scNODE latent.
    scnode_latent = {}
    if include_scnode:
        scnode_rna, scnode_atac = _loadscNODELatent(data_name)
        scnode_latent = {"rna": scnode_rna, "atac": scnode_atac}
    # -----
    spaces = {}
    for modality in MODALITIES:
        m_spaces = {"Joint": np.asarray(integrated[JOINT_MODEL][modality])}
        for baseline in BASELINES:
            m_spaces[baseline] = np.asarray(integrated[baseline][modality])
        m_spaces["Single"] = np.asarray(ae_latent[modality])
        if include_scnode:
            m_spaces["scNODE"] = np.asarray(scnode_latent[modality])
        spaces[modality] = m_spaces
    return spaces


# =============================================
# Metrics
# =============================================
def groupSilhouette(latent, labels, ct):
    '''
    One-vs-rest silhouette for cell type `ct`: excluded labels are dropped, the remaining cells are
    partitioned into `ct` vs. the rest, and the mean silhouette over the `ct` cells is returned.
    '''
    latent = np.asarray(latent)
    labels = np.asarray(labels)
    keep = ~np.isin(labels, list(EXCLUDE_LABELS))
    X, lab = latent[keep], labels[keep]
    if X.shape[0] < 3:
        return float("nan")
    binary = (lab == ct).astype(int)
    n_ct = int(binary.sum())
    if n_ct == 0 or n_ct == len(binary):
        return float("nan")
    sil = silhouette_samples(X, binary)
    return float(np.mean(sil[binary == 1]))


def _perTypeSeparability(latent, labels, cell_types):
    '''One-vs-rest metric bundle for every cell type on one latent space.'''
    labels = np.asarray(labels)
    out = {}
    for ct in cell_types:
        out[ct] = {"silhouette": groupSilhouette(latent, labels, ct)}
    return out



def computeSeparability(data_name):
    '''
    Compute the per-cell-type separability table for one dataset across all latent spaces and both modalities.
    '''
    print("=" * 70)
    print("[ Rare-Population Detection | {} ]".format(data_name).center(70))
    data = loadEvalData(data_name)
    # include_scnode=True adds the per-modality scNODE latent as a second single-modal reference.
    spaces = buildLatentSpaces(data, data_name, include_scnode=True)
    # -----
    sep_curve = {"rna": [], "atac": []}
    for modality in MODALITIES:
        m_labels = np.asarray(data["{}_cell_types".format(modality)])
        m_uniq, m_cnt = np.unique(m_labels, return_counts=True)
        valid = [(u, c) for u, c in zip(m_uniq, m_cnt) if u not in EXCLUDE_LABELS]
        count_of = dict(zip(m_uniq, m_cnt))
        cts = [u for u, _ in valid]
        # Per space, all cell types at once.
        per_space = {space_name: _perTypeSeparability(latent, m_labels, cts)
                     for space_name, latent in spaces[modality].items()}
        for ct in cts:
            bundles = {space_name: per_space[space_name][ct] for space_name in spaces[modality]}
            sep_curve[modality].append((ct, int(count_of.get(ct, 0)), bundles))
    return {"separability": sep_curve}


# =============================================
# Extraction
# =============================================
RESULTS = {}  # {dataset: {"separability": ...}}


def _loadResult(dataset):
    return RESULTS.get(dataset)


def _rareEntries(res, modality, n_rare):
    entries = [e for e in res.get("separability", {}).get(modality, [])
               if e[0] not in EXCLUDE_LABELS]
    return sorted(entries, key=lambda e: e[1])[:n_rare]


def _silhouetteBySpace(res, modality, n_rare):
    '''
    One-vs-rest silhouette per latent space, averaged over the `n_rare` least-abundant cell
    types of `modality`.
    '''
    acc = {}
    for _, _, bundles in _rareEntries(res, modality, n_rare):
        for space, metrics in bundles.items():
            acc.setdefault(space, []).append(metrics.get("silhouette", np.nan))
    return {space: float(np.nanmean(vals)) for space, vals in acc.items()}


def _rareCounts(res, modality, n_rare):
    entries = [e for e in res.get("separability", {}).get(modality, [])
               if e[0] not in EXCLUDE_LABELS]
    if not entries:
        return [], 0, 0
    rare = sorted(entries, key=lambda e: e[1])[:n_rare]
    return rare, sum(e[1] for e in rare), sum(e[1] for e in entries)


def _rareFraction(res, modality, n_rare):
    _, n_rare_cells, n_all = _rareCounts(res, modality, n_rare)
    return 100.0 * n_rare_cells / n_all if n_all else float("nan")


def printRarePopulationStats():
    '''Print how many rare cell types were used and what fraction of the full cell set they make up.'''
    rows = []
    for ds in DATASETS:
        res = _loadResult(ds)
        if res is None:
            continue
        n_rare = N_RARE.get(ds, DEFAULT_N_RARE)
        for modality in MODALITIES:
            rare, n_rare_cells, n_all = _rareCounts(res, modality, n_rare)
            if not rare:
                continue
            pct = 100.0 * n_rare_cells / n_all if n_all else float("nan")
            rows.append([DATASET_DISPLAY.get(ds, ds), modality.upper(),
                         len(rare), n_rare_cells, n_all, "{:.2f}%".format(pct)])

    headers = ["dataset", "mod", "#rare types", "#rare cells", "#all cells", "%rare"]
    print("\n" + "=" * 70)
    print("Rare-population summary (rare types = least-abundant cell types evaluated)")
    if not rows:
        print("(no rare-population results found)")
        return
    try:
        import tabulate
        print(tabulate.tabulate(rows, headers=headers, tablefmt="grid"))
    except ImportError:
        print("  ".join(headers))
        for r in rows:
            print("  ".join(str(c) for c in r))


def _barsForPanel(dataset, modality, res):
    '''
    The four bars (top-to-bottom: scMultiNODE, best integration baseline, scNODE, AE) for one
    panel. The six baselines are collapsed to the single strongest one (BEST_BASELINE), drawn in
    gray as "Best Integ.". Returns (display_label, color, silhouette_value) top-to-bottom.
    '''
    sil = _silhouetteBySpace(res, modality, N_RARE.get(dataset, DEFAULT_N_RARE))
    baseline = BEST_BASELINE.get(dataset)
    return [
        ("scMultiNODE", model_color["scMultiNODE"], sil.get("Joint", np.nan)),
        ("Best Integ.", _GRAY, sil.get(baseline, np.nan)),
        ("scNODE", model_color["scNODE"], sil.get("scNODE", np.nan)),
        ("AE", model_color["AE"], sil.get("Single", np.nan)),
    ]


# =============================================
# Plotting
# =============================================
def _panelUpper(bars):
    vals = [v for _, _, v in bars if np.isfinite(v)]
    vmax = max(vals) if vals else 0.1
    return max(0.1, np.ceil(vmax * 10.0) / 10.0)


def _drawPanel(ax, bars, show_ylabels):
    n = len(bars)
    y = np.arange(n)[::-1]
    labels = [b[0] for b in bars]
    upper = XLIM[1] if XLIM is not None else _panelUpper(bars)
    label_thresh = 0.3
    text_fontsize = 10
    for yi, (label, color, val) in zip(y, bars):
        width = max(val, 0.0) if np.isfinite(val) else 0.0
        ax.barh(yi, width, height=0.85, color=color, zorder=3)
        if not np.isfinite(val):
            ax.text(0.01 * upper, yi, "n/a", va="center", ha="left", fontsize=text_fontsize, color=_GRAY)
        elif val < 0:
            ax.text(0.01 * upper, yi, "< 0", va="center", ha="left", fontsize=text_fontsize, color="black")
        elif width >= label_thresh:
            ax.text(width - 0.02 * upper, yi, "{:.2f}".format(val), va="center", ha="right",
                    fontsize=text_fontsize, color="white", fontweight="bold", zorder=4)
        else:
            ax.text(width + 0.02 * upper, yi, "{:.2f}".format(val), va="center", ha="left",
                    fontsize=text_fontsize, color="black", fontweight="bold", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_ylabels else [], fontsize=15)
    ax.set_xlim(XLIM if XLIM is not None else (0.0, upper))
    lo, hi = ax.get_xlim()
    ax.set_xticks([lo, hi])
    ax.tick_params(axis="x", labelsize=15)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plotModality(modality):
    results = {ds: res for ds in DATASETS for res in [_loadResult(ds)] if res is not None}
    panels = {ds: _barsForPanel(ds, modality, res) for ds, res in results.items()}
    if not panels:
        print("[plotModality] no {} results found; nothing to plot.".format(modality.upper()))
        return
    first_drawn = next((ds for ds in DATASETS if ds in panels), None)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.0, 2.5), squeeze=False)
    for j, ds in enumerate(DATASETS):
        ax = axes[0][j]
        ax.set_title(DATASET_DISPLAY[ds], fontweight="bold", pad=16)
        if ds in results:
            pct = _rareFraction(results[ds], modality, N_RARE.get(ds, DEFAULT_N_RARE))
            ax.text(0.5, 1.02, "(rare={:.1f}%)".format(pct), ha="center", va="bottom",
                    fontsize=12, fontweight="normal", transform=ax.transAxes)
        if ds not in panels:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color=_GRAY,
                    transform=ax.transAxes)
            continue
        _drawPanel(ax, panels[ds], show_ylabels=(ds == first_drawn))
    axes[0][len(DATASETS) // 2].set_xlabel(
        r"{} / Rare Population Silhouette ($\uparrow$)".format(modality.upper()), fontsize=15)
    fig.tight_layout()
    plt.show()


# =============================================
if __name__ == '__main__':
    os.makedirs(RESULT_DIR, exist_ok=True)
    for ds in DATASETS:
        RESULTS[ds] = computeSeparability(ds)
        out = "{}/{}-{}-{}-{}-rare_population.npy".format(RESULT_DIR, ds, DATA_TYPE, SPLIT_TYPE, JOINT_MODEL)
        np.save(out, RESULTS[ds])
        print("Saved {}".format(out))
    # -----
    printRarePopulationStats()
    for modality in MODALITIES:
        plotModality(modality)
