'''
Description:
    Compare modality integration with quantitative metrics.


Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os.path
import numpy as np
import pandas as pd
import tabulate
from utils.FileUtils import loadSCData, tpSplitInd, getTimeStr, loadIntegratedLatent
from optim.evaluation import (
    neighborOverlap, FOSCTTM, transferAccuracy, featureCorrPerSample, batchEntropy, labelCorr
)
from modal_integration import DATA_DIR_DICT

# ================================================

def _evaluateCoassayAlignment(x1_feature, x2_feature, x1_label, x2_label, x1_tp, x2_tp, data_name):
    true_target_sample = np.arange(x2_feature.shape[0])
    # -----
    n_neighbors = 50
    print("-" * 70)
    print("Compute transfer accuracy...")
    transfer_acc_type = transferAccuracy(x1_feature, x2_feature, x1_label, x2_label, n=n_neighbors)
    transfer_acc_tp = transferAccuracy(x1_feature, x2_feature, x1_tp, x2_tp, n=n_neighbors)
    print("LTA-type = {:.4f} | LTA-time = {:.4f}".format(transfer_acc_type, transfer_acc_tp))
    # -----
    print("-" * 70)
    print("Compute neighbor overlap...")
    neigh_overlap = neighborOverlap(x1_feature, x2_feature, true_target_sample, n_neighbors=50)
    print("neighbor overlap = {:.4f}".format(neigh_overlap))
    # -----
    print("-" * 70)
    print("Compute correlation...")
    corr_score_sample = featureCorrPerSample(x1_feature, x2_feature)
    print("SCC = {:.4f}".format(corr_score_sample))
    # -----
    print("-" * 70)
    print("Compute FOSCTTM...")
    foscttm = FOSCTTM(x1_feature, x2_feature)
    print("foscttm = {:.4f}".format(foscttm))
    # -----
    print("-" * 70)
    print("Compute batch entropy...")
    batch_entropy = batchEntropy(
        np.concatenate([x1_feature, x2_feature], axis=0),
        np.concatenate([np.repeat(0, x1_feature.shape[0]), np.repeat(1, x2_feature.shape[0])], axis=0)
    )
    print("batch entropy = {:.4f}".format(batch_entropy))
    # -----
    print("-" * 70)
    print("Compute time correlation...")
    rna_tp_corr = labelCorr(
        x1_feature, x1_tp
    )
    atac_tp_corr = labelCorr(
        x2_feature, x2_tp
    )
    tp_corr = (rna_tp_corr + atac_tp_corr) / 2.0
    print("time correlation = {:.4f}".format(tp_corr))
    # -----
    return {
        "transfer_acc_type": transfer_acc_type,
        "transfer_acc_tp": transfer_acc_tp,
        "neigh_overlap": neigh_overlap,
        "foscttm": foscttm,
        "batch_entropy": batch_entropy,
        "corr_score_sample": corr_score_sample,
        "tp_corr": tp_corr,
    }


def _evaluateUnalignedAlignment(x1_feature, x2_feature, x1_label, x2_label, x1_tp, x2_tp, data_name):
    common_cell_types = np.intersect1d(np.unique(x1_label), np.unique(x2_label))
    if data_name == "drosophila":
        # DR dataset has lots of cell types, we use five cell types with the largest number of cells for metrics computation
        selected_cell_type = ["brain", "epidermis", "germ cell", "somatic muscle", "amnioserosa"]
    elif data_name == "human_cortex":
        selected_cell_type = common_cell_types
    elif data_name == "mouse_neocortex":
        selected_cell_type = common_cell_types
    elif data_name == "zebrahub":
        selected_cell_type = common_cell_types
    elif data_name == "amphioxus":
        selected_cell_type = common_cell_types
    else:
        raise ValueError("The cell types are not selected.")
    x1_idx = [i for i, x in enumerate(x1_label) if x in selected_cell_type and x != "unknown"]
    x2_idx = [i for i, x in enumerate(x2_label) if x in selected_cell_type and x != "unknown"]
    # -----
    n_neighbors = 50
    print("-" * 70)
    print("Compute transfer accuracy...")
    transfer_acc_type = transferAccuracy(x1_feature[x1_idx], x2_feature[x2_idx], x1_label[x1_idx], x2_label[x2_idx], n=n_neighbors)
    transfer_acc_tp = transferAccuracy(x1_feature, x2_feature, x1_tp, x2_tp, n=n_neighbors)
    print("LTA-type = {:.4f} | LTA-time = {:.4f}".format(transfer_acc_type, transfer_acc_tp))
    # -----
    print("-" * 70)
    print("Compute batch entropy...")
    batch_entropy = batchEntropy(
        np.concatenate([x1_feature, x2_feature], axis=0),
        np.concatenate([np.repeat(0, x1_feature.shape[0]), np.repeat(1, x2_feature.shape[0])], axis=0)
    )
    print("batch entropy = {:.4f}".format(batch_entropy))
    # -----
    print("-" * 70)
    print("Compute time correlation...")
    rna_tp_corr = labelCorr(
        x1_feature, x1_tp
    )
    atac_tp_corr = labelCorr(
        x2_feature, x2_tp
    )
    tp_corr = (rna_tp_corr + atac_tp_corr) / 2.0
    print("time correlation = {:.4f}".format(tp_corr))
    # -----
    return {
        "transfer_acc_type": transfer_acc_type,
        "transfer_acc_tp": transfer_acc_tp,
        "batch_entropy": batch_entropy,
        "tp_corr": tp_corr,
    }


def computeAlignmentScore(integrated_dict, rna_cell_types, atac_cell_types, rna_cell_tps, atac_cell_tps, data_name):
    if data_name in ["coassay_cortex", "human_organoid"]: # co-assay data
        _func = _evaluateCoassayAlignment
    else: # unaligned data
        _func = _evaluateUnalignedAlignment
    model_metric = {m: {} for m in integrated_dict}
    for m in integrated_dict:
        print("*" * 70)
        print("[ {} ] Computing metrics...".format(m))
        m_metric = _func(
            integrated_dict[m]["rna"], integrated_dict[m]["atac"],
            rna_cell_types, atac_cell_types, rna_cell_tps, atac_cell_tps, data_name
        )
        model_metric[m] = m_metric
    return model_metric


# ================================================


def _plotTable(df_data):
    print(tabulate.tabulate(
        df_data, headers=[df_data.index.names[0]] + list(df_data.columns),
        tablefmt="grid"
    ))


def plotMetric(model_metric, data_name):
    if data_name in ["coassay_cortex", "human_organoid"]: # co-assay data
        column_names = [
            "foscttm", "transfer_acc_type", "transfer_acc_tp", "neigh_overlap", "batch_entropy", "tp_corr", "corr_score_sample"
        ]
        column_name_ind = [
            "FOSCTTM ↓", "LTA-type ↑", "LTA-time ↑", "Neighbor Overlap ↑", "Batch Entropy ↑", "Time Corr. ↑", "SCC ↑"
        ]
    else: # unaligned data
        column_names = [
            "transfer_acc_type", "transfer_acc_tp", "batch_entropy", "tp_corr"
        ]
        column_name_ind = [
            "LTA-type ↑", "LTA-time ↑", "Batch Entropy ↑", "Time Corr. ↑"
        ]
    # -----
    model_list = list(model_metric.keys())
    metric_list = [[model_metric[m][i] for i in column_names] for m in model_list]
    print("\n" * 2)
    metric_df = pd.DataFrame(
        data=metric_list,
        index=pd.MultiIndex.from_tuples([
            (m,) for m in model_list
        ], names=("Model",)),
        columns=pd.MultiIndex.from_tuples([(x,) for x in column_name_ind], names=("Metrics", ))
    )
    _plotTable(metric_df)


if __name__ == '__main__':

    # ================================================
    latent_dim = 50
    output_dim = latent_dim
    model_list = ["scMultiNODE", "SCOTv1", "SCOTv2", "Pamona", "UnionCom", "uniPort", "Seurat"]
    data_name = "amphioxus"  # coassay_cortex, human_organoid, drosophila, mouse_neocortex, zebrahub, amphioxus
    split_type = "all"
    data_type = "reduce"
    # -----
    save_filename = "./res/comparison/{}-{}-{}-ae-{}dim-metric.npy".format(data_name, data_type, split_type, latent_dim)
    if not os.path.isfile(save_filename):
        # Loading data
        time_str = getTimeStr()
        (
            ann_rna_data, ann_atac_data, rna_cell_tps, atac_cell_tps,
            rna_n_tps, atac_n_tps, n_genes, n_peaks
        ) = loadSCData(data_name=data_name, data_type=data_type, split_type=split_type, data_dir=DATA_DIR_DICT[data_name])
        rna_train_tps, atac_train_tps, rna_test_tps, atac_test_tps = tpSplitInd(data_name, split_type)
        rna_cnt = ann_rna_data.X
        atac_cnt = ann_atac_data.X
        # cell type
        rna_cell_types = np.asarray([str(x).lower() for x in ann_rna_data.obs["cell_type"].values])
        atac_cell_types = np.asarray([str(x).lower() for x in ann_atac_data.obs["cell_type"].values])
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
        # ================================================
        rna_cell_types, atac_cell_types = np.concatenate(rna_traj_cell_type), np.concatenate(atac_traj_cell_type)
        rna_cell_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(rna_traj_data)])
        atac_cell_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(atac_traj_data)])
        # ================================================
        integrated_dict = loadIntegratedLatent(data_name, data_type, split_type, model_list, latent_dim)
        model_metric = computeAlignmentScore(integrated_dict, rna_cell_types, atac_cell_types, rna_cell_tps, atac_cell_tps, data_name)
        print("Saving res to {}".format(save_filename))
        np.save(save_filename, model_metric)
    else:
        print("Metrics already computed, loading file...")
        model_metric = np.load(save_filename, allow_pickle=True).item()
    # ================================================
    plotMetric(model_metric, data_name)

