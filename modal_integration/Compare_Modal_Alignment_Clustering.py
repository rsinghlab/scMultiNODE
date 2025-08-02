'''
Description:
    Compare preservation of cell type in method's integration with Noralized Mutual Information (NMI).

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os.path
import numpy as np
import pandas as pd
import tabulate
import scanpy
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from utils.FileUtils import loadSCData, tpSplitInd, getTimeStr, loadIntegratedLatent
from optim.evaluation import (
    neighborOverlap, FOSCTTM, transferAccuracy, featureCorrPerSample, batchEntropy, labelCorr
)
from modal_integration import DATA_DIR_DICT

# ================================================

def _louvain(x):
    '''Louvain clustering.'''
    x_ann = scanpy.AnnData(X=x)
    scanpy.pp.neighbors(x_ann)
    scanpy.tl.louvain(x_ann)
    cluster_id = x_ann.obs["louvain"]
    cluster_id = np.asarray(cluster_id.values, dtype=int)
    return cluster_id


def _evaluateClustering(x1_feature, x2_feature, x1_label, x2_label):
    if "unknown" in x1_label:
        x1_ind = np.where(x1_label != "unknown")[0]
        x1_feature = x1_feature[x1_ind, :]
        x1_label = x1_label[x1_ind]
    if "unknown" in x2_label:
        x2_ind = np.where(x2_label != "unknown")[0]
        x2_feature = x2_feature[x2_ind, :]
        x2_label = x2_label[x2_ind]
    # -----
    # NMI for RNA cells in the integration
    x1_type_pred = _louvain(x1_feature)
    x1_nmi_type = normalized_mutual_info_score(x1_label, x1_type_pred)
    # NMI for ATAC cells in the integration
    x2_type_pred = _louvain(x2_feature)
    x2_nmi_type = normalized_mutual_info_score(x2_label, x2_type_pred)
    # Average as final output
    avg_nmi_type = (x1_nmi_type + x2_nmi_type) / 2.0
    # -----
    return {
        "avg_nmi_type": avg_nmi_type,
        "x1_nmi_type": x1_nmi_type,
        "x2_nmi_type": x2_nmi_type,
    }


def computeClusteringScore(integrated_dict, rna_cell_types, atac_cell_types, rna_cell_tps, atac_cell_tps):
    model_metric = {m: {} for m in integrated_dict}
    for m in integrated_dict:
        print("*" * 70)
        print("[ {} ] Computing clustering metrics...".format(m))
        m_metric = _evaluateClustering(
            integrated_dict[m]["rna"], integrated_dict[m]["atac"],
            rna_cell_types, atac_cell_types
        )
        model_metric[m] = m_metric
    return model_metric

# ================================================

def _plotTable(df_data):
    print(tabulate.tabulate(
        df_data, headers=[df_data.index.names[0]] + list(df_data.columns),
        tablefmt="grid"
    ))


def plotMetric(model_metric):
    column_names = [
        "avg_nmi_type", "x1_nmi_type", "x2_nmi_type"
    ]
    model_list = list(model_metric.keys())
    column_name_ind = [
        "NMI ↑", "NMI/RNA ↑", "NMI/ATAC ↑"
    ]
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
    # Loading data
    data_name = "zebrahub"  # coassay_cortex, human_organoid, drosophila, mouse_neocortex, zebrahub, amphioxus
    split_type = "all"
    data_type = "reduce"  # reduce, full
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
    if data_name == "mouse_neocortex":
        rna_cell_types[np.where(rna_cell_types == "undetermined")[0]] = "unknown"
        atac_cell_types[np.where(atac_cell_types == "undetermined")[0]] = "unknown"
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
    model_list = ["scMultiNODE", "SCOTv2", "SCOTv1", "Pamona", "UnionCom", "uniPort", "Seurat"]
    latent_dim = 50
    # -----
    save_filename = "./res/clustering/{}-{}-{}-ae-{}dim-clustering.npy".format(data_name, data_type, split_type, latent_dim)
    if not os.path.isfile(save_filename):
        integrated_dict = loadIntegratedLatent(data_name, data_type, split_type, model_list, latent_dim)
        model_metric = computeClusteringScore(integrated_dict, rna_cell_types, atac_cell_types, rna_cell_tps, atac_cell_tps)
        print("Saving res to {}".format(save_filename))
        np.save(save_filename, model_metric)
    else:
        print("Metrics already computed, loading file...")
        model_metric = np.load(save_filename, allow_pickle=True).item()
    plotMetric(model_metric)




