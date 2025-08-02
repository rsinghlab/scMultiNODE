'''
Description:
    Compare modality integration with quantitative metrics.


Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os.path
import numpy as np
from utils.FileUtils import loadSCData, tpSplitInd, getTimeStr
from modal_integration import DATA_DIR_DICT
from cell_type_supervison.FileUtils import loadscMultiNODELatent_withSupervision
from modal_integration.Compare_Modal_Alignment_Metric import computeAlignmentScore, plotMetric


if __name__ == '__main__':
    latent_dim = 50
    data_name = "zebrahub"  # coassay_cortex, human_organoid, drosophila, mouse_neocortex, zebrahub, amphioxus
    split_type = "all"
    data_type = "reduce"
    # -----
    save_filename = "./res/{}-{}-{}-ae-{}dim-metric.npy".format(data_name, data_type, split_type, latent_dim)
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
        integrated_dict = loadscMultiNODELatent_withSupervision(data_name, data_type, split_type, latent_dim)
        model_metric = computeAlignmentScore(integrated_dict, rna_cell_types, atac_cell_types, rna_cell_tps, atac_cell_tps, data_name)
        print("Saving res to {}".format(save_filename))
        np.save(save_filename, model_metric)
    else:
        print("Metrics already computed, loading file...")
        model_metric = np.load(save_filename, allow_pickle=True).item()
    # ================================================
    plotMetric(model_metric, data_name)

