'''
Description:
    scMultiNODE integration with cell type supervision.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import time
import numpy as np
import torch
from utils.FileUtils import loadSCData, tpSplitInd
from cell_type_supervison.run_scMulti_w_Supervison import scMultiAlignWithSupervison
from modal_integration import DATA_DIR_DICT

# ================================================
# Loading data
data_name = "zebrahub"  # coassay_cortex, drosophila, mouse_neocortex, human_organoid, amphioxus, zebrahub
split_type = "all"
data_type = "reduce"
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
rna_cell_types = rna_traj_cell_type
atac_cell_types = atac_traj_cell_type
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
# Optimized hyperparameters
if data_name == "mouse_neocortex":
    align_coeff = 0.01
    dyn_reg_coeff = 0.001
    n_neighbors = 10
elif data_name == "coassay_cortex":
    align_coeff = 0.1
    dyn_reg_coeff = 0.1
    n_neighbors = 5
elif data_name == "zebrahub":
    align_coeff = 0.01
    dyn_reg_coeff = 0.001
    n_neighbors = 10
elif data_name == "amphioxus":
    align_coeff = 1.0
    dyn_reg_coeff = 0.001
    n_neighbors = 75
elif data_name == "drosophila":
    align_coeff = 0.01
    dyn_reg_coeff = 0.001
    n_neighbors = 5
elif data_name == "human_organoid":
    align_coeff = 1.0
    dyn_reg_coeff = 0.1
    n_neighbors = 10
else:
    raise ValueError("Unsupported data {}!".format(data_name))
# ================================================
# Run aligning method
print("=" * 70)
latent_dim = 50
output_dim = latent_dim
model_name = "scMultiNODE"

print("[Model: {}] Output Dim={}".format(model_name, output_dim))
start = time.time()

rna_traj_data_torch = [torch.FloatTensor(x) for x in rna_traj_data]
atac_traj_data_torch = [torch.FloatTensor(x) for x in atac_traj_data]
rna_train_tps = torch.FloatTensor(rna_train_tps)
atac_train_tps = torch.FloatTensor(atac_train_tps)
rna_integrated, atac_integrated = scMultiAlignWithSupervison(
    rna_traj_data_torch, atac_traj_data_torch,
    rna_train_tps, atac_train_tps,
    rna_cell_types, atac_cell_types,
    latent_dim=output_dim, align_coeff=align_coeff, dyn_reg_coeff=dyn_reg_coeff, n_neighbors=n_neighbors,
    iters=1000, batch_size=64, lr=1e-3,
    ae_iters=1000, ae_lr=1e-3, ae_batch_size=128,
    fusion_iters=2000, fusion_lr=1e-3, fusion_batch_size=128
)
if isinstance(rna_integrated, torch.Tensor):
    rna_integrated = rna_integrated.detach().numpy()
if isinstance(atac_integrated, torch.Tensor):
    atac_integrated = atac_integrated.detach().numpy()
print("RNA integrated shape: ", rna_integrated.shape)
print("ATAC integrated shape: ", atac_integrated.shape)
# ================================================
# Save results
print("=" * 70)
save_filename = "./res/{}-{}-{}-{}-{}dim.npy".format(
    data_name, data_type, split_type, model_name, latent_dim, output_dim)
print("Saving res to {}".format(save_filename))
np.save(save_filename, {
    "rna_integrated": rna_integrated,
    "atac_integrated": atac_integrated,
})

# ================================================
# Visualizations and metrics computation (for quick test)
from plotting.PlottingUtils import plotIntegration
from modal_integration.Compare_Modal_Alignment_Metric import computeAlignmentScore, plotMetric
rna_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(rna_traj_cell_type)])
atac_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(atac_traj_cell_type)])
model_metric = computeAlignmentScore(
    {"scMultiNODE": {"rna": rna_integrated, "atac": atac_integrated}},
    np.concatenate(rna_cell_types, axis=0), np.concatenate(atac_cell_types, axis=0), rna_tps, atac_tps, data_name
)
plotMetric(model_metric, data_name)
plotIntegration(rna_integrated, atac_integrated, np.concatenate(rna_cell_types, axis=0), np.concatenate(atac_cell_types, axis=0), rna_tps, atac_tps, model_name)

