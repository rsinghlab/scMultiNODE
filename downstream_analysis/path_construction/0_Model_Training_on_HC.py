'''
Description:
    Downstram analysis.
    Step I: train scMultiNODE on HC dataset to learn the joint latent space

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
import torch
from utils.FileUtils import loadSCData, tpSplitInd
from modal_integration.run_scMultiNODE import scMultiNODEAlign
from plotting.PlottingUtils import plotIntegration

# ================================================
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
rna_cell_types = np.concatenate(rna_traj_cell_type, axis=0)
atac_cell_types = np.concatenate(atac_traj_cell_type, axis=0)
# timepoint labels
rna_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(rna_traj_cell_type)])
atac_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(atac_traj_cell_type)])
# data per timepoint
rna_traj_data = [rna_cnt[np.where(rna_cell_tps == t)[0], :] for t in
                 range(1, rna_n_tps + 1)]  # (# tps, # cells, # genes)
atac_traj_data = [atac_cnt[np.where(atac_cell_tps == t)[0], :] for t in
                  range(1, atac_n_tps + 1)]  # (# tps, # cells, # peaks)
n_tps = len(rna_traj_data)
all_tps = list(range(n_tps))
print("RNA shape: ", ann_rna_data.shape)
print("ATAC shape: ", ann_atac_data.shape)
print("# RNA features={}, # ATAC features={}".format(n_genes, n_peaks))
# ================================================
# Run scMultiNODE method
print("=" * 70)
latent_dim = 50
output_dim = latent_dim
model_name = "scMultiNODE"
print("[Model: {}] Output Dim={}".format(model_name, output_dim))
rna_traj_data_torch = [torch.FloatTensor(x) for x in rna_traj_data]
atac_traj_data_torch = [torch.FloatTensor(x) for x in atac_traj_data]
rna_train_tps = torch.FloatTensor(rna_train_tps)
atac_train_tps = torch.FloatTensor(atac_train_tps)
dynamic_model, rna_integrated, atac_integrated = scMultiNODEAlign(
    rna_traj_data_torch, atac_traj_data_torch,
    rna_train_tps, atac_train_tps,
    latent_dim=output_dim, align_coeff=0.1, dyn_reg_coeff=0.1, n_neighbors=10,
    iters=1000, batch_size=64, lr=1e-3,
    ae_iters=2000, ae_lr=1e-3, ae_batch_size=128,
    fusion_iters=1000, fusion_lr=1e-3, fusion_batch_size=128,
    return_model=True
)
if isinstance(rna_integrated, torch.Tensor):
    rna_integrated = rna_integrated.detach().numpy()
if isinstance(atac_integrated, torch.Tensor):
    atac_integrated = atac_integrated.detach().numpy()
print("RNA integrated shape: ", rna_integrated.shape)
print("ATAC integrated shape: ", atac_integrated.shape)
# ================================================
print("=" * 70)
save_filename = "./res/trained_model/{}-{}-{}-{}-{}dim.npy".format(
    data_name, data_type, split_type, model_name, latent_dim
)
print("Saving res to {}".format(save_filename))
np.save(save_filename, {
    "rna_integrated": rna_integrated,
    "atac_integrated": atac_integrated,
    "aux": {},
})
dict_filename = "./res/trained_model/{}-{}-{}-{}-{}dim-state_dict.pt".format(
    data_name, data_type, split_type, model_name, latent_dim
)
torch.save(dynamic_model.state_dict(), dict_filename)
# ================================================
print("=" * 70)
print("Plotting...")
plotIntegration(rna_integrated, atac_integrated, rna_cell_types, atac_cell_types, rna_tps, atac_tps, model_name)


