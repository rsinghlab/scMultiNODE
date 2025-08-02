'''
Description:
    Experiment - modality integration through scMultiNODE.

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
import torch
from utils.FileUtils import loadSCData, tpSplitInd
from modal_integration.run_scMultiNODE import scMultiNODEAlign
from plotting.PlottingUtils import plotIntegration
from modal_integration.Compare_Modal_Alignment_Metric import computeAlignmentScore, plotMetric

# ================================================
#   Co-Assay Datasets (HC and HO)
# ================================================

def integrateHCData(data_dir, need_save=False, need_metric=False, need_plot=False):
    # Loading data
    data_name = "coassay_cortex"
    split_type = "all"
    data_type = "reduce"
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
    rna_traj_data = [rna_cnt[np.where(rna_cell_tps == t)[0], :] for t in range(1, rna_n_tps + 1)]  # (# tps, # cells, # genes)
    atac_traj_data = [atac_cnt[np.where(atac_cell_tps == t)[0], :] for t in range(1, atac_n_tps + 1)]  # (# tps, # cells, # peaks)
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
    rna_integrated, atac_integrated = scMultiNODEAlign(
        rna_traj_data_torch, atac_traj_data_torch,
        rna_train_tps, atac_train_tps,
        latent_dim=output_dim, align_coeff = 0.1, dyn_reg_coeff = 0.1, n_neighbors = 10,
        iters = 1000, batch_size = 64, lr = 1e-3,
        ae_iters = 2000, ae_lr = 1e-3, ae_batch_size = 128,
        fusion_iters = 2000, fusion_lr = 1e-3, fusion_batch_size = 128,
        return_model=False
    )
    if isinstance(rna_integrated, torch.Tensor):
        rna_integrated = rna_integrated.detach().numpy()
    if isinstance(atac_integrated, torch.Tensor):
        atac_integrated = atac_integrated.detach().numpy()
    print("RNA integrated shape: ", rna_integrated.shape)
    print("ATAC integrated shape: ", atac_integrated.shape)
    # ================================================
    if need_save:
        print("=" * 70)
        save_filename = "./res/model_latent/{}-{}-{}-{}-{}dim.npy".format(
            data_name, data_type, split_type, model_name, latent_dim)
        print("Saving res to {}".format(save_filename))
        np.save(save_filename, {
            "rna_integrated": rna_integrated,
            "atac_integrated": atac_integrated,
            "aux": None,
        })
    # ================================================
    if need_metric:
        model_metric = computeAlignmentScore(
            {"scMultiNODE": {"rna_integrated": rna_integrated, "atac_integrated": atac_integrated}},
            rna_cell_types, atac_cell_types, rna_tps, atac_tps, data_name
        )
        plotMetric(model_metric, data_name)
    # ================================================
    if need_plot:
        plotIntegration(rna_integrated, atac_integrated, rna_cell_types, atac_cell_types, rna_tps, atac_tps, model_name)


def integrateHOData(data_dir, need_save=False, need_metric=False, need_plot=False):
    # Loading data
    data_name = "human_organoid"
    split_type = "all"
    data_type = "reduce"
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
    rna_traj_data = [rna_cnt[np.where(rna_cell_tps == t)[0], :] for t in range(1, rna_n_tps + 1)]  # (# tps, # cells, # genes)
    atac_traj_data = [atac_cnt[np.where(atac_cell_tps == t)[0], :] for t in range(1, atac_n_tps + 1)]  # (# tps, # cells, # peaks)
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
    rna_integrated, atac_integrated = scMultiNODEAlign(
        rna_traj_data_torch, atac_traj_data_torch,
        rna_train_tps, atac_train_tps,
        latent_dim=output_dim, align_coeff = 1.0, dyn_reg_coeff = 0.1, n_neighbors = 10,
        iters = 1000, batch_size = 64, lr = 1e-3,
        ae_iters = 2000, ae_lr = 1e-3, ae_batch_size = 128,
        fusion_iters = 2000, fusion_lr = 1e-3, fusion_batch_size = 128,
        return_model=False
    )
    if isinstance(rna_integrated, torch.Tensor):
        rna_integrated = rna_integrated.detach().numpy()
    if isinstance(atac_integrated, torch.Tensor):
        atac_integrated = atac_integrated.detach().numpy()
    print("RNA integrated shape: ", rna_integrated.shape)
    print("ATAC integrated shape: ", atac_integrated.shape)
    # ================================================
    if need_save:
        print("=" * 70)
        save_filename = "./res/model_latent/{}-{}-{}-{}-{}dim.npy".format(
            data_name, data_type, split_type, model_name, latent_dim)
        print("Saving res to {}".format(save_filename))
        np.save(save_filename, {
            "rna_integrated": rna_integrated,
            "atac_integrated": atac_integrated,
            "aux": None,
        })
    # ================================================
    if need_metric:
        model_metric = computeAlignmentScore(
            {"scMultiNODE": {"rna_integrated": rna_integrated, "atac_integrated": atac_integrated}},
            rna_cell_types, atac_cell_types, rna_tps, atac_tps, data_name
        )
        plotMetric(model_metric, data_name)
    # ================================================
    if need_plot:
        plotIntegration(rna_integrated, atac_integrated, rna_cell_types, atac_cell_types, rna_tps, atac_tps, model_name)


# ================================================
#   Unaligned Datasets (DR, MN, ZB, and AM)
# ================================================

def integrateDRData(data_dir, need_save=False, need_metric=False, need_plot=False):
    # Loading data
    data_name = "drosophila"
    split_type = "all"
    data_type = "reduce"
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
    rna_traj_data = [rna_cnt[np.where(rna_cell_tps == t)[0], :] for t in range(1, rna_n_tps + 1)]  # (# tps, # cells, # genes)
    atac_traj_data = [atac_cnt[np.where(atac_cell_tps == t)[0], :] for t in range(1, atac_n_tps + 1)]  # (# tps, # cells, # peaks)
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
    rna_integrated, atac_integrated = scMultiNODEAlign(
        rna_traj_data_torch, atac_traj_data_torch,
        rna_train_tps, atac_train_tps,
        latent_dim=output_dim, align_coeff = 10.0, dyn_reg_coeff = 1.0, n_neighbors = 10,
        iters = 2000, batch_size = 64, lr = 1e-3,
        ae_iters = 2000, ae_lr = 1e-3, ae_batch_size = 128,
        fusion_iters = 50, fusion_lr = 1e-3, fusion_batch_size = 128,
        return_model=False
    )
    if isinstance(rna_integrated, torch.Tensor):
        rna_integrated = rna_integrated.detach().numpy()
    if isinstance(atac_integrated, torch.Tensor):
        atac_integrated = atac_integrated.detach().numpy()
    print("RNA integrated shape: ", rna_integrated.shape)
    print("ATAC integrated shape: ", atac_integrated.shape)
    # ================================================
    if need_save:
        print("=" * 70)
        save_filename = "./res/model_latent/{}-{}-{}-{}-{}dim.npy".format(
            data_name, data_type, split_type, model_name, latent_dim)
        print("Saving res to {}".format(save_filename))
        np.save(save_filename, {
            "rna_integrated": rna_integrated,
            "atac_integrated": atac_integrated,
            "aux": None,
        })
    # ================================================
    if need_metric:
        model_metric = computeAlignmentScore(
            {"scMultiNODE": {"rna_integrated": rna_integrated, "atac_integrated": atac_integrated}},
            rna_cell_types, atac_cell_types, rna_tps, atac_tps, data_name
        )
        plotMetric(model_metric, data_name)
    # ================================================
    if need_plot:
        plotIntegration(rna_integrated, atac_integrated, rna_cell_types, atac_cell_types, rna_tps, atac_tps, model_name)



def integrateMNData(data_dir, need_save=False, need_metric=False, need_plot=False):
    # Loading data
    data_name = "mouse_neocortex"
    split_type = "all"
    data_type = "reduce"
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
    rna_traj_data = [rna_cnt[np.where(rna_cell_tps == t)[0], :] for t in range(1, rna_n_tps + 1)]  # (# tps, # cells, # genes)
    atac_traj_data = [atac_cnt[np.where(atac_cell_tps == t)[0], :] for t in range(1, atac_n_tps + 1)]  # (# tps, # cells, # peaks)
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
    rna_integrated, atac_integrated = scMultiNODEAlign(
        rna_traj_data_torch, atac_traj_data_torch,
        rna_train_tps, atac_train_tps,
        latent_dim=output_dim, align_coeff = 0.01, dyn_reg_coeff = 1.0, n_neighbors = 10,
        iters = 1000, batch_size = 64, lr = 1e-3,
        ae_iters = 2000, ae_lr = 1e-3, ae_batch_size = 128,
        fusion_iters = 2000, fusion_lr = 1e-3, fusion_batch_size = 128,
        return_model=False
    )
    if isinstance(rna_integrated, torch.Tensor):
        rna_integrated = rna_integrated.detach().numpy()
    if isinstance(atac_integrated, torch.Tensor):
        atac_integrated = atac_integrated.detach().numpy()
    print("RNA integrated shape: ", rna_integrated.shape)
    print("ATAC integrated shape: ", atac_integrated.shape)
    # ================================================
    if need_save:
        print("=" * 70)
        save_filename = "./res/model_latent/{}-{}-{}-{}-{}dim.npy".format(
            data_name, data_type, split_type, model_name, latent_dim)
        print("Saving res to {}".format(save_filename))
        np.save(save_filename, {
            "rna_integrated": rna_integrated,
            "atac_integrated": atac_integrated,
            "aux": None,
        })
    # ================================================
    if need_metric:
        model_metric = computeAlignmentScore(
            {"scMultiNODE": {"rna_integrated": rna_integrated, "atac_integrated": atac_integrated}},
            rna_cell_types, atac_cell_types, rna_tps, atac_tps, data_name
        )
        plotMetric(model_metric, data_name)
    # ================================================
    if need_plot:
        plotIntegration(rna_integrated, atac_integrated, rna_cell_types, atac_cell_types, rna_tps, atac_tps, model_name)


def integrateZBData(data_dir, need_save=False, need_metric=False, need_plot=False):
    # Loading data
    data_name = "zebrahub"
    split_type = "all"
    data_type = "reduce"
    (
        ann_rna_data, ann_atac_data, rna_cell_tps, atac_cell_tps,
        rna_n_tps, atac_n_tps, n_genes, n_peaks
    ) = loadSCData(data_name=data_name, data_type=data_type, split_type=split_type, data_dir=data_dir)
    rna_train_tps, atac_train_tps, rna_test_tps, atac_test_tps = tpSplitInd(data_name, split_type)
    rna_cnt = ann_rna_data.X
    atac_cnt = ann_atac_data.X
    # cell type
    rna_cell_types = np.asarray([str(x).lower() for x in ann_rna_data.obs["cell_type"].values])
    atac_cell_types = np.asarray([str(x).lower() for x in ann_atac_data.obs["cell_type"].values])
    rna_traj_cell_type = [rna_cell_types[np.where(rna_cell_tps == t)[0]] for t in range(1, rna_n_tps + 1)]
    atac_traj_cell_type = [atac_cell_types[np.where(atac_cell_tps == t)[0]] for t in range(1, atac_n_tps + 1)]
    rna_cell_types = np.concatenate(rna_traj_cell_type, axis=0)
    atac_cell_types = np.concatenate(atac_traj_cell_type, axis=0)
    # timepoint labels
    rna_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(rna_traj_cell_type)])
    atac_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(atac_traj_cell_type)])
    # data per timepoint
    rna_traj_data = [rna_cnt[np.where(rna_cell_tps == t)[0], :] for t in range(1, rna_n_tps + 1)]  # (# tps, # cells, # genes)
    atac_traj_data = [atac_cnt[np.where(atac_cell_tps == t)[0], :] for t in range(1, atac_n_tps + 1)]  # (# tps, # cells, # peaks)
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
    rna_integrated, atac_integrated = scMultiNODEAlign(
        rna_traj_data_torch, atac_traj_data_torch,
        rna_train_tps, atac_train_tps,
        latent_dim=output_dim, align_coeff = 0.001, dyn_reg_coeff = 0.001, n_neighbors = 10,
        iters = 1000, batch_size = 64, lr = 1e-3,
        ae_iters = 1000, ae_lr = 1e-3, ae_batch_size = 128,
        fusion_iters = 2000, fusion_lr = 1e-3, fusion_batch_size = 128,
        return_model=False
    )
    if isinstance(rna_integrated, torch.Tensor):
        rna_integrated = rna_integrated.detach().numpy()
    if isinstance(atac_integrated, torch.Tensor):
        atac_integrated = atac_integrated.detach().numpy()
    print("RNA integrated shape: ", rna_integrated.shape)
    print("ATAC integrated shape: ", atac_integrated.shape)
    # ================================================
    if need_save:
        print("=" * 70)
        save_filename = "./res/model_latent/{}-{}-{}-{}-{}dim.npy".format(
            data_name, data_type, split_type, model_name, latent_dim)
        print("Saving res to {}".format(save_filename))
        np.save(save_filename, {
            "rna_integrated": rna_integrated,
            "atac_integrated": atac_integrated,
            "aux": None,
        })
    # ================================================
    if need_metric:
        model_metric = computeAlignmentScore(
            {"scMultiNODE": {"rna_integrated": rna_integrated, "atac_integrated": atac_integrated}},
            rna_cell_types, atac_cell_types, rna_tps, atac_tps, data_name
        )
        plotMetric(model_metric, data_name)
    # ================================================
    if need_plot:
        plotIntegration(rna_integrated, atac_integrated, rna_cell_types, atac_cell_types, rna_tps, atac_tps, model_name, data_name="zebrahub")


def integrateAMData(data_dir, need_save=False, need_metric=False, need_plot=False):
    # Loading data
    data_name = "amphioxus"
    split_type = "all"
    data_type = "reduce"
    (
        ann_rna_data, ann_atac_data, rna_cell_tps, atac_cell_tps,
        rna_n_tps, atac_n_tps, n_genes, n_peaks
    ) = loadSCData(data_name=data_name, data_type=data_type, split_type=split_type, data_dir=data_dir)
    rna_train_tps, atac_train_tps, rna_test_tps, atac_test_tps = tpSplitInd(data_name, split_type)
    rna_cnt = ann_rna_data.X
    atac_cnt = ann_atac_data.X
    # cell type
    rna_cell_types = np.asarray([str(x).lower() for x in ann_rna_data.obs["cell_type"].values])
    atac_cell_types = np.asarray([str(x).lower() for x in ann_atac_data.obs["cell_type"].values])
    rna_traj_cell_type = [rna_cell_types[np.where(rna_cell_tps == t)[0]] for t in range(1, rna_n_tps + 1)]
    atac_traj_cell_type = [atac_cell_types[np.where(atac_cell_tps == t)[0]] for t in range(1, atac_n_tps + 1)]
    rna_cell_types = np.concatenate(rna_traj_cell_type, axis=0)
    atac_cell_types = np.concatenate(atac_traj_cell_type, axis=0)
    # timepoint labels
    rna_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(rna_traj_cell_type)])
    atac_tps = np.concatenate([np.repeat(t, x.shape[0]) for t, x in enumerate(atac_traj_cell_type)])
    # data per timepoint
    rna_traj_data = [rna_cnt[np.where(rna_cell_tps == t)[0], :] for t in range(1, rna_n_tps + 1)]  # (# tps, # cells, # genes)
    atac_traj_data = [atac_cnt[np.where(atac_cell_tps == t)[0], :] for t in range(1, atac_n_tps + 1)]  # (# tps, # cells, # peaks)
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
    rna_integrated, atac_integrated = scMultiNODEAlign(
        rna_traj_data_torch, atac_traj_data_torch,
        rna_train_tps, atac_train_tps,
        latent_dim=output_dim, align_coeff = 0.001, dyn_reg_coeff = 0.1, n_neighbors = 25,
        iters = 1000, batch_size = 64, lr = 1e-3,
        ae_iters = 2000, ae_lr = 1e-3, ae_batch_size = 128,
        fusion_iters = 2000, fusion_lr = 1e-3, fusion_batch_size = 128,
        return_model=False
    )
    if isinstance(rna_integrated, torch.Tensor):
        rna_integrated = rna_integrated.detach().numpy()
    if isinstance(atac_integrated, torch.Tensor):
        atac_integrated = atac_integrated.detach().numpy()
    print("RNA integrated shape: ", rna_integrated.shape)
    print("ATAC integrated shape: ", atac_integrated.shape)
    # ================================================
    if need_save:
        print("=" * 70)
        save_filename = "./res/model_latent/{}-{}-{}-{}-{}dim.npy".format(
            data_name, data_type, split_type, model_name, latent_dim)
        print("Saving res to {}".format(save_filename))
        np.save(save_filename, {
            "rna_integrated": rna_integrated,
            "atac_integrated": atac_integrated,
            "aux": None,
        })
    # ================================================
    if need_metric:
        model_metric = computeAlignmentScore(
            {"scMultiNODE": {"rna_integrated": rna_integrated, "atac_integrated": atac_integrated}},
            rna_cell_types, atac_cell_types, rna_tps, atac_tps, data_name
        )
        plotMetric(model_metric, data_name)
    # ================================================
    if need_plot:
        plotIntegration(rna_integrated, atac_integrated, rna_cell_types, atac_cell_types, rna_tps, atac_tps, model_name)


if __name__ == '__main__':
    need_save = False # save integration or not
    need_metric = True # compute evaluation metrics or not
    need_plot = True # plot integration or not
    # Specify dataset:

    # Co-asssay datasets:
    #   HC: human cortex
    #   HO: human organoid
    #
    # Unaligned dataset:
    #   DR: drosophila embryogenesis
    #   MN: mouse neocortex
    #   ZB: zebahub
    #   AM: amphioxus development
    data_name = "AM"
    # -----
    if data_name == "HC":
        integrateHCData(
            "../data/human_prefrontal_cortex_multiomic/reduce_processed/",
            need_save, need_metric, need_plot
        )
    elif data_name == "HO":
        integrateHOData(
            "../data/human_organoid_Fleck2022/reduce_processed/",
            need_save, need_metric, need_plot
        )
    elif data_name == "DR":
        integrateDRData(
            "../data/drosophila_embryonic/reduce_processed/",
            need_save, need_metric, need_plot
        )
    elif data_name=="MN":
        integrateMNData(
            "../data/Yuan2022_MouseNeocortex/reduce_processed/",
            need_save, need_metric, need_plot
        )
    elif data_name=="ZB":
        integrateZBData(
            "../data/Kim2024_Zebrahub/reduce_processed/",
            need_save, need_metric, need_plot
        )
    elif data_name=="AM":
        integrateAMData(
            "../data/Ma2022_Amphioxus/reduce_processed/",
            need_save, need_metric, need_plot
        )
    else:
        raise ValueError("Unknown data name {}!".format(data_name))