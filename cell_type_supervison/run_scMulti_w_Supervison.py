'''
Description:
    scMultiNODE training with cell type supervision

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''

import numpy as np
import torch
from optim.running import constructscMultiNODEModel
from cell_type_supervison.running import scMultiTrain_with_Celltype
from model.layer import LinearNet
from utils.DataUtils import sampleGaussian

# ================================================
from sklearn.preprocessing import LabelEncoder

def _splitTensor(x, s_list):
    splits = []
    start = 0
    for size in s_list:
        splits.append(x[start:start + size])
        start += size
    return splits


def scMultiAlignWithSupervison(
        rna_train_data, atac_train_data,
        rna_train_tps, atac_train_tps,
        rna_train_cell_type, atac_train_cell_type,
        latent_dim, align_coeff = 0.01, dyn_reg_coeff = 0.001, n_neighbors = 100,
        iters = 1000, batch_size = 64, lr = 1e-3,
        ae_iters = 1000, ae_lr = 1e-3, ae_batch_size = 128,
        fusion_iters = 2000, fusion_lr = 1e-3, fusion_batch_size = 128,
        return_model=False
):
    n_genes = rna_train_data[0].shape[1]
    n_peaks = atac_train_data[0].shape[1]
    # Construct scMulti model
    anchor_mod = "rna"
    rna_enc_latent = [50]
    rna_dec_latent = [50]
    atac_enc_latent = [50]
    atac_dec_latent = [50]
    fusion_latent = [50]
    drift_latent = [50]
    dynamic_model = constructscMultiNODEModel(
        n_genes, n_peaks, latent_dim, anchor_mod,
        rna_enc_latent=rna_enc_latent, rna_dec_latent=rna_dec_latent,
        atac_enc_latent=atac_enc_latent, atac_dec_latent=atac_dec_latent,
        fusion_latent=fusion_latent, drift_latent=drift_latent,
        act_name="relu", ode_method="euler"
    )
    print("Model:")
    print(dynamic_model)
    print("-" * 70)
    # scMulti training
    sim_coeff = 0.0
    train_all = True
    graph_metric = "correlation"  # cosine, correlation, wasserstein
    graph_type = "scot" if graph_metric is not "wasserstein" else "pynn"  # scot, pynn
    qgw_sample_ratio = 0.1
    gw_type = "gw"  # gw, egw
    epsilon = 1e-2

    l_enc = LabelEncoder()
    tmp_rna = np.concatenate(rna_train_cell_type)
    tmp_atac = np.concatenate(atac_train_cell_type)
    all_cell_type = np.concatenate([tmp_rna, tmp_atac])
    all_one_hot = l_enc.fit_transform(all_cell_type)
    all_one_hot = torch.LongTensor(all_one_hot)
    rna_train_one_hot = all_one_hot[:tmp_rna.shape[0]]
    atac_train_one_hot = all_one_hot[tmp_rna.shape[0]:]

    rna_train_one_hot = _splitTensor(rna_train_one_hot, [x.shape[0] for x in rna_train_cell_type])
    atac_train_one_hot = _splitTensor(atac_train_one_hot, [x.shape[0] for x in atac_train_cell_type])

    num_classes = len(l_enc.classes_)
    print("Num classes: ", num_classes)

    dynamic_model, ae_trained_model, fusion_trained_model, sgw, C1, C2 = scMultiTrain_with_Celltype(
        rna_train_data, atac_train_data,
        rna_train_tps, atac_train_tps,
        rna_train_one_hot, atac_train_one_hot,
        dynamic_model,
        iters, batch_size, lr,
        ae_iters=ae_iters, ae_lr=ae_lr, ae_batch_size=ae_batch_size,
        fusion_iters=fusion_iters, fusion_lr=fusion_lr, fusion_batch_size=fusion_batch_size,
        align_coeff=align_coeff, sim_coeff=sim_coeff,
        dyn_reg_coeff=dyn_reg_coeff, train_all=train_all,
        n_neighbors=n_neighbors, graph_metric=graph_metric, graph_type=graph_type, qgw_sample_ratio=qgw_sample_ratio,
        gw_type=gw_type, epsilon=epsilon, plot=False, num_classes=num_classes
    )
    latent_rna_fusion, latent_atac_fusion, _, _ = _computeFusionLatent(rna_train_data, atac_train_data, dynamic_model)
    if return_model:
        return dynamic_model, latent_rna_fusion.detach().numpy(), latent_atac_fusion.detach().numpy()
    else:
        return latent_rna_fusion.detach().numpy(), latent_atac_fusion.detach().numpy()



# ================================================

def _computeFusionLatent(
        rna_data, atac_data, dynamic_model
):
    '''
    Computing joint latent representations.
    '''
    dynamic_model.eval()
    all_rna_ata = torch.cat(rna_data, dim=0)
    all_atac_data = torch.cat(atac_data, dim=0)
    if isinstance(dynamic_model.fusion_layer, LinearNet):
        latent_rna_sample = dynamic_model.fusion_layer(dynamic_model.rna_enc(all_rna_ata))
        latent_atac_sample = dynamic_model.fusion_layer(dynamic_model.atac_enc(all_atac_data))
    else:
        latent_rna_sample = sampleGaussian(*dynamic_model.fusion_layer(dynamic_model.rna_enc(all_rna_ata)))
        latent_atac_sample = sampleGaussian(*dynamic_model.fusion_layer(dynamic_model.atac_enc(all_atac_data)))
    rna_recon = dynamic_model.rna_dec(latent_rna_sample)
    atac_recon = dynamic_model.atac_dec(latent_atac_sample)
    return latent_rna_sample, latent_atac_sample, rna_recon, atac_recon