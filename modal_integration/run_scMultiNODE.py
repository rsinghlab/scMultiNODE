'''
Description:
    Align RNA and ATAC measurements with scMultiNODE.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''

import numpy as np
import torch
from model.layer import LinearNet
from utils.DataUtils import sampleGaussian
from optim.running import constructscMultiNODEModel, scMultiNODETrain, scMultiNODEPredict

# ================================================


def scMultiNODEAlign(
        rna_train_data, atac_train_data,
        rna_train_tps, atac_train_tps,
        latent_dim, align_coeff = 0.01, dyn_reg_coeff = 0.001, n_neighbors = 100,
        iters = 1000, batch_size = 64, lr = 1e-3,
        ae_iters = 2000, ae_lr = 1e-3, ae_batch_size = 128,
        fusion_iters = 1000, fusion_lr = 1e-3, fusion_batch_size = 128,
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
    train_all = True
    # QGW parameters
    qgw_sample_ratio = 0.1
    gw_type = "gw"
    epsilon = 1e-2
    # Model training
    dynamic_model, ae_trained_model, fusion_trained_model, sgw, C1, C2 = scMultiNODETrain(
        rna_train_data, atac_train_data,
        rna_train_tps, atac_train_tps,
        dynamic_model,
        iters, batch_size, lr,
        ae_iters=ae_iters, ae_lr=ae_lr, ae_batch_size=ae_batch_size,
        fusion_iters=fusion_iters, fusion_lr=fusion_lr, fusion_batch_size=fusion_batch_size,
        align_coeff=align_coeff, dyn_reg_coeff=dyn_reg_coeff, train_all=train_all,
        n_neighbors=n_neighbors, qgw_sample_ratio=qgw_sample_ratio,
        gw_type=gw_type, epsilon=epsilon
    )
    # Compute joint latent representations
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