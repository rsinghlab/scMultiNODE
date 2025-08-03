'''
Description:
    Integrate RNA and ATAC assays with uniPort.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>

Reference:
    [1] Cao, K., Gong, Q., Hong, Y., & Wan, L. (2022).
        A unified computational framework for single-cell data integration with optimal transport.
        Nature Communications, 13(1), 7419.
    [2] https://github.com/caokai1073/uniPort
'''
import sys
sys.path.append("uniPort/uniport/")
sys.path.append("uniPort/uniport/model/")
sys.path.append("uniPort/")
from baseline.uniPort import Run as runUniPort
import numpy as np


def uniPortAlign(
        rna_data, atac_data, lambda_kl=1.0, lambda_ot=1.0, reg=0.1, reg_m=1.0,
        output_dim=32, iteration=1000, batch_size=32, lr=1e-3, verbose=False
):
    '''
    lambda_kl:
        Balanced parameter for KL divergence. Default: 1.0
    lambda_ot:
        Balanced parameter for OT. Default: 1.0
    reg:
        Entropy regularization parameter in OT. Default: 0.1
    reg_m:
        Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
    batch_size
        Number of samples per batch to load. Default: 256
    lr
        Learning rate. Default: 2e-4
    enc
        Structure of encoder
    '''
    enc = [['fc', output_dim, 1, 'relu'], ['fc', output_dim, '', '']]
    res_ann = runUniPort(
        adatas=[atac_data, rna_data], mode="d", loss_type="MSE", num_workers=0,
        iteration=iteration, batch_size=batch_size, lr=lr, enc=enc,
        reg=reg, reg_m=reg_m, lambda_kl=lambda_kl, lambda_ot=lambda_ot, verbose=verbose)
    mod_ind = res_ann.obs.source.values
    rna_mod_idx = np.where(mod_ind == "rna")[0]
    atac_mod_idx = np.where(mod_ind == "atac")[0]
    res_latent = res_ann.obsm["latent"]
    rna_integrated = res_latent[rna_mod_idx]
    atac_integrated = res_latent[atac_mod_idx]
    return rna_integrated, atac_integrated
