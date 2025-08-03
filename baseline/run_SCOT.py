'''
Description:
    Integrate RNA and ATAC assays with SCOT.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>

Reference:
    [1] Demetci, P., Santorella, R., Sandstede, B., Noble, W. S., & Singh, R. (2022).
        SCOT: single-cell multi-omics alignment with optimal transport.
        Journal of computational biology, 29(1), 3-18.
    [2] Demetçi, P., Santorella, R., Sandstede, B., & Singh, R. (2022, April).
        Unsupervised integration of single-cell multi-omics datasets with disproportionate cell-type representation.
        In International Conference on research in computational molecular biology (pp. 3-19). Cham: Springer International Publishing.
    [3] https://github.com/rsinghlab/SCOT
'''
import sys
sys.path.append("SCOT/src/")
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

import scotv1
import scotv2



def SCOTv1Align(rna_data, atac_data, k=20, e=5e-3, normalize=False, norm="l2", selfTune=False, verbose=False):
    SCOTv1 = scotv1.SCOT(rna_data, atac_data)
    if selfTune == False:
        rna_integrated, atac_integrated = SCOTv1.align(k=k, e=e, normalize=normalize, XontoY=False, verbose=verbose, norm=norm)
    else:
        rna_integrated, atac_integrated = SCOTv1.align(selfTune=True, normalize=normalize, XontoY=False, verbose=verbose, norm=norm)
    return rna_integrated, atac_integrated



def SCOTv2Align(rna_data, atac_data, out_dim, normalize=False, norm="l2", k=20, eps=5e-3, rho=0.01, projMethod="embedding"):
    '''
    Align and compute shared embedding space with SCOTv2.
    :param rna_data: Cell-by-gene RNA expression matrix.
    :param atac_data: Cell-by-feature ATAC matrix.
    :param out_dim: Output dimensions.
    :param normalize: Determines whether to normalize input data ahead of alignment. True or False (boolean parameter). Default = True.
    :param norm: Determines what sort of normalization to run, "l2", "l1", "max", "zscore". Default="l2"
    :param k: Number of neighbors to be used when constructing kNN graphs. Default= min(min(n_1, n_2), 50), where n_i, for i=1,2 corresponds to the number of samples in the i^th domain.
    :param eps: Regularization constant for the entropic regularization term in entropic Gromov-Wasserstein optimal transport formulation. Default= 1e-3
    :param rho:
    :param projMethod: "embedding", "barycentric"
    :return: List of integrated data in the embedding space.
    '''
    SCOTv2 = scotv2.SCOTv2([rna_data, atac_data])
    integrated_data = SCOTv2.align(normalize=normalize, k=k, eps=eps, rho=rho, projMethod=projMethod, mode= "connectivity", metric="correlation", Lambda=1.0, out_dim=out_dim, norm=norm)
    rna_integrated = integrated_data[0]
    atac_integrated = integrated_data[1]
    return rna_integrated, atac_integrated



