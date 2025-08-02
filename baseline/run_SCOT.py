'''
Description:
    Align RNA and ATAC assays with SCOT.

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





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # -----
    # load the data:
    Xrna = normalize(np.load("SCOT/data/SNARE/SNAREseq_rna_feat.npy"))
    Yatac = normalize(np.load("SCOT/data/SNARE/SNAREseq_atac_feat.npy"))
    print("Dimensions of input datasets are: ", "X(rna)= ", Xrna.shape, " Y(atac)= ", Yatac.shape)

    Xrna_ctypes = np.loadtxt("./SCOT/data/SNARE/SNAREseq_rna_types.txt").astype(int)
    Yatac_ctypes = np.loadtxt("./SCOT/data/SNARE/SNAREseq_atac_types.txt").astype(int)
    # -----
    pca = PCA(n_components=2)
    # Xrna_2Dpca = pca.fit_transform(Xrna)
    # Yatac_2Dpca = pca.fit_transform(Yatac)
    # plt.scatter(Xrna_2Dpca[:, 0], Xrna_2Dpca[:, 1], s=5, c=Xrna_ctypes)
    # plt.show()
    # plt.scatter(Yatac_2Dpca[:, 0], Yatac_2Dpca[:, 1], s=5, c=Yatac_ctypes)
    # plt.show()
    # -----



    # Xrna_integrated, Yatac_subsamp_integrated = SCOTv2Align(Xrna, Yatac, out_dim=10, normalize=False, norm="l2", k=20, eps=5e-3, rho=0.01, projMethod="embedding")
    # Xrna_integrated, Yatac_subsamp_integrated = SCOTv1Align(Xrna, Yatac, k=20, e=5e-3, normalize=False, norm="l2", selfTune=False, verbose=True)
    Xrna_integrated, Yatac_subsamp_integrated = SCOTv1Align(Xrna, Yatac, normalize=False, norm="l2", selfTune=True, verbose=True)

    # run PC jointly:
    concatenated = np.concatenate((Xrna_integrated, Yatac_subsamp_integrated), axis=0)
    concatenated_pc = pca.fit_transform(concatenated)
    Xrna_integrated_pc = concatenated_pc[0:Xrna_integrated.shape[0], :]
    Yatac_subsamp_integrated_pc = concatenated_pc[Xrna_integrated.shape[0]:, :]

    print(Xrna_integrated.shape, Yatac_subsamp_integrated.shape)
    print(Xrna_integrated_pc.shape, Yatac_subsamp_integrated_pc.shape)

    plt.scatter(Xrna_integrated_pc[:, 0], Xrna_integrated_pc[:, 1], s=5, c=Xrna_ctypes)
    plt.show()
    plt.scatter(Yatac_subsamp_integrated_pc[:, 0], Yatac_subsamp_integrated_pc[:, 1], s=5, c=Yatac_ctypes)
    plt.show()
