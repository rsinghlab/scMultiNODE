'''
Description:
    Align RNA and ATAC assays with uniPort.

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
import scanpy


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



if __name__ == '__main__':
    from scipy.sparse import csr_matrix

    data1 = np.loadtxt("./UnionCom/simu1/domain1.txt")
    data2 = np.loadtxt("./UnionCom/simu1/domain2.txt")
    type1 = np.loadtxt("./UnionCom/simu1/type1.txt")
    type2 = np.loadtxt("./UnionCom/simu1/type2.txt")

    data1 = csr_matrix(data1[:, :50])
    data2 = csr_matrix(data2[:, :50])

    data1_ann = scanpy.AnnData(data1)
    data2_ann = scanpy.AnnData(data2)

    data1_ann.obs['domain_id'] = 0
    data1_ann.obs['domain_id'] = data1_ann.obs['domain_id'].astype('category')
    data1_ann.obs['source'] = 'rna'

    data2_ann.obs['domain_id'] = 1
    data2_ann.obs['domain_id'] = data2_ann.obs['domain_id'].astype('category')
    data2_ann.obs['source'] = 'atac'

    adata_cm = data1_ann.concatenate(data2_ann, join='inner', batch_key='domain_id')

    rna_integrated, atac_integrated = uniPortAlign(
        data1_ann, data2_ann, lambda_kl=1.0, lambda_ot=1.0, reg=0.1, reg_m=1.0,
        output_dim=32, iteration=100, batch_size=32, lr=1e-3
    )

    # -----

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2, svd_solver="arpack")
    concatenated = np.concatenate((rna_integrated, atac_integrated), axis=0)
    concatenated_pc = pca.fit_transform(concatenated)
    Xrna_integrated_pc = concatenated_pc[0:rna_integrated.shape[0], :]
    Yatac_integrated_pc = concatenated_pc[rna_integrated.shape[0]:, :]

    print(rna_integrated.shape, atac_integrated.shape)
    print(Xrna_integrated_pc.shape, Yatac_integrated_pc.shape)

    plt.scatter(Xrna_integrated_pc[:, 0], Xrna_integrated_pc[:, 1], s=5, c="b", label="rna")
    plt.scatter(Yatac_integrated_pc[:, 0], Yatac_integrated_pc[:, 1], s=5, c="r", label="atac")
    plt.legend()
    plt.show()

