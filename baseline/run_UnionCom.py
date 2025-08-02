'''
Description:
    Align RNA and ATAC assays with UnionCom.

Reference:
    [1] Cao, K., Bai, X., Hong, Y., & Wan, L. (2020).
        Unsupervised topological alignment for single-cell multi-omics integration.
        Bioinformatics, 36(Supplement_1), i48-i56.
    [2] https://github.com/caokai1073/UnionCom
'''
from baseline.UnionCom import UnionCom
import numpy as np


def UnionComAlign(rna_data, atac_data, beta=1.0, perplexity=30, kmax=40, output_dim=32, return_aux=False):
    '''
    integration_type: "MultiOmics" or "BatchCorrect", default is "MultiOmics". "BatchCorrect" needs aligned features.
    epoch_pd: epoch of Prime-dual algorithm.
    epoch_DNN: epoch of training Deep Neural Network.
    epsilon: training rate of data matching matrix F.
    lr: training rate of DNN.
    batch_size: batch size of DNN.
    beta: trade-off parameter of structure preserving and matching.
    perplexity: perplexity of tsne projection
    rho: damping term.
    log_DNN: log step of training DNN.
    log_pd: log step of prime dual method
    manual_seed: random seed.
    delay: delay updata of alpha
    kmax: largest number of neighbors in geodesic distance
    output_dim: output dimension of integrated data.
    distance_mode: mode of distance, 'geodesic' or distances in sklearn.metrics.pairwise.pairwise_distances, default is 'geodesic'.
    project_mode:　mode of project, ['tsne', 'barycentric'], default is tsne.
    '''
    uc = UnionCom.UnionCom(
        integration_type='MultiOmics', epoch_pd=2000, epoch_DNN=100,
        epsilon=0.01, lr=0.001, batch_size=100, rho=10, beta=beta, perplexity=perplexity,
        log_DNN=10, log_pd=100, manual_seed=666, delay=0, kmax=kmax,
        output_dim=output_dim, distance_mode ='geodesic', project_mode='tsne'
    )
    integrated_data = uc.fit_transform(dataset=[rna_data, atac_data])
    rna_integrated = integrated_data[0]
    atac_integrated = integrated_data[1]
    if return_aux:
        return rna_integrated, atac_integrated, uc
    else:
        return rna_integrated, atac_integrated



if __name__ == '__main__':
    data1 = np.loadtxt("./UnionCom/simu1/domain1.txt")
    data2 = np.loadtxt("./UnionCom/simu1/domain2.txt")
    type1 = np.loadtxt("./UnionCom/simu1/type1.txt")
    type2 = np.loadtxt("./UnionCom/simu1/type2.txt")
    type1 = type1.astype(np.int)
    type2 = type2.astype(np.int)
    rna_integrated, atac_integrated, uc = UnionComAlign(data1, data2, beta=1.0, perplexity=30, kmax=40, output_dim=32, return_aux=True)
    integrated_data = [rna_integrated, atac_integrated]
    uc.test_LabelTA(integrated_data, [type1, type2])
    uc.Visualize([data1, data2], integrated_data, mode='PCA')  # without datatype
    uc.Visualize([data1, data2], integrated_data, [type1, type2], mode='PCA')  # with datatype