"""
Description:
    Evaluation metrics of integration.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>

Reference:
    [1] https://github.com/rsinghlab/SCOT/tree/master
"""
import numpy as np
import scipy.spatial.distance
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import dcor

# ================================================

def transferAccuracy(domain1, domain2, type1, type2, n):
    """
    Metric from UnionCom: "Label Transfer Accuracy".
    """
    knn = KNeighborsClassifier(n_neighbors=n)
    # knn = SVC()
    knn.fit(domain2, type2)
    type1_predict = knn.predict(domain1)
    acc = accuracy_score(type1, type1_predict)
    return acc


def featureCorrPerSample(domain1, domain2):
    avg_x = np.mean(domain1, axis=1, keepdims=True)
    avg_y = np.mean(domain2, axis=1, keepdims=True)
    xm = domain1 - avg_x
    ym = domain2 - avg_y
    multi = xm * ym
    xs = np.square(xm)
    ys = np.square(ym)
    corr = np.sum(multi, axis=1) / (np.sqrt(np.sum(xs, axis=1)) * np.sqrt(np.sum(ys, axis=1)))
    corr = np.mean(corr)
    return corr


def neighborOverlap(source_feature, target_feature, target_sample_name, n_neighbors):
    '''Neighborhood overlap.'''
    n_src = source_feature.shape[0]
    overlap_ind_list = []
    pbar = tqdm(range(n_src), desc="[ Neighborhood Overlap ]")
    for i in pbar:
        tmp = np.concatenate([source_feature[i, :][np.newaxis, :], target_feature], axis=0)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(tmp)
        nbr_idx = nbrs.kneighbors(source_feature[i, :][np.newaxis, :], n_neighbors, return_distance=False).squeeze()
        overlap_ind_list.append((target_sample_name[i] + 1) in nbr_idx)
    score = np.mean(overlap_ind_list)
    return score


def FOSCTTM(x1_mat, x2_mat):
    """FOSCTTM metrics."""
    true_dist = np.asarray([scipy.spatial.distance.euclidean(x1_mat[i], x2_mat[i]) for i in range(x1_mat.shape[0])])
    true_dist = np.hstack([true_dist[:, np.newaxis] for _ in range(x2_mat.shape[0])])
    euc_dist = scipy.spatial.distance.cdist(x1_mat, x2_mat)
    closer_ind = euc_dist < true_dist
    closer_cnt = np.sum(closer_ind, axis=1)
    closer_cnt = np.maximum(closer_cnt - 2, np.zeros_like(closer_cnt))
    score = np.mean(closer_cnt / (x1_mat.shape[0] - 2))
    return score


def batchEntropy(data, batches, n_neighbors=100, n_pools=100, n_samples_per_pool=100):
    """
    Function "batch_entropy_mixing_score".
    Calculate batch entropy mixing score
        Higher is better.
    Algorithm
    ---------
        * 1. Calculate the regional mixing entropies at the location of 100 randomly chosen cells from all batches
        * 2. Define 100 nearest neighbors for each randomly chosen cell
        * 3. Calculate the mean mixing entropy as the mean of the regional entropies
        * 4. Repeat above procedure for 100 iterations with different randomly chosen cells.

    Parameters
    ----------
    data
        np.array of shape nsamples x nfeatures.
    batches
        batch labels of nsamples.
    n_neighbors
        The number of nearest neighbors for each randomly chosen cell. By default, n_neighbors=100.
    n_samples_per_pool
        The number of randomly chosen cells from all batches per iteration. By default, n_samples_per_pool=100.
    n_pools
        The number of iterations with different randomly chosen cells. By default, n_pools=100.

    Returns
    -------
    Batch entropy mixing score
    """
    # Reference: https://github.com/caokai1073/uniPort/blob/main/uniport/metrics.py
    # Modified from SCALEX

    #     print("Start calculating Entropy mixing score")
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i] / P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i] / P[i]) / a
            entropy = entropy - adapt_p[i] * np.log(adapt_p[i] + 10 ** -8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
        P[i] = np.mean(batches == batches_[i])
    for t in range(n_pools):
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
        [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    Score = score / float(n_pools)
    return Score / float(np.log2(N_batches))


def labelCorr(x_mat, x_label):
    # Use distance correlation
    corr = dcor.distance_correlation(x_mat, x_label)
    return corr

