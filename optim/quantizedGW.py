'''
Description:
    Quantized Gromov-Wasserstein algorithm.
    We adopt codes from https://github.com/trneedham/QuantizedGromovWasserstein/blob/main/quantizedGW.py

Reference:
    [1] Chowdhury, S., Miller, D., & Needham, T. (2021).
        Quantized gromov-wasserstein.
        In Machine Learning and Knowledge Discovery in Databases.
        Research Track: European Conference, ECML PKDD 2021, Bilbao, Spain, September 13–17, 2021, Proceedings, Part III 21 (pp. 811-827).
        Springer International Publishing.
    [2] https://github.com/trneedham/QuantizedGromovWasserstein
'''
import numpy as np
import ot
from scipy.sparse import coo_matrix
import time

""""
---quantized Gromov Wasserstein---
The main algorithm is here.
Variants are below (quantized Fused GW and a version specifically for point clouds).
"""

def renormalize_prob(pv):
    # Robust method to turn an arbitrary vector into a probability vector
    q = pv.copy()
    if pv.sum() > 1:
        diff = pv.sum()-1
        q[q.argmax()] -= diff # take off mass from the heaviest
    elif pv.sum() < 1:
        diff = 1-pv.sum()
        q[q.argmin()] += diff # add mass to the lightest

    return q


def compress_graph(Dist,p_compressed):

    good_inds = [j for j in range(len(p_compressed)) if p_compressed[j] > 0]

    Dist_new = Dist[np.ix_(good_inds,good_inds)]

    p_new = renormalize_prob(np.array([p_compressed[j] for j in range(len(p_compressed)) if p_compressed[j] > 0]))

    return Dist_new, p_new


def find_support(p_compressed):

    supp = list(np.argwhere(p_compressed > 0).ravel())

    return supp


def find_submatching_locally_linear(Dist1,Dist2,coup1,coup2,i,j):

    subgraph_i = find_support(coup1[:,i])
    p_i = coup1[:,i][subgraph_i]/np.sum(coup1[:,i][subgraph_i])

    subgraph_j = find_support(coup2[:,j])
    p_j = coup2[:,j][subgraph_j]/np.sum(coup2[:,j][subgraph_j])

    x_i = list(Dist1[i,:][subgraph_i].reshape(len(subgraph_i),))
    x_j = list(Dist2[j,:][subgraph_j].reshape(len(subgraph_j),))

    coup_sub_ij = ot.emd_1d(x_i,x_j,p_i,p_j,p=2)

    return coup_sub_ij


"""
The point cloud version (just assuming unique nearest neighbors).
"""


"""
--- quantized GW for Point Clouds ---

The code below uses the generic assumption that pairwise distances are unique.
This allows us to do certain steps more efficiently.
"""

def deterministic_coupling_point_cloud(Dist,p,node_subset):

    n = Dist.shape[0]

    # Get distance matrix from all nodes to the subset nodes
    D_subset = Dist[:,node_subset]

    # Find shortest distances to the subset
    dists_to_subset_idx = np.argmin(D_subset,axis = 1)

    # Construct the coupling
    row = list(range(n))
    col = [node_subset[j] for j in dists_to_subset_idx]
    data = p
    coup = coo_matrix((data,(row,col)),shape = (n,n))

    return coup


def compress_graph_from_subset_point_cloud(Dist,p,node_subset):
    """
    Update Feb 8, 2020: this is the version of `compress_graph_from_subset`
    that we're using for point cloud experiments -- sparse matrices help a lot
    """
    coup = deterministic_coupling_point_cloud(Dist,p,node_subset)
    p_compressed = renormalize_prob(np.squeeze(np.array(np.sum(coup, axis = 0))))


    return coup.toarray(), p_compressed


# main function of qGW coupling between data point clouds
def compressed_gw_point_cloud(Dist1,Dist2,p1,p2,node_subset1,node_subset2, verbose = False, return_dense = True, gw_type="gw", epsilon=None):
    # gw_type: gw, egw

    # Compress Graphs
    start = time.time()
    if verbose:
        print('Compressing Graphs...')

    coup1, p_compressed1 = compress_graph_from_subset_point_cloud(Dist1,p1,node_subset1)
    coup2, p_compressed2 = compress_graph_from_subset_point_cloud(Dist2,p2,node_subset2)

    Dist_new1, p_new1 = compress_graph(Dist1,p_compressed1)
    Dist_new2, p_new2 = compress_graph(Dist2,p_compressed2)

    if verbose:
        print('Time for Compressing:', time.time() - start)

    # Match compressed graphs
    start = time.time()
    if verbose:
        print('Matching Compressed Graphs...')

    if gw_type == "gw":
        coup_compressed, log = ot.gromov.gromov_wasserstein(Dist_new1, Dist_new2, p_new1, p_new2, 'square_loss', verbose=False, log=True)
    elif gw_type == "egw":
        coup_compressed, log = ot.gromov.entropic_gromov_wasserstein(Dist_new1, Dist_new2, p_new1, p_new2, loss_fun='square_loss', epsilon=epsilon, log=True, verbose=False)

    # If coupling is dense, abort the algorithm and return a dense full size coupling.
    if np.sum(coup_compressed > 1e-10) > len(coup_compressed)**1.5:
        print('Dense Compressed Matching, returning dense coupling...')
        return p1[:,None]*p2[None,:]

    # coup_compressed, log = gwa.gromov_wasserstein(Dist_new1, Dist_new2, p_new1, p_new2)
    if verbose:
        print('Time for Matching Compressed:', time.time() - start)

    # Find submatchings and create full coupling
    if verbose:
        print('Matching Subgraphs and Constructing Coupling...')
    supp1 = find_support(p_compressed1)
    supp2 = find_support(p_compressed2)

    full_coup = coo_matrix((Dist1.shape[0], Dist2.shape[0]))

    matching_time = 0
    matching_and_expanding_time = 0
    num_local_matches = 0

    for (i_enum, i) in enumerate(supp1):
        subgraph_i = find_support(coup1[:,i])
        for (j_enum, j) in enumerate(supp2):
            start = time.time()
            w_ij = coup_compressed[i_enum,j_enum]
            if w_ij > 1e-10:
                num_local_matches += 1
                subgraph_j = find_support(coup2[:,j])
                # Compute submatching
                coup_sub_ij = find_submatching_locally_linear(Dist1,Dist2,coup1,coup2,i,j)
                matching_time += time.time()-start
                # Expand to correct size
                idx = np.argwhere(coup_sub_ij > 1e-10)
                idx_i = idx.T[0]
                idx_j = idx.T[1]
                row = np.array(subgraph_i)[idx_i]
                col = np.array(subgraph_j)[idx_j]
                data = w_ij*np.array([coup_sub_ij[p[0],p[1]] for p in list(idx)])
                expanded_coup_sub_ij = coo_matrix((data, (row,col)), shape=(full_coup.shape[0], full_coup.shape[1]))
                # Update full coupling
                full_coup += expanded_coup_sub_ij
                matching_and_expanding_time += time.time()-start

    if verbose:
        print('Total Time for',num_local_matches,'local matches:')
        print('Local matching:', matching_time)
        print('Local Matching Plus Expansion:', matching_and_expanding_time)

    if return_dense:
        return full_coup.toarray()
    else:
        return full_coup