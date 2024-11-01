'''
Description:
    Model (with alignment) training and prediction.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import copy
import torch
from tqdm import tqdm
import numpy as np
import itertools
import ot
import time
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph

from model.layer import LinearNet
from model.diff_solver import ODE
from model.dynamic_model import scMultiNODE
from optim.loss_func import SinkhornLoss, MSELoss
from optim.quantizedGW import compressed_gw_point_cloud as qGW
from random import sample


# =============================================

def constructscMultiNODEModel(
        n_genes, n_peaks, latent_dim, anchor_mod,
        rna_enc_latent=None, rna_dec_latent=None, atac_enc_latent=None, atac_dec_latent=None,
        fusion_latent = None, drift_latent=None,
        act_name="relu", ode_method="euler"
):
    rna_enc = LinearNet(input_dim=n_genes, latent_size_list=rna_enc_latent, output_dim=latent_dim, act_name=act_name)
    rna_dec = LinearNet(input_dim=latent_dim, latent_size_list=rna_dec_latent, output_dim=n_genes, act_name=act_name)
    atac_enc = LinearNet(input_dim=n_peaks, latent_size_list=atac_enc_latent, output_dim=latent_dim, act_name=act_name)
    atac_dec = LinearNet(input_dim=latent_dim, latent_size_list=atac_dec_latent, output_dim=n_peaks, act_name=act_name)
    fusion_layer = LinearNet(input_dim=latent_dim, latent_size_list=fusion_latent, output_dim=latent_dim, act_name=act_name)
    diffeq_drift_net = LinearNet(input_dim=latent_dim, latent_size_list=drift_latent, output_dim=latent_dim, act_name=act_name)
    diffeq_decoder = ODE(input_dim=latent_dim, drift_net=diffeq_drift_net, ode_method=ode_method)
    model = scMultiNODE(
        n_genes, n_peaks,
        latent_dim=latent_dim, anchor_mod=anchor_mod,
        rna_enc=rna_enc, rna_dec=rna_dec, atac_enc=atac_enc, atac_dec=atac_dec,
        fusion_layer=fusion_layer, diffeq_decoder=diffeq_decoder
    )
    return model

# =============================================

def scMultiNODETrain(
        rna_train_data, atac_train_data,
        rna_train_tps, atac_train_tps,
        dynamic_model,
        iters, batch_size, lr,
        ae_iters=200, ae_lr=1e-3, ae_batch_size=128,
        fusion_iters=200, fusion_lr=1e-3, fusion_batch_size=128,
        align_coeff=1.0, dyn_reg_coeff=1.0, train_all=True,
        n_neighbors=10, qgw_sample_ratio=0.1, gw_type="gw", epsilon=1e-2
):
    # =========================
    # Preparation
    all_rna_train_data = torch.cat(rna_train_data, dim=0)
    all_atac_train_data = torch.cat(atac_train_data, dim=0)
    # =========================
    # Step I: train two encoder-decoders separately for two modalities
    rna_enc = dynamic_model.rna_enc
    rna_dec = dynamic_model.rna_dec
    atac_enc = dynamic_model.atac_enc
    atac_dec = dynamic_model.atac_dec
    rna_loss_list = []
    atac_loss_list = []
    rna_ae_params = itertools.chain(*[rna_enc.parameters(), rna_dec.parameters()])
    atac_ae_params = itertools.chain(*[atac_enc.parameters(), atac_dec.parameters()])
    rna_optimizer = torch.optim.Adam(params=rna_ae_params, lr=ae_lr, betas=(0.95, 0.99))
    atac_optimizer = torch.optim.Adam(params=atac_ae_params, lr=ae_lr, betas=(0.95, 0.99))
    ae_pbar = tqdm(range(ae_iters), desc="[ AE Pre-Training ]")
    rna_enc.train()
    rna_dec.train()
    atac_enc.train()
    atac_dec.train()
    for t in ae_pbar: # we put RNA and ATAC training in the same loop just for simplicity, but they are independent
        rna_optimizer.zero_grad()
        atac_optimizer.zero_grad()
        rna_batch_data = all_rna_train_data[np.random.choice(np.arange(all_rna_train_data.shape[0]), ae_batch_size, replace=False), :]
        atac_batch_data = all_atac_train_data[np.random.choice(np.arange(all_atac_train_data.shape[0]), ae_batch_size, replace=False), :]
        # -----
        latent_rna_sample = rna_enc(rna_batch_data)
        rna_recon = rna_dec(latent_rna_sample)
        latent_atac_sample = atac_enc(atac_batch_data)
        atac_recon = atac_dec(latent_atac_sample)
        # -----
        rna_recon_loss = MSELoss(rna_batch_data, rna_recon)
        atac_recon_loss = MSELoss(atac_batch_data, atac_recon)
        ae_pbar.set_postfix({
            "RNA Loss": "{:.4f}".format(rna_recon_loss),
            "ATAC Loss": "{:.4f}".format(atac_recon_loss),
        })
        rna_loss_list.append([rna_recon_loss.item()])
        atac_loss_list.append([atac_recon_loss.item()])
        rna_recon_loss.backward()
        rna_optimizer.step()
        atac_recon_loss.backward()
        atac_optimizer.step()
    dynamic_model.rna_enc = rna_enc
    dynamic_model.rna_dec = rna_dec
    dynamic_model.atac_enc = atac_enc
    dynamic_model.atac_dec = atac_dec
    ae_trained_model = copy.deepcopy(dynamic_model)
    # =========================
    # Step II: QGW aligns two modalities based on latent representations
    dynamic_model.eval()
    latent_rna_sample = dynamic_model.rna_enc(all_rna_train_data)
    latent_atac_sample = dynamic_model.atac_enc(all_atac_train_data)
    n_rna_cells = latent_rna_sample.shape[0]
    n_atac_cells = latent_atac_sample.shape[0]
    # Compute cell similarities
    latent_rna_sample_copy = latent_rna_sample.detach().numpy()
    latent_atac_sample_copy = latent_atac_sample.detach().numpy()
    C1 = _mod_distance(latent_rna_sample_copy, n_neighbors, "correlation")
    C2 = _mod_distance(latent_atac_sample_copy, n_neighbors, "correlation")
    C1 = np.nan_to_num(C1, nan=0.0)
    C2 = np.nan_to_num(C2, nan=0.0)
    C1 /= np.max(C1)
    C2 /= np.max(C2)
    # GW to align anchor cells between two modalities
    start = time.time()
    p = ot.unif(n_rna_cells)
    q = ot.unif(n_atac_cells)
    # Subset of node features
    sample_ratio = qgw_sample_ratio
    node_subset1 = list(set(sample(list(range(n_rna_cells)), int(sample_ratio*n_rna_cells))))
    node_subset2 = list(set(sample(list(range(n_atac_cells)), int(sample_ratio*n_atac_cells))))
    sgw = qGW(C1, C2, p, q, node_subset1, node_subset2, return_dense=False, verbose=True, gw_type=gw_type, epsilon=epsilon)
    end = time.time()
    length = end - start
    print('Quantized Gromov-Wasserstein distance: Time cost = {} secs'.format(length))
    sgw = sgw / sgw.max()
    sgw[np.abs(sgw) <= 0.01] = 0.0 # sgw is cell correspondence matrix
    if not isinstance(sgw, csr_matrix):
        sgw = csr_matrix(sgw)
    # =========================
    # Step III: learn joint latent space based on cell correspondence
    rna_enc = dynamic_model.rna_enc
    rna_dec = dynamic_model.rna_dec
    atac_enc = dynamic_model.atac_enc
    atac_dec = dynamic_model.atac_dec
    fusion_layer = dynamic_model.fusion_layer
    fusion_loss_list = []
    fusion_params = itertools.chain(*[
        rna_dec.parameters(), atac_dec.parameters(),
        rna_enc.parameters(), atac_enc.parameters(),
        fusion_layer.parameters()
    ])
    fusion_optimizer = torch.optim.Adam(params=fusion_params, lr=fusion_lr, betas=(0.95, 0.99))
    fusion_pbar = tqdm(range(fusion_iters), desc="[ Joint Latent Training ]")
    rna_dec.train()
    rna_enc.train()
    atac_enc.train()
    atac_dec.train()
    fusion_layer.train()
    for t in fusion_pbar:
        fusion_optimizer.zero_grad()
        rna_batch_idx = np.random.choice(np.arange(all_rna_train_data.shape[0]), fusion_batch_size, replace=False)
        atac_batch_idx = np.random.choice(np.arange(all_atac_train_data.shape[0]), fusion_batch_size, replace=False)
        source_idx1, target_idx1 = _extractAlignIdx(sgw, rna_batch_idx) # from RNA to ATAC
        source_idx2, target_idx2 = _extractAlignIdx(sgw.T, atac_batch_idx) # from ATAC to RNA
        # -----
        # Reconstruction
        rna_batch = all_rna_train_data[rna_batch_idx, :]
        atac_batch = all_atac_train_data[atac_batch_idx,:]
        rna_latent = fusion_layer(rna_enc(rna_batch))
        atac_latent = fusion_layer(atac_enc(atac_batch))
        rna_recon = rna_dec(rna_latent)
        atac_recon = atac_dec(atac_latent)
        rna_recon_loss = MSELoss(rna_batch, rna_recon)
        atac_recon_loss = MSELoss(atac_batch, atac_recon)
        # -----
        # RNA-ATAC Alignment
        if target_idx1 is not None:
            source_batch1 = all_rna_train_data[source_idx1, :]
            target_batch1 = all_atac_train_data[target_idx1, :]
            source_latent1 = fusion_layer(rna_enc(source_batch1))
            target_latent1 = fusion_layer(atac_enc(target_batch1))
            rna_align_loss = align_coeff * MSELoss(source_latent1, target_latent1, reduction="mean")
        else:
            rna_align_loss = 0.0
        # -----
        # ATAC-RNA Alignment
        if target_idx2 is not None:
            source_batch2 = all_atac_train_data[source_idx2, :]
            target_batch2 = all_rna_train_data[target_idx2, :]
            source_latent2 = fusion_layer(atac_enc(source_batch2))
            target_latent2 = fusion_layer(rna_enc(target_batch2))
            atac_align_loss = align_coeff * MSELoss(source_latent2, target_latent2, reduction="mean")
        else:
            atac_align_loss = 0.0
        # -----
        # Loss computation
        fusion_loss = rna_recon_loss + atac_recon_loss + rna_align_loss + atac_align_loss
        # Backward
        fusion_pbar.set_postfix({
            "Loss": "{:.3f}".format(fusion_loss),
            "RNA recon": "{:.3f}".format(rna_recon_loss),
            "ATAC recon": "{:.3f}".format(atac_recon_loss),
            "RNA align": "{:.3f}".format(rna_align_loss),
            "ATAC align": "{:.3f}".format(atac_align_loss),
        })
        fusion_loss_list.append([
            fusion_loss.item(), rna_recon_loss.item(), atac_recon_loss.item(),
            rna_align_loss.item(), atac_align_loss.item()
        ])
        fusion_loss.backward()
        fusion_optimizer.step()
    dynamic_model.rna_enc = rna_enc
    dynamic_model.rna_dec = rna_dec
    dynamic_model.atac_enc = atac_enc
    dynamic_model.atac_dec = atac_dec
    dynamic_model.fusion_layer = fusion_layer
    fusion_trained_model = copy.deepcopy(dynamic_model)
    # =========================
    # Step IV: incorporate dynamics in the joint latent space
    # =========================
    if train_all:
        dynamic_params = dynamic_model.parameters()
    else:
        dynamic_params = itertools.chain(*[
            dynamic_model.diffeq_decoder.parameters(),
            dynamic_model.rna_dec.parameters(),
            dynamic_model.atac_dec.parameters(),
        ])
    dynamic_optimizer = torch.optim.Adam(params=dynamic_params, lr=lr, betas=(0.95, 0.99))
    dynamic_pbar = tqdm(range(iters), desc="[ Dynamic Training ]")
    dynamic_model.train()
    rna_loss_coeff = 1.0
    atac_loss_coeff = 1.0
    dyn_reg_coeff = dyn_reg_coeff
    blur = 0.05
    scaling = 0.5
    dynamic_loss_list = []
    for t in dynamic_pbar:
        dynamic_optimizer.zero_grad()
        rna_recon_obs, atac_recon_obs, first_tp_data, rna_latent_seq, atac_latent_seq = dynamic_model(
            rna_train_data[0], rna_train_tps, atac_train_tps, batch_size=batch_size
        )
        # -----
        # Reconstruction loss
        rna_loss = rna_loss_coeff * SinkhornLoss(rna_train_data, rna_recon_obs, blur=blur, scaling=scaling, batch_size=200)
        atac_loss = atac_loss_coeff * SinkhornLoss(atac_train_data, atac_recon_obs, blur=blur, scaling=scaling, batch_size=200)
        # -----
        # Regularization
        rna_pretrain_latents = [dynamic_model.fusion_layer(dynamic_model.rna_enc(rna_train_curr_tp)) for rna_train_curr_tp in rna_train_data]
        atac_pretrain_latents = [dynamic_model.fusion_layer(dynamic_model.atac_enc(atac_train_curr_tp)) for atac_train_curr_tp in atac_train_data]
        rna_latent_loss = dyn_reg_coeff * SinkhornLoss(rna_pretrain_latents, rna_latent_seq, blur=blur, scaling=scaling, batch_size=200)
        atac_latent_loss = dyn_reg_coeff * SinkhornLoss(atac_pretrain_latents, atac_latent_seq, blur=blur, scaling=scaling, batch_size=200)
        # -----
        # Backward
        dynamic_loss = rna_loss + atac_loss + rna_latent_loss + atac_latent_loss
        dynamic_pbar.set_postfix({
            "Loss": "{:.3f}".format(dynamic_loss),
            "RNA Loss": "{:.3f}".format(rna_loss),
            "ATAC Loss": "{:.3f}".format(atac_loss),
            "RNA Reg": "{:.3f}".format(rna_latent_loss),
            "ATAC Reg": "{:.3f}".format(atac_latent_loss),
        })
        dynamic_loss_list.append(
            [dynamic_loss.item(), rna_loss.item(), atac_loss.item(), rna_latent_loss.item(), atac_latent_loss.item()])
        dynamic_loss.backward()
        dynamic_optimizer.step()
    # =========================
    # Wrap up
    return dynamic_model, ae_trained_model, fusion_trained_model, sgw, C1, C2



def _extractAlignIdx(coup_mat, source_idx):
    used_source_idx = source_idx[np.where(np.asarray(np.sum(coup_mat[source_idx, :], axis=1)).squeeze() != 0)[0]]
    if len(used_source_idx) == 0:
        return None, None
    source_mat = coup_mat[used_source_idx, :]
    target_idx = np.asarray(np.argmax(source_mat, axis=1)).squeeze()
    return used_source_idx, target_idx


def _mod_distance(feature_mat, n_neighbors, metric):
    # Compute intra-modality distance matrix
    # Reference: https://github.com/rsinghlab/SCOT
    start = time.time()
    # cosine, correlation
    knn_graph = kneighbors_graph(feature_mat, n_neighbors, mode="connectivity", metric=metric, include_self=True)
    # Compute shortest distances
    shortest_path = dijkstra(csgraph=csr_matrix(knn_graph), directed=False, return_predecessors=False)
    # shortest_path = dijkstra(csgraph=csr_matrix(knn_graph), directed=False, return_predecessors=False, limit=5)
    # Deal with unconnected stuff (infinities):
    distance_max = np.nanmax(shortest_path[shortest_path != np.inf])
    shortest_path[shortest_path > distance_max] = distance_max
    # shortest_path = csr_matrix(shortest_path)
    end = time.time()
    time_cost = end - start
    print("[Intra-Mod. KNN | {}] Time cost = {:.4f} secs".format(metric, time_cost))
    return shortest_path


# =============================================

def scMultiNODEPredict(dynamic_model, first_tp_data, rna_tps, atac_tps, n_cells):
    '''
    scMultiNODE predicts expressions.
    :param latent_ode_model (torch.Model): scNODE model.
    :param first_tp_data (torch.FloatTensor): Expression at the first timepoint.
    :param tps (torch.FloatTensor): A list of timepoints to predict.
    :param n_cells (int): The number of cells to predict at each timepoint.
    :param batch_size (None or int): Either None indicates predicting in a whole or an integer representing predicting
                                     batch-wise to save computational costs. Default as None.
    :return: (torch.FloatTensor) Predicted expression with the shape of (# cells, # tps, # genes).
    '''
    dynamic_model.eval()
    rna_latent_seq, atac_latent_seq, rna_recon_obs, atac_recon_obs = dynamic_model.predict(
        first_tp_data, rna_tps, atac_tps, n_cells=n_cells
    )
    return rna_recon_obs, atac_recon_obs
