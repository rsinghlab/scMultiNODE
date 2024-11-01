'''
Description:
    The main class of our scMultiNODE model.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

# ===========================================

class scMultiNODE(nn.Module):
    '''
    scMultiNODE model: dynamic learning for temporal multi-modal single-cell data.
    '''
    def __init__(self, n_genes, n_peaks, latent_dim, rna_enc, rna_dec, atac_enc, atac_dec, fusion_layer, diffeq_decoder, anchor_mod):
        '''
        Initialize scMultiNODE model.
        :param input_dim (int): Input space size.
        :param latent_dim (int): Latent space size.
        :param output_dim (int): Output space size.
        :param rna_enc (LinearNet): RNA encoder.
        :param rna_dec (LinearNet): RNA decoder.
        :param atac_enc (LinearNet): ATAC encoder.
        :param atac_dec (LinearSigmoidNet): ATAC encoder.
        :param fusion_layer (LinearVAENet): Fusion layer.
        :param diffeq_decoder (ODE): Differential equation solver.
        '''
        super(scMultiNODE, self).__init__()
        self.n_genes = n_genes
        self.n_peaks = n_peaks
        self.latent_dim = latent_dim
        # -----
        self.rna_enc = rna_enc
        self.rna_dec = rna_dec
        self.atac_enc = atac_enc
        self.atac_dec = atac_dec
        self.fusion_layer = fusion_layer
        self.diffeq_decoder = diffeq_decoder
        self.anchor_mod = anchor_mod
        if anchor_mod == "rna":
            self.anchor_enc = self.rna_enc
        elif anchor_mod == "atac":
            self.anchor_enc = self.atac_enc
        else:
            raise ValueError("Unknown anchor modality {}.".format(anchor_mod))


    def forward(self, first_tp_data, rna_tp, atac_tp, batch_size=None):
        '''
        scMultiNODE generative process.
        :param first_tp_data (torch.FloatTensor): Data at the first timepoint
        :param rna_tp (torch.FloatTensor): A list of timepoints to predict for RNA.
        :param atac_tp (torch.FloatTensor): A list of timepoints to predict for ATAC.
        :param batch_size (int or None): The batch size (default is None).
        '''
        if batch_size is not None:
            cell_idx = np.random.choice(np.arange(first_tp_data.shape[0]), size = batch_size, replace = (first_tp_data.shape[0] < batch_size))
            first_tp_data = first_tp_data[cell_idx, :]
        # Map data at the first timepoint to the latent space
        first_latent_sample = self.fusion_layer(self.anchor_enc(first_tp_data))
        # Predict forward with ODE solver in the latent space
        rna_latent_seq = self.diffeq_decoder(first_latent_sample, rna_tp)
        atac_latent_seq = self.diffeq_decoder(first_latent_sample, atac_tp)
        # Convert latent variables (at all timepoints) back to the gene space
        rna_recon_obs = self.rna_dec(rna_latent_seq) # (batch size, # tps, # genes)
        atac_recon_obs = self.atac_dec(atac_latent_seq) # (batch size, # tps, # peaks)
        return rna_recon_obs, atac_recon_obs, first_tp_data, rna_latent_seq, atac_latent_seq


    def predict(self, first_tp_data, rna_tp, atac_tp, n_cells):
        '''
        Predicts at given timepoints.
        :param first_tp_data (torch.FloatTensor): Expression at the first timepoint.
        :param rna_tp (torch.FloatTensor): A list of timepoints to predict for RNA.
        :param atac_tp (torch.FloatTensor): A list of timepoints to predict for ATAC.
        :param n_cells (int): The number of cells to predict.
        '''
        first_latent_sample = self.fusion_layer(self.anchor_enc(first_tp_data))
        repeat_times = (n_cells // first_latent_sample.shape[0]) + 1
        first_latent_sample = torch.repeat_interleave(first_latent_sample, repeat_times, dim=0)[:n_cells, :]
        rna_latent_seq = self.diffeq_decoder(first_latent_sample, rna_tp)
        atac_latent_seq = self.diffeq_decoder(first_latent_sample, atac_tp)
        # Convert latent variables (at all timepoints) back to the gene space
        rna_recon_obs = self.rna_dec(rna_latent_seq)  # (batch size, # tps, # genes)
        atac_recon_obs = self.atac_dec(atac_latent_seq)  # (batch size, # tps, # peaks)
        return rna_latent_seq, atac_latent_seq, rna_recon_obs, atac_recon_obs


    def _sampleGaussian(self, mean, std):
        '''
        Sampling with the re-parametric trick.
        '''
        d = dist.normal.Normal(torch.Tensor([0.]), torch.Tensor([1.]))
        r = d.sample(mean.size()).squeeze(-1)
        x = r * std.float() + mean.float()
        return x
