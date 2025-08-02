import numpy as np


def loadscMultiNODELatent_withSupervision(data_name, data_type, split_type, latent_dim):
    m = "scMultiNODE"
    save_filename = "./res/{}-{}-{}-{}-{}dim.npy"
    res = np.load(save_filename.format(data_name, data_type, split_type, m, latent_dim), allow_pickle=True).item()
    rna_integrated = res["rna_integrated"]
    atac_integrated = res["atac_integrated"]
    integrated_dict = {"scMultiNODE-w-supervision": {"rna": rna_integrated, "atac": atac_integrated}}
    return integrated_dict