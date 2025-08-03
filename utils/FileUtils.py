'''
Description:
    Utility functions for loading data.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import numpy as np
import scanpy
import pandas as pd
import natsort

# --------------------------------
# Load multi-modal datasets

def loadDrosophilaData(data_dir, split_type):
    # Load RNA data
    cnt_data = pd.read_csv("{}/{}-RNA_count_data.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/rna_meta.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.set_index("cell")
    meta_data = meta_data.loc[cnt_data.index,:]
    cell_stage = meta_data["time"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage), ))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_rna_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    ann_rna_data.obs["cell_type"] = ann_rna_data.obs["manual_annot"]
    # -----
    # Load ATAC data
    cnt_data = pd.read_csv("{}/{}-ATAC_count_data.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/atac_meta.csv".format(data_dir), header=0, index_col=0)
    meta_data = meta_data.set_index("cell")
    meta_data = meta_data.loc[cnt_data.index, :]
    cell_stage = meta_data["time"]
    unique_cell_stages = natsort.natsorted(np.unique(cell_stage))
    cell_tp = np.zeros((len(cell_stage),))
    cell_tp[cell_tp == 0] = np.nan
    for idx, s in enumerate(unique_cell_stages):
        cell_tp[np.where(cell_stage == s)[0]] = idx
    cell_tp += 1
    meta_data["tp"] = cell_tp
    ann_atac_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    ann_atac_data.obs["cell_type"] = ann_atac_data.obs["refined_annotation"]
    return ann_rna_data, ann_atac_data


def loadCoassayCortex(data_dir, split_type):
    # Load RNA data
    cnt_data = pd.read_csv("{}/{}-RNA-data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-RNA-cell_meta.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data["tp"] = meta_data["tp"] + 1
    ann_rna_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    # -----
    # Load ATAC data
    cnt_data = pd.read_csv("{}/{}-ATAC-data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-ATAC-cell_meta.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data["tp"] = meta_data["tp"] + 1
    ann_atac_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    return ann_rna_data, ann_atac_data


def loadHumanOrganoidData(data_dir, split_type):
    # Load RNA data
    cnt_data = pd.read_csv("{}/{}-RNA-data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-RNA-cell_meta.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data["tp"] = meta_data["tp"] + 1
    ann_rna_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    # -----
    # Load ATAC data
    cnt_data = pd.read_csv("{}/{}-ATAC-data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-ATAC-cell_meta.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data["tp"] = meta_data["tp"] + 1
    ann_atac_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    return ann_rna_data, ann_atac_data


def loadZebrafishRetinaData(data_dir, split_type):
    # Load RNA data
    cnt_data = pd.read_csv("{}/{}-RNA-data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-RNA-cell_meta.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data["tp"] = meta_data["tp"] + 1
    ann_rna_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    # -----
    # Load ATAC data
    cnt_data = pd.read_csv("{}/{}-ATAC-data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-ATAC-cell_meta.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data["tp"] = meta_data["tp"] + 1
    ann_atac_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    return ann_rna_data, ann_atac_data


def loadMouseNeocortexData(data_dir, split_type):
    # Load RNA data
    cnt_data = pd.read_csv("{}/{}-RNA-data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-RNA-cell_meta.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data["tp"] = meta_data["tp"] + 1
    meta_data["cell_type"] = meta_data["CellType"]
    ann_rna_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    # -----
    # Load ATAC data
    cnt_data = pd.read_csv("{}/{}-ATAC-data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-ATAC-cell_meta.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data["tp"] = meta_data["tp"] + 1
    meta_data["cell_type"] = meta_data["CellType"]
    ann_atac_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    return ann_rna_data, ann_atac_data


def loadZebrahubCortex(data_dir, split_type):
    # Load RNA data
    cnt_data = pd.read_csv("{}/{}-RNA-data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-RNA-cell_meta.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data["tp"] = meta_data["tp"] + 1
    ann_rna_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    # -----
    # Load ATAC data
    cnt_data = pd.read_csv("{}/{}-ATAC-data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-ATAC-cell_meta.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data["tp"] = meta_data["tp"] + 1
    ann_atac_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    return ann_rna_data, ann_atac_data


def loadAmphioxus(data_dir, split_type):
    # Load RNA data
    cnt_data = pd.read_csv("{}/{}-RNA-data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-RNA-cell_meta.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data["tp"] = meta_data["tp"] + 1
    ann_rna_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    ann_rna_data.obs.cell_type = ann_rna_data.obs.cell_type.apply(lambda x: "unknown" if x=="Unassigned" else x)
    # -----
    # Load ATAC data
    cnt_data = pd.read_csv("{}/{}-ATAC-data-hvg.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data = pd.read_csv("{}/{}-ATAC-cell_meta.csv".format(data_dir, split_type), header=0, index_col=0)
    meta_data["tp"] = meta_data["tp"] + 1
    ann_atac_data = scanpy.AnnData(X=cnt_data, obs=meta_data, dtype=np.float32)
    ann_atac_data.obs.cell_type = ann_atac_data.obs.cell_type.apply(lambda x: "unknown" if x == "Unassigned" else x)
    return ann_rna_data, ann_atac_data


# --------------------------------
# Dataset directories
coassay_cortex_dir = "../data/human_prefrontal_cortex_multiomic/reduce_processed/"
human_organoid_dir = "../data/human_organoid_Fleck2022/reduce_processed/"
drosophila_dir = "../data/drosophila_embryonic/reduce_processed/"
mouse_neocortex_dir = "../data/Yuan2022_MouseNeocortex/reduce_processed"
zebrahub_dir = "../data/Kim2024_Zebrahub/reduce_processed"
amphioxus_dir = "../data/Ma2022_Amphioxus/reduce_processed"


def loadSCData(data_name, data_type, split_type, data_dir=None):
    '''
    Main function to load scRNA-seq dataset and pre-process it.
    '''
    print("[ Data={}/{} ] Loading data...".format(data_name, data_type))
    if data_name == "drosophila":
        data_dir = drosophila_dir if data_dir is None else data_dir
        ann_rna_data, ann_atac_data = loadDrosophilaData(data_dir, split_type)
        print("Pre-processing...")
        ann_rna_data.X = ann_rna_data.X.astype(float)
        processed_data = preprocess(ann_rna_data.copy()) # preprocess RNA data
        ann_rna_data = processed_data
    elif data_name == "coassay_cortex":
        data_dir = coassay_cortex_dir if data_dir is None else data_dir
        # this dataset is already normalized
        ann_rna_data, ann_atac_data = loadCoassayCortex(data_dir, split_type)
        ann_rna_data.X = ann_rna_data.X.astype(float)
    elif data_name == "human_organoid":
        data_dir = human_organoid_dir if data_dir is None else data_dir
        ann_rna_data, ann_atac_data = loadHumanOrganoidData(data_dir, split_type)
        print("Pre-processing...")
        ann_rna_data.X = ann_rna_data.X.astype(float)
        processed_data = preprocess(ann_rna_data.copy()) # preprocess RNA data
        ann_rna_data = processed_data
    elif data_name == "mouse_neocortex":
        data_dir = mouse_neocortex_dir if data_dir is None else data_dir
        ann_rna_data, ann_atac_data = loadMouseNeocortexData(data_dir, split_type)
        print("Pre-processing...")
        ann_rna_data.X = ann_rna_data.X.astype(float)
        processed_data = preprocess(ann_rna_data.copy())
        ann_rna_data = processed_data
        ann_atac_data.X = ann_atac_data.X.astype(float)
    elif data_name == "zebrahub":
        data_dir = zebrahub_dir if data_dir is None else data_dir
        ann_rna_data, ann_atac_data = loadZebrahubCortex(data_dir, split_type)
        print("Pre-processing...")
        ann_rna_data.X = ann_rna_data.X.astype(float)
        ann_atac_data.X = ann_atac_data.X.astype(float)
        ann_atac_data = preprocessLog(ann_atac_data.copy()) # zebrahub provides gene activity matrix for ATAC
    elif data_name == "amphioxus":
        data_dir = amphioxus_dir if data_dir is None else data_dir
        ann_rna_data, ann_atac_data = loadAmphioxus(data_dir, split_type)
        print("Pre-processing...")
        ann_rna_data.X = ann_rna_data.X.astype(float)
        processed_data = preprocess(ann_rna_data.copy())
        ann_rna_data = processed_data
        ann_atac_data.X = ann_atac_data.X.astype(float)
        ann_atac_data = binarize(ann_atac_data.copy())
    else:
        raise ValueError("Unknown data name.")
    rna_cell_tps = ann_rna_data.obs["tp"]
    atac_cell_tps = ann_atac_data.obs["tp"]
    rna_n_tps = len(np.unique(rna_cell_tps))
    atac_n_tps = len(np.unique(atac_cell_tps))
    n_genes = ann_rna_data.shape[1]
    n_peaks = ann_atac_data.shape[1]
    return (
        ann_rna_data, ann_atac_data, rna_cell_tps, atac_cell_tps,
        rna_n_tps, atac_n_tps, n_genes, n_peaks
    )


def tpSplitInd(data_name, split_type):
    '''
    Get the training/testing timepoint split for each dataset.
    '''
    if data_name == "drosophila":
        if split_type == "all":
            rna_train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            rna_test_tps = []
            atac_train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            atac_test_tps = []
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "coassay_cortex":
        if split_type == "all":
            rna_train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            rna_test_tps = []
            atac_train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            atac_test_tps = []
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "human_organoid":
        if split_type == "all":
            rna_train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            rna_test_tps = []
            atac_train_tps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            atac_test_tps = []
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "mouse_neocortex":
        if split_type == "all":
            rna_train_tps = [0, 1, 2]
            rna_test_tps = []
            atac_train_tps = [0, 1, 2]
            atac_test_tps = []
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "amphioxus":
        if split_type == "all":
            rna_train_tps = [0, 1, 2, 3, 4, 5]
            rna_test_tps = []
            atac_train_tps = [0, 1, 2, 3, 4, 5]
            atac_test_tps = []
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    elif data_name == "zebrahub":
        if split_type == "all":
            rna_train_tps = [0, 1, 2, 3, 4, 5]
            rna_test_tps = []
            atac_train_tps = [0, 1, 2, 3, 4, 5]
            atac_test_tps = []
        else:
            raise ValueError("Unknown split type {}!".format(split_type))
    else:
        raise ValueError("Unknown data name.")
    return rna_train_tps, atac_train_tps, rna_test_tps, atac_test_tps


# ---------------------------------

def preprocess(ann_data):
    # adopt recipe_zheng17 w/o HVG selection
    # omit scaling part to avoid information leakage
    scanpy.pp.normalize_per_cell(  # normalize with total UMI count per cell
        ann_data, key_n_counts='n_counts_all', counts_per_cell_after=1e4
    )
    scanpy.pp.log1p(ann_data)  # log transform: adata.X = log(adata.X + 1)
    return ann_data


def preprocessLog(ann_data):
    scanpy.pp.log1p(ann_data)  # log transform: adata.X = log(adata.X + 1)
    return ann_data


def binarize(ann_data):
    ann_data.X = np.where(ann_data.X > 0, 1.0, 0.0)
    return ann_data

# ---------------------------------
from datetime import datetime

def getTimeStr():
    now = datetime.now()  # current date and time
    time_str = now.strftime("%Y%m%d%H%M%S")
    return time_str

# ---------------------------------

def loadAELatent(data_name, data_type, split_type, latent_dim, file_dir="./res/preprocess_latent"):
    res = np.load("{}/{}-{}-{}-ae_latent-{}dim.npy".format(file_dir, data_name, data_type, split_type, latent_dim), allow_pickle=True).item()
    rna_data = res["rna_data"].detach().numpy()
    atac_data = res["atac_data"].detach().numpy()
    rna_recon = res["rna_recon"].detach().numpy()
    atac_recon = res["atac_recon"].detach().numpy()
    rna_latent = res["rna_latent"].detach().numpy()
    atac_latent = res["atac_latent"].detach().numpy()
    rna_tps = res["rna_tps"]
    atac_tps = res["atac_tps"]
    rna_cell_types = res["rna_cell_types"]
    atac_cell_types = res["atac_cell_types"]
    return rna_data, atac_data, rna_recon, atac_recon, rna_latent, atac_latent, rna_tps, atac_tps, rna_cell_types, atac_cell_types


# ---------------------------------

def loadIntegratedLatent(data_name, data_type, split_type, model_list, latent_dim):
    save_filename = "./res/model_latent/{}-{}-{}-{}-{}dim.npy"
    seurat_save_filename = "./res/model_latent/{}-{}-{}-{}-{}dim-{}.csv"
    integrated_dict = {}
    for m in model_list:
        if m == "Seurat":
            rna_integrated = pd.read_csv(seurat_save_filename.format(data_name, data_type, split_type, m, latent_dim, "rna"), header=0, index_col=None).values
            atac_integrated = pd.read_csv(seurat_save_filename.format(data_name, data_type, split_type, m, latent_dim, "atac"), header=0, index_col=None).values
        else:
            res = np.load(save_filename.format(data_name, data_type, split_type, m, latent_dim), allow_pickle=True).item()
            rna_integrated = res["rna_integrated"]
            atac_integrated = res["atac_integrated"]
        integrated_dict[m] = {"rna": rna_integrated, "atac": atac_integrated}
    return integrated_dict