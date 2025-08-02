import scanpy
import pandas as pd
import numpy as np
from scipy.io import mmread, mmwrite
import scipy.sparse
from data.ATAC_HVF import select_var_feature



def subsetRNA():
    print("-" * 70)
    print("Loading RNA meta...")
    rna_meta_df = pd.read_csv("./raw/metadata_merged.csv")
    rna_meta_df["cell_type"] = rna_meta_df["lineage"]
    rna_meta_df["age"] = rna_meta_df["stage_name"]
    rna_meta_df = rna_meta_df.set_index("index")
    rna_subset_idx = np.where(np.isin(rna_meta_df.age, np.asarray(["B", "G3", "G6", "N1", "N3", "L0"])))[0]
    sub_rna_meta_df = rna_meta_df.iloc[rna_subset_idx]
    print(rna_meta_df)
    print(rna_meta_df.age.unique())
    print(rna_meta_df.cell_type.unique())
    print(sub_rna_meta_df)
    print(sub_rna_meta_df.age.unique())
    print(sub_rna_meta_df.cell_type.unique())
    # -----
    print("-" * 70)
    print("Loading RNA...")
    rna_cell_name = pd.read_csv("./raw/Embryos_genes.barcodes.txt.gz", compression="gzip", header=None).values.squeeze()
    rna_gene_name = pd.read_csv("./raw/Embryos_genes.genes.txt.gz", compression="gzip", header=None).values.squeeze()

    cell_df = pd.DataFrame(data={"barcode": rna_cell_name, "ind": np.arange(len(rna_cell_name))})
    cell_df = cell_df.set_index("barcode")
    cell_idx = cell_df.loc[sub_rna_meta_df.index.values, :].ind.values
    rna_cell_name = rna_cell_name[cell_idx]
    sub_rna_meta_df = sub_rna_meta_df.loc[rna_cell_name]
    print(sub_rna_meta_df.shape)
    print(rna_cell_name.shape)

    rna_mat = mmread("./raw/Embryos_genes.mtx.gz").tocsr()
    print(rna_mat.shape)
    sub_rna_mat = rna_mat[cell_idx, :].tocoo()
    print(sub_rna_mat.shape)
    # -----
    mmwrite("./combined/rna_mat.mtx", sub_rna_mat) # cell-by-gene
    pd.DataFrame({"cell": rna_cell_name}).to_csv("./combined/rna_cell_names.csv")
    pd.DataFrame({"gene": rna_gene_name}).to_csv("./combined/rna_gene_names.csv")
    sub_rna_meta_df.to_csv("./combined/rna_cell_meta.csv")


def subsetATAC():
    print("-" * 70)
    print("Loading ATAC meta...")
    atac_meta_df = pd.read_csv("./raw/metadata_merged-ATAC.csv")
    atac_meta_df["cell_type"] = atac_meta_df["lineage"]
    atac_meta_df["age"] = atac_meta_df["stage_name"]
    atac_meta_df = atac_meta_df.set_index("index")
    # atac_meta_df = atac_meta_df.iloc[rand_idx, :]
    print(atac_meta_df)
    print(atac_meta_df.age.unique())
    print(atac_meta_df.cell_type.unique())
    atac_meta_df.index = [x.split("-")[0] for x in atac_meta_df.index.values]
    # -----
    atac_mat_list = []
    peak_name_list = []
    print("Loading ATAC data...")
    for t in ["blastula", "G3", "G6", "N1", "N3", "L0"]:
        atac_mat = pd.read_csv("./raw/{}_scATAC".format(t), header=0, index_col=0).T # cell-by-peak
        print(atac_mat.shape)
        atac_mat.index = [x.split(".")[0] for x in atac_mat.index.values]
        atac_mat_list.append(atac_mat)
        peak_name_list.append(set(atac_mat.columns.values.tolist()))
    common_peak = set.intersection(*peak_name_list)
    print("Num of common peaks: ", len(common_peak))
    merge_atac = pd.concat([x.loc[:, common_peak] for x in atac_mat_list], axis=0)
    print(merge_atac.shape)
    sub_merge_atac = merge_atac.loc[atac_meta_df.index, :]
    print(sub_merge_atac.shape)
    # -----
    mmwrite("./combined/atac_mat.mtx", scipy.sparse.coo_matrix(sub_merge_atac.values))  # cell-by-gene
    pd.DataFrame({"cell": sub_merge_atac.index.values}).to_csv("./combined/atac_cell_names.csv")
    pd.DataFrame({"gene": sub_merge_atac.columns.values}).to_csv("./combined/atac_gene_names.csv")
    atac_meta_df.to_csv("./combined/atac_cell_meta.csv")




if __name__ == '__main__':
    # subsetRNA()
    subsetATAC()