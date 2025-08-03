'''
Description:
    Cell path construction | Step III: Compare DE genes of cell path with RNA/ATAC-derived cell type marker genes.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import os
import scanpy
import pandas as pd
from utils.FileUtils import loadSCData

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)


# ======================================================

def compareDE(path_de, type_de):
    top10_de_coverage = set(path_de[:10]).intersection(set(type_de[:10]))
    top100_de_coverage = set(path_de[:100]).intersection(set(type_de[:100]))
    print("Top 10 coverage = {}".format(len(top10_de_coverage)/10))
    print("Top 100 coverage = {}".format(len(top100_de_coverage)/100))
    # -----
    top10_de_common = set(path_de[:10]).intersection(set(type_de[:10]))
    print("Top 10 DE common: ", top10_de_common)

    top10_de_diff = set(path_de[:10]).difference(set(type_de[:10]))
    print("Top 10 DE difference: ", top10_de_diff)


if __name__ == '__main__':
    # Loading data
    data_name = "coassay_cortex"
    split_type = "all"
    data_type = "reduce"
    data_dir = "../data/human_prefrontal_cortex_multiomic/reduce_processed/"
    (
        ann_rna_data, ann_atac_data, rna_cell_tps, atac_cell_tps,
        rna_n_tps, atac_n_tps, n_genes, n_peaks
    ) = loadSCData(data_name=data_name, data_type=data_type, split_type=split_type, data_dir=data_dir)
    # -----
    print("=" * 70)
    print("Compute RNA-derived cell type marker genes...")
    rna_marker_filename = "res/cell_type_marker/rna_cell_type_DE_genes_wilcoxon.csv"
    if not os.path.isfile(rna_marker_filename):
        scanpy.tl.rank_genes_groups(ann_rna_data, 'cell_type', method='wilcoxon')
        scanpy.pl.rank_genes_groups(ann_rna_data, n_genes=25, sharey=False)
        group_id = ann_rna_data.uns["rank_genes_groups"]["names"].dtype.names
        gene_names = ann_rna_data.uns["rank_genes_groups"]["names"]
        print(group_id)
        print(gene_names[:10])
        marker_gene_dict = {}
        for i, g_name in enumerate(group_id):
            g_gene = [x[i] for x in gene_names]
            marker_gene_dict[g_name] = g_gene
        pd.DataFrame(marker_gene_dict).to_csv(rna_marker_filename)
    # -----
    print("=" * 70)
    print("Compute ATAC-derived cell type marker genes...")
    atac_marker_filename = "res/cell_type_marker/atac_cell_type_DE_genes_wilcoxon.csv"
    if not os.path.isfile(rna_marker_filename):
        scanpy.tl.rank_genes_groups(ann_atac_data, 'cell_type', method='wilcoxon')
        scanpy.pl.rank_genes_groups(ann_atac_data, n_genes=25, sharey=False)
        group_id = ann_atac_data.uns["rank_genes_groups"]["names"].dtype.names
        gene_names = ann_atac_data.uns["rank_genes_groups"]["names"]
        print(group_id)
        print(gene_names[:10])
        marker_gene_dict = {}
        for i, g_name in enumerate(group_id):
            g_gene = [x[i] for x in gene_names]
            marker_gene_dict[g_name] = g_gene
        pd.DataFrame(marker_gene_dict).to_csv(atac_marker_filename)
    # -----
    # Compare DE genes detected based on cell type & path
    print("=" * 70)
    rna_marker = pd.read_csv(rna_marker_filename)  # from RNA
    atac_marker = pd.read_csv(atac_marker_filename)  # from ATAC gene activity
    path_de = pd.read_csv("res/path_DE/path_DE_genes_wilcoxon.csv")
    # -----
    OL_path_DE = path_de["OL"].values
    GN_path_DE = path_de["GN"].values
    # -----
    print("=" * 70)
    print("[ RNA ]")
    OL_type_marker = rna_marker["oligodendrocyte"].values
    GN_type_marker = rna_marker["glutamatergic neuron"].values
    print("-" * 70)
    print("-- OL --")
    compareDE(OL_path_DE, OL_type_marker)
    print("-" * 70)
    print("-- GN --")
    compareDE(GN_path_DE, GN_type_marker)
    # -----
    print("=" * 70)
    print("[ ATAC ]")
    OL_type_marker = atac_marker["oligodendrocyte"].values
    GN_type_marker = atac_marker["glutamatergic neuron"].values
    print("-" * 70)
    print("-- OL --")
    compareDE(OL_path_DE, OL_type_marker)
    print("-" * 70)
    print("-- GN --")
    compareDE(GN_path_DE, GN_type_marker)




