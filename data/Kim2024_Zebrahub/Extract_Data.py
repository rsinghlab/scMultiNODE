import numpy as np
import pandas as pd
import scanpy

print("=" * 70)
print("Loading ATAC data...")
atac_ann = scanpy.read_h5ad("./raw/zf_multiome_atlas_full_RNA_v1_release.h5ad")
print(atac_ann)

print("=" * 70)
print("Loading RNA data...")
rna_ann = scanpy.read_h5ad("./raw/zf_atlas_full_v4_release.h5ad")
print(rna_ann)

# -----

print("=" * 70)
print("Select RNA data...")
tp_list = np.asarray(['10hpf', '12hpf', '14hpf', '16hpf', '19hpf', '24hpf'])
rna_idx = np.where(np.isin(rna_ann.obs.timepoint.values, tp_list))[0]
rna_ann = rna_ann[rna_idx,:]
print(rna_ann)
print(rna_ann.obs.timepoint.unique())

print("=" * 70)
print("Subsampling...")
n_rna_cell = rna_ann.shape[0]
n_atac_cell = atac_ann.shape[0]
sample_ratio = 0.1
rna_sub_idx = np.random.choice(np.arange(n_rna_cell), int(n_rna_cell*sample_ratio), replace=False)
atac_sub_idx = np.random.choice(np.arange(n_atac_cell), int(n_atac_cell*sample_ratio), replace=False)

sub_rna_ann = rna_ann[rna_sub_idx,:]
sub_atac_ann = atac_ann[atac_sub_idx,:]
print(sub_rna_ann.shape)
print(sub_atac_ann.shape)

# -----
print("=" * 70)
print("Saving data...")
sub_rna_ann.write_h5ad("./raw/sub_rna_ann.h5ad")
sub_atac_ann.write_h5ad("./raw/sub_atac_ann.h5ad")


