'''
Description:
    Apply PAGA for pseudotime estimation.
    Results will be saved to ./res/data4PAGA, if they do not exist therein.

Authro:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>

Reference:
    [1] https://scanpy-tutorials.readthedocs.io/en/latest/paga-paul15.html
'''
import os
import scanpy
import numpy as np
import pandas as pd

# ======================================================

if __name__ == '__main__':
    # Loading data
    data_name = "coassay_cortex"
    split_type = "all"
    data_type = "reduce"
    model_name_list = [
        "scMultiNODE", "SCOTv1", "SCOTv2", "Pamona", "UnionCom", "uniPort", "Seurat",
        "scNODE-RNA", "scNODE-ATAC", "Static_AE-RNA", "Static_AE-ATAC"
    ]
    for model_name in model_name_list:
        print("-" * 70)
        print("Model: {}".format(model_name))
        if "AE" in model_name or "scNODE" in model_name:
            model_name_str = model_name + "_"
        else:
            model_name_str = model_name + "-"
        # -----
        save_filename = "./res/aux_data/{}-reduce-all-{}aux_data.npy".format(data_name, model_name_str)
        print("Loading data for analysis...")
        data_res = np.load(save_filename, allow_pickle=True).item()
        # -----
        paga_filename = './res/data4PAGA/{}-{}-PAGA_pseudotime.h5ad'.format(data_name, model_name)
        if not os.path.isfile(paga_filename):
            # Construct AnnData
            print("Constructing AnnData...")
            if "AE" in model_name or "scNODE" in model_name:
                modal_latent = data_res["modal_latent"]
                modal_cell_types = data_res["modal_cell_types"]
                modal_cell_tps = data_res["modal_cell_tps"]
                umap_latent_data = data_res["umap_latent_data"]
                exp_mat = modal_latent
                meta_df = pd.DataFrame(data={
                    "cell_types": modal_cell_types,
                    "cell_tps": modal_cell_tps,
                })
                all_ann = scanpy.AnnData(X=exp_mat, obs=meta_df, obsm={"X_umap": umap_latent_data})
            else:
                rna_integrate = data_res["rna_integrate"]
                atac_integrate = data_res["atac_integrate"]
                rna_cell_types = data_res["rna_cell_types"]
                atac_cell_types = data_res["atac_cell_types"]
                rna_cell_tps = data_res["rna_cell_tps"]
                atac_cell_tps = data_res["atac_cell_tps"]
                umap_latent_data = data_res["umap_latent_data"]
                umap_model = data_res["umap_model"]  # Note: The UMAP is for the model latent space
                exp_mat = np.concatenate([rna_integrate, atac_integrate], axis=0)
                meta_df = pd.DataFrame(data={
                    "cell_types": np.concatenate([rna_cell_types, atac_cell_types], axis=0),
                    "cell_tps": np.concatenate([rna_cell_tps, atac_cell_tps], axis=0),
                    "mod": np.concatenate(
                        [np.repeat("rna", rna_integrate.shape[0]), np.repeat("atac", atac_integrate.shape[0])], axis=0)
                })
                all_ann = scanpy.AnnData(X=exp_mat, obs=meta_df, obsm={"X_umap": umap_latent_data})
            print(all_ann)
            # ----------
            print("Run Leiden clustering...")
            scanpy.pp.neighbors(all_ann, n_pcs=0, n_neighbors=20, use_rep=None)
            scanpy.tl.leiden(all_ann, key_added="leiden_1.0", resolution=1.0)
            print(all_ann)
            # ----------
            # Run PAGA following its tutorial:
            # https://scanpy-tutorials.readthedocs.io/en/latest/paga-paul15.html
            print("Run PAGA...")
            scanpy.tl.draw_graph(all_ann, init_pos='X_umap')
            scanpy.tl.paga(all_ann, groups='leiden_1.0')
            scanpy.pl.paga(all_ann, color='leiden_1.0', edge_width_scale=0.3)
            print(all_ann)
            # -----
            # Computing pseudotime with DPT
            all_ann.uns['iroot'] = np.flatnonzero(all_ann.obs['cell_tps'] == 0)[10]
            scanpy.tl.dpt(all_ann)
            # all_ann = all_ann[all_ann.obs.dpt_pseudotime <= 0.5]
            scanpy.pl.draw_graph(
                all_ann, color=['leiden_1.0', "cell_tps", 'dpt_pseudotime'],
                legend_loc='on data', legend_fontsize='xx-small'
            )
            print(all_ann)
            all_ann.write(paga_filename)
        else:
            print("PAGA pseudotime estimations already exist in ./res/data4PAGA")