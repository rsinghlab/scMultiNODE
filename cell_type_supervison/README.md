# cell_type_supervision

Investigate the effect of using cell type supervision into our scMultiNODE framework.


We provide precomputed integration and evaluation metrics ([cell_type_supervision.zip](https://doi.org/10.6084/m9.figshare.27418872.v4)). 
You can download them and put in the [cell_type_supervision/res](./res) directory.



## Model running


- [./Modal_Integration_w_Supervison_scMultiNODE.py](./Modal_Integration_w_Supervison_scMultiNODE.py): Run scMultiNODE w/ supervision on all datasets.
- [./Compute_Modal_Integration_Metric-w_supervision.py](./Compute_Modal_Integration_Metric-w_supervision.py): Compute evaluation metrics for integration. 
- [./Compute_Modal_Integration_Clustering-w_supervision.py](./Compute_Modal_Integration_Clustering-w_supervision.py): Compute cell clustering NMI for integration. 
- [./Compute_Modal_Integration_UMAP-w_supervision.py](./Compute_Modal_Integration_UMAP-w_supervision.py): Visualize integration with UMAP.
- [./running.py](./running.py) and [./run_scMulti_w_Supervison.py](./run_scMulti_w_Supervison.py): scMultiNODE model training with cell type supervision.