# modal_integration

Multi-modal integration with scMultiNODE model and baselines on six single-cell datasets.

The preprocessed data can be downloaded from [here](https://doi.org/10.6084/m9.figshare.27420657.v2) (for HC, HO, DR, and MN data) and 
[here](https://doi.org/10.6084/m9.figshare.29737034.v1) (for ZB and AM data).
You can put preprocessed data in the `data` directory, otherwise, you should specify the data file path in [./Modal_Integration_scMultiNODE.py](./Modal_Integration_scMultiNODE.py).

We provide precomputed integration and evaluation metrics of each model for all datasets [here (modal_integration.zip)](https://doi.org/10.6084/m9.figshare.27418872.v4). 
You can download them and put in the [modal_integration/res](./res) directory.



## Model running

- [./Modal_Integration_scMultiNODE.py](./Modal_Integration_scMultiNODE.py): Run scMultiNODE model on each datasets.
- [./Compare_Modal_Alignment_Metric.py](./Compare_Modal_Alignment_Metric.py): Compute evaluation metrics of each model and make comparison. 
- [./Compare_Modal_Alignment_Clustering.py](./Compare_Modal_Alignment_Clustering.py): Compute cell clustering NMI of each model and make comparison. 
- [./Compare_Modal_Integration_UMAP.py](./Compare_Modal_Integration_UMAP.py): Visualize integration with UMAP.
- [./Compare_Modal_Integration_PCA.py](./Compare_Modal_Integration_PCA.py): Visualize integration with PCA.
- [./run_scMultiNODE.py](./run_scMultiNODE.py): scMultiNODE training details.
