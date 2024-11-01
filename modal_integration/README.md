# modal_integration

Compare our scMultiNODE model with baselines on four single-cell datasets.

The preprocessed data can be downloaded from [here](https://doi.org/10.6084/m9.figshare.27420657.v1).
You can put preprocessed data in the `data` directory, otherwise, you should specify the data file path in [./Modal_Integration_scMultiNODE.py](./Modal_Integration_scMultiNODE.py).

We provide precomputed integration and evaluation metrics of each model for all datasets ([modal_integration.zip](https://doi.org/10.6084/m9.figshare.27418872.v2)). 
You can put download them and put them in the [modal_integration/res](./res) directory.

*We will update the baseline model scripts shortly.*


## Model running

- [./Modal_Integration_scMultiNODE.py](./Modal_Integration_scMultiNODE.py): Run scMultiNODE model on four datasets (HC, HO, DR, and MN)
- [./Compare_Modal_Alignment_Metric.py](./Compare_Modal_Alignment_Metric.py): Compute evaluation metrics of each model and make comparison. 
- [./Compare_Modal_Integration_UMAP.py](./Compare_Modal_Integration_UMAP.py): Visualize integration of each model and make comparison. 
- [./Paper_Metric_Plotting.py](./Paper_Metric_Plotting.py): Figures plotting for the paper (Fig. 3).
