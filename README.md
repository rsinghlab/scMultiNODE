# scMultiNODE: Integrative and Scalable Framework for Multi-Modal Temporal Single-Cell Data

We introduce scMultiNODE, an unsupervised integration model that combines gene expression and chromatin accessibility 
measurements in developing single cells, while preserving cell type variations and cellular dynamics. First, scMultiNODE 
uses a scalable, Quantized Gromov-Wasserstein optimal transport to align a large number of cells across different 
measurements. Next, it utilizes neural ordinary differential equations to explicitly model cell development with a 
regularization term to learn a dynamic latent space.
[(bioRxiv preprint)](https://www.biorxiv.org/content/10.1101/2024.10.27.620531v2)

![scMultiNODE model overview](https://github.com/rsinghlab/scMultiNODE/blob/main/model_illustration.jpg?raw=true)

**If you have questions or find any problems with our codes, feel free to submit issues or send emails to jiaqi_zhang2@brown.edu or other corresponding authors.**


## Requirements

Our codes have been tested in Python 3.7. Required packages are listed in [./installation](./installation).

## Data

- Raw and preprocessed data of six temporally resolved multi-modal single-cell datasets can be downloaded 
from [here](https://doi.org/10.6084/m9.figshare.27420657.v2) (for HC, HO, DR, and MN data) and 
[here](https://doi.org/10.6084/m9.figshare.29737034.v1) (for ZB and AM data).

- All model integrations and corresponding evaluation metrics on six datasets are available at [here (modal_integration.zip)](https://doi.org/10.6084/m9.figshare.27418872.v4).

- Investigation of scMultiNODE with cell type supervision are available at [here (cell_type_supervision.zip)](https://doi.org/10.6084/m9.figshare.27418872.v4).

- Experiment results for downstream analysis (cell trajectory pseudotime estimation, cell path construction, and 
cross-modal cell label transfer) are available at [here (downstream_analysis.zip)](https://doi.org/10.6084/m9.figshare.27418872.v4).

- Data visualization and figures in the paper are available at [here (journal_figs.zip)](https://doi.org/10.6084/m9.figshare.27418872.v4).

-----


## Models

- scMultiNODE is implemented in [./model/dynamic_model.py](./model/dynamic_model.py). 

- All baseline codes are provided in [baseline](./baseline). See the documentation therein for more details.


## Example Usage

The script of using scMultiNODE for integration is shown in [./modal_integration/Modal_Integration_scMultiNODE.py](./modal_integration/Modal_Integration_scMultiNODE.py).

-----

## Repository Structure

- [data](./data): Scripts for data preprocessing. Some scripts are implemented in R and need installation of [Seurat](https://satijalab.org/seurat/).
- [model](./model): Implementation of scMultiNODE model.
- [optim](./optim): Loss computations, QGW algorithm, evaluation metrics.
- [baseline](./baseline): Implementation of baseline models.
- [modal_integration](./modal_integration): Run each model on six multi-modal single-cell datasets and compute integrations. Evaluation metrics computation and comparison.
- [downstream_analysis](./downstream_analysis): Use scMultiNODE for cell trajectory pseudotime estimation, cell path construction, and cross-modal label transfer.
- [hyperparameter_investigation](./hyperparameter_investigation): Ablation study and investigation of hyperparameter settings for scMultiNODE.
- [tuning](./tuning): Hyperparameter tuning for scMultiNODE and baselines.
- [plotting](./plotting): Visualization / figure plotting.
- [utils](./utils): Utility functions.

-----

## Bugs & Suggestions

Please report any bugs, problems, suggestions, or requests as a [Github issue](https://github.com/rsinghlab/scMultiNODE/issues) 
or send emails to *jiaqi_zhang2@brown.edu* or other corresponding authors.