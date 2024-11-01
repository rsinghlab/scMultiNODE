# scMultiNODE: Temporal Single-Cell Data Integration across Unaligned Modalities

scMultiNODE is a model that integrates gene expression and chromatin accessibility measurements in developing single cells 
while preserving cell type variations and cellular dynamics. scMultiNODE uses autoencoders (AEs) to learn nonlinear low-dimensional 
cell representation and optimal transport to align cells across different measurements. Next, it utilizes neural ordinary 
differential equations (ODEs) to explicitly model cell development with a regularization term to learn a dynamic latent space.
[(bioRxiv preprint)](https://www.biorxiv.org/content/10.1101/2024.10.27.620531v1)

![scMultiNODE model overview](https://github.com/rsinghlab/scMultiNODE/blob/main/model_illustration.jpg?raw=true)

**If you have questions or find any problems with our codes, feel free to submit issues or send emails to jiaqi_zhang2@brown.edu or other corresponding authors.**

*(11/01/2024 updates) We have updated major parts of the experiments corresponding to our paper, including scMultiNODE 
implementation and its integration, integration performance comparison, and downstream analysis.*


## Requirements

Our codes have been tested in Python 3.7. Required packages are listed in [./installation](./installation).

## Data

- Raw and preprocessed data of four temporal multi-modal single-cell datasets can be downloaded from [here](https://doi.org/10.6084/m9.figshare.27420657.v1).
- All model integrations and corresponding evaluation metrics on four datasets are available at [here (modal_integration.zip)](https://doi.org/10.6084/m9.figshare.27418872.v2).
- Experiment results for downstream analysis (cell path construction and DE genes detection) are available at [here (downstream_analysis.zip)](https://doi.org/10.6084/m9.figshare.27418872.v2).
- Data visualization and figures in the paper are available at [here (figs.zip)](https://doi.org/10.6084/m9.figshare.27418872.v2).


## Models

scMultiNODE is implemented in [./model/dynamic_model.py](./model/dynamic_model.py). 


## Example Usage

The script of using scMultiNODE for integration is shown in [./modal_integration/Modal_Integration_scMultiNODE.py](./modal_integration/Modal_Integration_scMultiNODE.py).


## Repository Structure

- [data](./data): Scripts for data preprocessing. Some scripts are implemented in R and need installation of [Seurat](https://satijalab.org/seurat/).
- [model](./model): Implementation of scMultiNODE model.
- [optim](./optim): Loss computations, QGW algorithm, evaluation metrics.
- [baseline](./baseline): Implementation of baseline models.
- [modal_integration](./modal_integration): Run each model on four single-cell datasets and compute integrations. Evaluation metrics computation and comparison.
- [downstream_analysis](./downstream_analysis): Use scMultiNODE for cell path construction and finding driver genes.
- [hyperparameter_investigation](./hyperparameter_investigation): Ablation study and investigation of hyperparameter settings.
- [tuning](./tuning): Hyperparameter tuning.
- [plotting](./plotting): Integration visualization. Compare model predictions. Paper figures plotting.
- [utils](./utils): Utility functions.


## Bugs & Suggestions

Please report any bugs, problems, suggestions, or requests as a [Github issue](https://github.com/rsinghlab/scMultiNODE/issues)

