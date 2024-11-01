'''
Description:
    Utility functions for pre-processing data.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import scanpy

# ==============================================================


def sampleGaussian(mean, std):
    '''
    Sampling with the re-parametric trick.
    '''
    d = dist.normal.Normal(torch.Tensor([0.]), torch.Tensor([1.]))
    r = d.sample(mean.size()).squeeze(-1)
    x = r * std.float() + mean.float()
    return x


# ==============================================================
# Reference: https://github.com/colomemaria/epiScanpy/blob/master/episcanpy/preprocessing/_quality_control.py
def select_var_feature(adata, nb_features=None):
    """
    This function computes a variability score to rank the most variable features across all cells.
    Then it selects the most variable features according to a specified number of features.

    Parameters
    ----------
    adata: adata object

    nb_features: default value is None, if specify it will select a the top most variable features.
    if the nb_features is larger than the total number of feature, it filters based on the max_score argument
    """
    adata = adata.copy()
    # calculate variability score
    _cal_var(adata) # adds variability score for each feature
    # adata.var['variablility_score'] = abs(adata.var['prop_shared_cells']-0.5)
    var_annot = adata.var.sort_values(ascending=False, by ='variability_score')
    # calculate the max score to get a specific number of feature
    min_score = var_annot['variability_score'][nb_features]
    adata_tmp = adata[:, adata.var['variability_score'] >= min_score]
    return (adata_tmp)


def _cal_var(adata):
    """
    Show distribution plots of cells sharing features and variability score.
    """
    adata.var['n_cells'] = adata.X.sum(axis=0)
    adata.var['prop_shared_cells'] = adata.var['n_cells'] / len(adata.obs_names.tolist())
    adata.var['variability_score'] = [1 - abs(n - 0.5) for n in adata.var['prop_shared_cells']]
