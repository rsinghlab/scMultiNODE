'''
Description:
    Select highly variable features for ATAC data.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>

Reference:
    [1] https://github.com/colomemaria/epiScanpy/blob/0498043252c58b4c4187007c215dd6d2143d121b/episcanpy/preprocessing/_quality_control.py#L64
'''
from scipy.sparse import issparse


def cal_var(adata):
    """
    Show distribution plots of cells sharing features and variability score.
    """

    if issparse(adata.X):
        adata.var['n_cells'] = adata.X.sum(axis=0).tolist()[0] ## double check this
        adata.var['prop_shared_cells'] = adata.var['n_cells']/len(adata.obs_names.tolist())
        adata.var['variability_score'] = [1-abs(n-0.5) for n in adata.var['prop_shared_cells']]
    else:
        adata.var['n_cells'] = adata.X.sum(axis=0)
        adata.var['prop_shared_cells'] = adata.var['n_cells']/len(adata.obs_names.tolist())
        adata.var['variability_score'] = [1-abs(n-0.5) for n in adata.var['prop_shared_cells']]


def select_var_feature(adata, min_score=0.5, nb_features=None, copy=False):
    """
    This function computes a variability score to rank the most variable features across all cells.
    Then it selects the most variable features according to either a specified number of features (nb_features) or a maximum variance score (max_score).

    Parameters
    ----------

    adata: adata object

    min_score: minimum threshold variability score to retain features,
    where 1 is the score of the most variable features and 0.5 is the score of the least variable features.

    nb_features: default value is None, if specify it will select a the top most variable features.
    if the nb_features is larger than the total number of feature, it filters based on the max_score argument

    show: default value True, it will plot the distribution of var.

    copy: return a new adata object if copy == True.

    Returns
    -------
    Depending on ``copy``, returns a new AnnData object or overwrite the input


    """
    if copy:
        inplace=False
    else:
        inplace=True

    adata = adata.copy() if not inplace else adata

    # calculate variability score
    cal_var(adata) # adds variability score for each feature
    # adata.var['variablility_score'] = abs(adata.var['prop_shared_cells']-0.5)
    var_annot = adata.var.sort_values(ascending=False, by ='variability_score')
    selected_vars = var_annot.iloc[:nb_features]
    # calculate the max score to get a specific number of feature
    if nb_features != None and nb_features < len(adata.var_names):
        min_score = var_annot['variability_score'][nb_features]

    adata_tmp = adata[:,adata.var['variability_score']>=min_score].copy()
    ## return the filtered AnnData objet.
    if not inplace:
        # adata_tmp = adata[:,adata.var['variability_score']>=min_score]
        adata_tmp = adata[:,selected_vars.index.values]
        return(adata_tmp)
    else:
        # adata._inplace_subset_var(adata.var['variability_score']>=min_score)
        adata._inplace_subset_var(selected_vars.id.values)