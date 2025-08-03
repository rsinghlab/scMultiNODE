'''
Description:
    Integrate RNA and ATAC assays with Pamona.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>

Reference:
    [1] Cao, K., Hong, Y., & Wan, L. (2022).
        Manifold alignment for heterogeneous single-cell multi-omics data integration using Pamona.
        Bioinformatics, 38(1), 211-219.
    [2] https://github.com/caokai1073/Pamona
'''
import sys
sys.path.append("Pamona/")
import numpy as np
import baseline.Pamona.Pamona as Pamona


def PamonaAlign(rna_data, atac_data, n_shared=None, epsilon=1e-3, n_neighbors=10, Lambda=10, output_dim=10, verbose=False, return_aux=False):
    Pa = Pamona.Pamona(
        n_shared=n_shared, epsilon=epsilon, Lambda=Lambda, n_neighbors=n_neighbors,
        output_dim=output_dim, verbose=verbose
    )
    integrated_data, T = Pa.run_Pamona([rna_data, atac_data])
    rna_integrated, atac_integrated = integrated_data
    if return_aux:
        return rna_integrated, atac_integrated, T, Pa
    else:
        return rna_integrated, atac_integrated
