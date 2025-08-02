'''
Description:
    Align RNA and ATAC assays with Pamona.

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


if __name__ == '__main__':
    data1 = np.loadtxt("./Pamona/scGEM/methylation_partial.txt")
    data2 = np.loadtxt("./Pamona/scGEM/expression_partial.txt")
    type1 = np.loadtxt("./Pamona/scGEM/methylation_type_partial.txt")
    type2 = np.loadtxt("./Pamona/scGEM/expression_type_partial.txt")
    type1 = type1.astype(np.int)
    type2 = type2.astype(np.int)
    # Pa = Pamona.Pamona(n_shared=[138], Lambda=10, output_dim=5)  # shared cell number 138 is estimated by SPL
    # integrated_data, T = Pa.run_Pamona([data1, data2])
    rna_integrated, atac_integrated, T, Pa = PamonaAlign(
        data1, data2, n_shared=[138], epsilon=1e-3, n_neighbors=20, Lambda=10, output_dim=5, verbose=True, return_aux=True
    )
    integrated_data = [rna_integrated, atac_integrated]
    Pa.test_LabelTA(integrated_data[0], integrated_data[-1], type1, type2)
    Pa.alignment_score(integrated_data[0], integrated_data[-1][0:142], data2_specific=integrated_data[-1][142:177])
    Pa.Visualize([data1, data2], integrated_data, mode='UMAP')  # without datatype
    Pa.Visualize([data1, data2], integrated_data, [type1, type2], mode='UMAP')  # with datatype
