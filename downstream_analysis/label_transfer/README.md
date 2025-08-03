# label_transfer

Cross-modal cell germ layer label transfer with multi-modal integrations. 

- [1.1_scMultiNODE_GermLabelTransfer.py](./1.1_scMultiNODE_GermLabelTransfer.py): Label transfer with scMultiNODE integration. 
- [1.2_Integration_Baseline_GermLabelTransfer.py](./1.2_Integration_Baseline_GermLabelTransfer.py): Label transfer with baseline' integration. 
- [2.1_Compare_GermLabelTransfer_MarkerGeneExpression.py](./2.1_Compare_GermLabelTransfer_MarkerGeneExpression.py): Compare marker gene expression of predicted germ layer group.
- [2.2_Compare_GermLabelTransfer_UMAP.py](./2.2_Compare_GermLabelTransfer_UMAP.py): Compute label transfer with UMAP.
- [3.0_Extract_GermLayer_GeneList.py](./3.0_Extract_GermLayer_GeneList.py): Preparation for GO enrichment analysis.
- [3.1_scMultiNODE_GermLayer_GO.py](./3.1_scMultiNODE_GermLayer_GO.py): GO enrichment analysis on scMultiNODE integration with transferred labels. We use [clusterProfiler](https://guangchuangyu.github.io/software/clusterProfiler/) following its [documentation](https://guangchuangyu.github.io/2016/11/showcategory-parameter-for-visualizing-comparecluster-output/).
- [3.2_Plot_enrichGO_res.py](./3.2_Plot_enrichGO_res.py): Visualize GO enrichment results.
