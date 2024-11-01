# downstream analysis

This directory contains experiments for downstream analysis tasks (Sec. 4.2 in the paper).
You can download the pre-trained model and pre-computed cell path from [here (downstream_analysis.zip)](https://doi.org/10.6084/m9.figshare.27418872.v2).


- [0_Model_Training_on_HC.py](./0_Model_Training_on_HC.py): Train the scMultiNODE and computes joint latent space on HC dataset. 
- [1_Construct_Cell_Path.py](./1_Construct_Cell_Path.py): Construct cell path and find DE genes of the path.
- [2_Compare_Genes](./2_Compare_Genes.py): Compute RNA-ATAC-derived cell type marker genes with path DE genes. 
