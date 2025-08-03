# trajectory_pseudotime

Cell trajectory pseudotime estimation from each model's integration/latent variables..

- [0.1_PrepareData_scMultiNODE.py](./0.1_PrepareData_scMultiNODE.py), [0.2_PrepareData_Integration_Baseline.py](./0.2_PrepareData_Integration_Baseline.py), 
[0.3_PrepareData_Static_AE.py](./0.3_PrepareData_Static_AE.py), and [0.4_PrepareData_scNODE.py](./0.4_PrepareData_scNODE.py): 
Preparation for peudotime estimation.

- [1.1_scMultiNODE_Monocle.py](./1.1_scMultiNODE_Monocle.py), [1.2_Integration_Baseline_Monocle.py](./1.2_Integration_Baseline_Monocle.py), 
[1.3_Static_AE_Monocle.py](./1.3_Static_AE_Monocle.py), and [1.4_scNODE_Monocle.py](./1.4_scNODE_Monocle.py): 
Pseudotime estimation with [Monocle3](https://cole-trapnell-lab.github.io/monocle3/). Need R to run these scripts.

- [2.1_Compare_Monocle_Pseudotime.py](./2.1_Compare_Monocle_Pseudotime.py): Compare Monocle3 pseudotime estimation across different methods.

- [3.1_All_PAGA.py](./3.1_All_PAGA.py): Pseudotime estimation with [PAGA](https://github.com/theislab/paga). 

- [3.2_Compare_PAGA_Pseudotime.py](./3.2_Compare_PAGA_Pseudotime.py): Compare PAGA pseudotime estimation across different methods.

- [4.1_Plot_Monocle_Pseudotime_Violin_for_All.py](./4.1_Plot_Monocle_Pseudotime_Violin_for_All.py): Visualize Monocle3 pseudotime estimation with violin plots.
- [4.2_Plot_PAGA_Pseudotime_Violin_for_All.py](./4.2_Plot_PAGA_Pseudotime_Violin_for_All.py): Visualize PAGA pseudotime estimation with violin plots.

