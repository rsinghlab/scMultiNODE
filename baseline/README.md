# baseline

Implementation of baseline models. 

We compare scMultiNODE with six state-of-the-art unsupervised single-cell integration methods:

- **Seurat**: an R package for multi-modal integration. We use [Seurat v5](https://satijalab.org/seurat/) in our experiments. 
To install, please follow its [instructions](https://satijalab.org/seurat/articles/install_v5.html).

- **SCOTv1**: We use the SCOTv1 implementation on [https://github.com/rsinghlab/SCOT](https://github.com/rsinghlab/SCOT).

- **SCOTv2**: It is an improvement upon SCOTv1. We use the SCOTv2 implementation on [https://github.com/rsinghlab/SCOT](https://github.com/rsinghlab/SCOT).

- **UnionCom**: Use geometrical matrix matching to integrate datasets. We use the UnionCom implementation on [https://github.com/caokai1073/UnionCom](https://github.com/caokai1073/UnionCom).

- **Pamona**: We use the Pamona implementation on [https://github.com/caokai1073/Pamona](https://github.com/caokai1073/Pamona).

- **uniPort**: We use its implementation on [https://github.com/caokai1073/uniPort](https://github.com/caokai1073/uniPort).


For each baseline model (except Seurat), we downloaded its source codes from the above links and created wrapper scripts
(script filename starts with `run_`) to enable compatibility in our experiments.  

*We will shortly update the scripts for running baseline models on datasets.*