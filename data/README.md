# data

Pre-processing of six temporal multi-modal single-cell datasets. 
The raw and pre-processed data can be downloaded from [here](https://doi.org/10.6084/m9.figshare.27420657.v2) (for HC, 
HO, DR, and MN data) and [here](https://doi.org/10.6084/m9.figshare.29737034.v1) (for ZB and AM data).

The preprocessing script for each dataset are provided in the corresponding directory. If you want to re-run the preprocessing, 
remember to change file path according to your environments.


| ID | Dataset        | Species                 | # RNA Cells | # RNA Timepoints | # ATAC Cells | # ATAC Timepoints | Co-Assay Data | Source                                                  |
|----|----------------|-------------------------|-------------|------------------|--------------|-------------------|---------------|---------------------------------------------------------|
| HC | human cortex   | *Homo sapiens*            | 2277        | 10               | 2277         | 10                | Yes           | [[1]](https://doi.org/10.1126/sciadv.adg3754)             |
| HO | human organoid | *Homo sapiens*            | 10000       | 11               | 10000        | 11                | Yes           | [[2]](https://www.nature.com/articles/s41586-022-05279-8) |
| DR | drosophila     | *Drosophila melanogaster* | 2738        | 11               | 4246         | 11                | No            | [[3]](https://doi.org/10.1126/science.abn5800)            |
| MN | mouse neoctex  | *Mus musculus*            | 6098        | 3                | 1914         | 3                 | No            | [[4]](https://www.nature.com/articles/s41593-022-01123-4) |
| ZB | zebrahub   | *Danio rerio*            | 3692        | 6                | 9456         | 6                 | No            | [[5]](https://www.cell.com/cell/fulltext/S0092-8674(24)01147-4) |
| AM | amphioxus development   | *Branchiostoma lanceolatum*            | 9630        | 6                | 3538         | 6                 | No            | [[6]](https://www.cell.com/cell-reports/fulltext/S2211-1247(22)00765-3) |


### Reference

[1] Zhu, K., Bendl, J., Rahman, S., Vicari, J. M., Coleman, C., Clarence, T., ... & Roussos, P. (2023). Multi-omic profiling of the developing human cerebral cortex at the single-cell level. Science Advances, 9(41), eadg3754.

[2] Fleck, J. S., Jansen, S. M. J., Wollny, D., Zenk, F., Seimiya, M., Jain, A., ... & Treutlein, B. (2023). Inferring and perturbing cell fate regulomes in human brain organoids. Nature, 621(7978), 365-372.

[3] Calderon, D., Blecher-Gonen, R., Huang, X., Secchia, S., Kentro, J., Daza, R. M., ... & Shendure, J. (2022). The continuum of Drosophila embryonic development at single-cell resolution. Science, 377(6606), eabn5800.

[4] Yuan, W., Ma, S., Brown, J. R., Kim, K., Murek, V., Trastulla, L., ... & Arlotta, P. (2022). Temporally divergent regulatory mechanisms govern neuronal diversification and maturation in the mouse and marmoset neocortex. Nature Neuroscience, 25(8), 1049-1058.

[5] Lange, M., Granados, A., VijayKumar, S., Bragantini, J., Ancheta, S., Kim, Y. J., ... & Royer, L. A. (2024). A multimodal zebrafish developmental atlas reveals the state-transition dynamics of late-vertebrate pluripotent axial progenitors. Cell, 187(23), 6742-6759.

[6] Ma, P., Liu, X., Xu, Z., Liu, H., Ding, X., Huang, Z., ... & Chen, D. (2022). Joint profiling of gene expression and chromatin accessibility during amphioxus development at single-cell resolution. Cell Reports, 39(12).

