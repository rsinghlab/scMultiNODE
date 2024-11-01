# Description: Extract the human organoid data (Fleck2022).
# Author: Jiaqi Zhang <jiaqi_zhang2@brown.edu>
# Reference:
#   https://www.nature.com/articles/s41586-022-05279-8
#   https://zenodo.org/records/5242913
library(Seurat)
library(Signac)
library(Matrix)
setwd("/oscar/data/rsingh47/jzhan322/MultiOmic_scDynamic/data/human_organoid_Fleck2022/")

# -----
# CROPseq_obj <- readRDS("./raw/CROPseq_srt.rds") 
multiomic_obj <- readRDS("./raw/RNA_ATAC_metacells_srt.rds")

# -----
# Extract RNA count data
rna_data_count <- GetAssayData(multiomic_obj[["RNA"]], slot = "counts")
rna_cell_name <- rna_data_count@Dimnames[[2]]
rna_gene_name <- rna_data_count@Dimnames[[1]]

# -----
# Extract ATAC gene activity
atac_data_activity <- GetAssayData(multiomic_obj[["gene_activity"]])
atac_cell_name <- atac_data_activity@Dimnames[[2]]
atac_gene_name <- atac_data_activity@Dimnames[[1]]

# -----
# Extract meta data
meta_df <- multiomic_obj@meta.data

# # -----
# # Save files
# writeMM(rna_data_count, "./raw/rna_data_count.mtx")
# write.csv(rna_cell_name, "./raw/rna_cell_name")
# write.csv(rna_gene_name, "./raw/rna_gene_name")
# 
# writeMM(atac_data_activity, "./raw/atac_data_activity")
# write.csv(atac_cell_name, "./raw/atac_cell_name")
# write.csv(atac_gene_name, "./raw/atac_gene_name")
# 
# write.csv(meta_df, "./raw/meta_data.csv")


# ==========================================

sample_idx <- sample(1:34088, 10000)

sample_rna_data_count <- rna_data_count[,sample_idx]
sample_rna_cell_name <- sample_rna_data_count@Dimnames[[2]]
sample_rna_gene_name <- sample_rna_data_count@Dimnames[[1]]

sample_atac_data_activity <- atac_data_activity[,sample_idx]
sample_atac_cell_name <- sample_atac_data_activity@Dimnames[[2]]
sample_atac_gene_name <- sample_atac_data_activity@Dimnames[[1]]

sample_meta_df <- meta_df[sample_idx,]

writeMM(sample_rna_data_count, "./raw/sample_rna_data_count.mtx")
write.csv(sample_rna_cell_name, "./raw/sample_rna_cell_name")
write.csv(sample_rna_gene_name, "./raw/sample_rna_gene_name")
writeMM(sample_atac_data_activity, "./raw/sample_atac_data_activity")
write.csv(sample_atac_cell_name, "./raw/sample_atac_cell_name")
write.csv(sample_atac_gene_name, "./raw/sample_atac_gene_name")
write.csv(sample_meta_df, "./raw/sample_meta_data.csv")
write.csv(sample_idx, "./raw/sample_idx")