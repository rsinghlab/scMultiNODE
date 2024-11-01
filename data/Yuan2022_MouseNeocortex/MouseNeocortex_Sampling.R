library(dplyr)
library(Matrix)
setwd("D:\\Projects\\MultiOmic_scDynamic\\data\\Yuan2022_MouseNeocortex")


# -----
print("-----")
print("Loading RNA...")
rna_cnt <- readRDS("./raw/GSE204759_mouse_scRNA_raw_counts.rds/GSE204759_mouse_scRNA_raw_counts.rds")
rna_cnt <- t(rna_cnt)
rna_meta <- read.csv(
  "./raw/GSE204759_mouse_scRNA_metadata.txt/GSE204759_mouse_scRNA_metadata.txt", 
  sep="\t", header=TRUE
)
rownames(rna_meta) <- rna_meta$cellId
print(dim(rna_cnt))
print(dim(rna_meta))

rand_idx <- sample(1:dim(rna_cnt)[1], as.integer(0.1*dim(rna_cnt)[1]), replace=FALSE)
rna_cnt <- rna_cnt[rand_idx,]
rna_meta <- rna_meta[rand_idx,]

print(dim(rna_cnt))
print(dim(rna_meta))
rna_meta %>% group_by(timePoint) %>%tally()

# -----

print("-----")
print("Loading ATAC...")
atac_cnt <- readRDS("./raw/GSE204761_mouse_scATAC_raw_counts.rds/GSE204761_mouse_scATAC_raw_counts.rds")
atac_cnt <- t(atac_cnt)
atac_meta <- read.csv(
  "./raw/GSE204761_mouse_scATAC_metadata.txt/GSE204761_mouse_scATAC_metadata.txt",
  sep="\t", header=TRUE
)
rownames(atac_meta) <- atac_meta$cellId
print(dim(atac_cnt))
print(dim(atac_meta))

rand_idx <- sample(1:dim(atac_cnt)[1], as.integer(0.1*dim(atac_cnt)[1]), replace=FALSE)
atac_cnt <- atac_cnt[rand_idx,]
atac_meta <- atac_meta[rand_idx,]

print(dim(atac_cnt))
print(dim(atac_meta))
atac_meta %>% group_by(timePoint) %>%tally()

# -----

writeMM(rna_cnt, "./raw/sample_rna_cnt.mtx")
writeMM(atac_cnt, "./raw/sample_atac_cnt.mtx")

write.csv(rna_meta, "./raw/sample_rna_meta.csv")
write.csv(atac_meta, "./raw/sample_atac_meta.csv")

write.csv(as.matrix(rna_cnt), "./raw/sample_rna_cnt.csv")
write.csv(as.matrix(atac_cnt), "./raw/sample_atac_cnt.csv")
