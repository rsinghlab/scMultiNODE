# Description: Preprocessing of drosophila scRNA-seq and scATAC-seq data.
# Author: Jiaqi Zhang <jiaqi_zhang2@brown.edu>
# Reference:
#   https://www.science.org/doi/10.1126/science.abn5800
#   https://shendure-web.gs.washington.edu/content/members/DEAP_website/public/
library(Seurat)
library(stringr)
library(Signac)

# ----------------------------------


processATAC <- function(split_type) {
  all_data_obj <- readRDS("D:/Projects/MultiOmic_scDynamic/data/drosophila_embryonic/raw/subsampled_ATAC_data_common_feature.rds") # peak x cell
  # -----
  # Split data by traing and testing sets
  cell_tp <- all_data_obj@meta.data$time
  unique_tp <- c("00-02", "01-03", "02-04", "03-07", "04-08", "06-10", "08-12", "10-14", "12-16", "14-18", "16-20")
  if (split_type == "all") {
    train_tps <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    test_tps <- c()
  } else {
    stop(sprintf("Unknown split type %s!", split_type))
  }
  print(sprintf("Num tps: %d", length(unique_tp)))
  print("Train tps:")
  print(train_tps)
  print("Test tps:")
  print(test_tps)
  train_obj <- all_data_obj[, which(cell_tp %in% unique_tp[train_tps])]
  print(sprintf("Train data shape (peak x cell): %d x %d", dim(train_obj)[1], dim(train_obj)[2]))
  # -----
  # Select highly variables based on training data
  top_feature <- FindTopFeatures(object = GetAssayData(train_obj[["RNA"]], slot = "counts"))[1:2000,]
  top_feature_names <- rownames(top_feature)
  data_count_hvg <- GetAssayData(all_data_obj[["RNA"]], slot = "counts")[top_feature_names,]
  print(sprintf("HVG data shape (peak x cell): %d x %d", dim(data_count_hvg)[1], dim(data_count_hvg)[2]))
  # -----
  # Save data
  write.csv(t(as.matrix(data_count_hvg)), sprintf("D:/Projects/MultiOmic_scDynamic/data/drosophila_embryonic/processed/%s-ATAC_count_data.csv", split_type))
  write.csv(top_feature_names, sprintf("D:/Projects/MultiOmic_scDynamic/data/drosophila_embryonic/processed/%s-ATAC_var_features_list.csv", split_type))
}


processRNA <- function(split_type) {
  # Load data and subsampling
  print("Loading drosophila data...")
  all_data_obj <- readRDS("D:/Projects/MultiOmic_scDynamic/data/drosophila_embryonic/raw/subsampled_RNA_data.rds") # gene x cell
  cell_tp <- all_data_obj@meta.data$time
  unique_tp <- c(
    "hrs_00_02", "hrs_01_03", "hrs_02_04", "hrs_03_07", "hrs_04_08", "hrs_06_10",
    "hrs_08_12", "hrs_10_14", "hrs_12_16", "hrs_14_18", "hrs_16_20"
  )
  if (split_type == "all") {
    train_tps <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    test_tps <- c()
  } else {
    stop(sprintf("Unknown split type %s!", split_type))
  }
  print(sprintf("Num tps: %d", length(unique_tp)))
  print("Train tps:")
  print(train_tps)
  print("Test tps:")
  print(test_tps)
  train_obj <- all_data_obj[, which(cell_tp %in% unique_tp[train_tps])]
  print(sprintf("Train data shape (gene x cell): %d x %d", dim(train_obj)[1], dim(train_obj)[2]))
  # -----
  # Select highly variables based on training data
  train_obj <- FindVariableFeatures(NormalizeData(train_obj), selection.method = "vst", nfeatures = 2000)
  hvgs <- VariableFeatures(train_obj)
  data_count_hvg <- GetAssayData(all_data_obj[["RNA"]], slot = "counts")[hvgs,]
  print(sprintf("HVG data shape (gene x cell): %d x %d", dim(data_count_hvg)[1], dim(data_count_hvg)[2]))
  # -----
  # Save data
  write.csv(t(as.matrix(data_count_hvg)), sprintf("D:/Projects/MultiOmic_scDynamic/data/drosophila_embryonic/processed/%s-RNA_count_data.csv", split_type))
  write.csv(hvgs, sprintf("D:/Projects/MultiOmic_scDynamic/data/drosophila_embryonic/processed/%s-RNA_var_genes_list.csv", split_type))
}

# ----------------------------------

split_type <- "all"
print(split_type)
processATAC(split_type)
processRNA(split_type)


