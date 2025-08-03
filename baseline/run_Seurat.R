# Description: Run Seurat for modality integration
#Author: Jiaqi Zhang <jiaqi_zhang2@brown.edu>
# Reference:
#   [1] https://satijalab.org/seurat/reference/transferdata
#   [2] https://satijalab.org/seurat/reference/findtransferanchors
#   [3] https://satijalab.org/seurat/articles/seurat5_atacseq_integration_vignette#co-embedding-scrna-seq-and-scatac-seq-datasets
library(Seurat)
library(SeuratData)
library(data.table)
library(reticulate)
metric_py <- import("metric4R")

# -------------------------

SeuratAlign <- function (rna_mat, atac_mat, k_anchor, ref_domain, k_weight=50, sd_weight=1.0){
  # rna_mat: a cell-by-feature matrix
  # atac_mat: a cell-by-feature matrix
  # -----
  # Construct seurat object
  rna_mat <- t(rna_mat)
  atac_mat <- t(atac_mat)
  rna_data <- CreateSeuratObject(counts = rna_mat, assay="RNA")
  atac_data <- CreateSeuratObject(counts = atac_mat, assay="ATAC")
  rna_data[["RNA"]]["data"] <- rna_data[["RNA"]]["counts"]
  atac_data[["ATAC"]]["data"] <- atac_data[["ATAC"]]["counts"]
  if (ref_domain == "RNA"){
    ref_data <- rna_data
    tar_data <- atac_data
    ref_dom <- "RNA"
    tar_dom <- "ATAC"
  } else {
    ref_data <- atac_data
    tar_data <- rna_data
    ref_dom <- "ATAC"
    tar_dom <- "RNA"
  }
  print("Find transfer anchors...")
  transfer.anchors <- FindTransferAnchors(
    reference=ref_data, query=tar_data, features=Features(ref_data),
    reference.assay=ref_dom, query.assay=tar_dom, reduction="cca", scale=FALSE, k.anchor=k_anchor,
  )
  ref_mat <- GetAssayData(ref_data, assay=ref_dom, slot="counts")
  # -----
  print("Transfer data...")
  imputation <- TransferData(anchorset=transfer.anchors, refdata=ref_mat, dims=NULL, weight.reduction = "cca", k.weight=k_weight, sd.weight=sd_weight)
  # -----
  if (ref_domain == "RNA"){
    rna_integrated <- rna_mat
    atac_integrated <- as.matrix(GetAssayData(imputation))
  } else {
    rna_integrated <- as.matrix(GetAssayData(imputation))
    atac_integrated <- atac_mat
  }
  rna_integrated <- t(rna_integrated)
  atac_integrated <- t(atac_integrated)
  rna_integrated[is.nan(rna_integrated)] <- 0
  atac_integrated[is.nan(atac_integrated)] <- 0
  return(list(rna_integrated, atac_integrated))
}




