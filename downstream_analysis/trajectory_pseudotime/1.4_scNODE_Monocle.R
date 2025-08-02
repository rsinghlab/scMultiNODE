# Remember to change this and following directory paths based on your working direction
setwd("D:\\Projects\\scMultiNODE\\downstream_analysis\\trajectory_pseudotime\\")

library(monocle3)
library(Matrix)
library(igraph)

# ==================================================
# Load pre-computed data and constructing object for monocle3

modality <- "RNA" # RNA, ATAC

print("Loading expression matrix...")
expression_matrix <- read.csv(sprintf("./res/data4Monocle/coassay_cortex-scNODE-%s-integrate.csv", modality), row.names=NULL, header=FALSE)
expression_matrix <- as(expression_matrix, "sparseMatrix")
expression_matrix <- t(expression_matrix)
print(dim(expression_matrix))

print("Loading umap latent...")
umap_matrix <- read.csv(sprintf("./res/data4Monocle/coassay_cortex-scNODE-%s-umap_latent_data.csv", modality), row.names=NULL, header=FALSE)
umap_matrix <- as.matrix(umap_matrix)
print(dim(umap_matrix))

print("Loading cell meta data...")
cell_metadata <- read.csv(sprintf("./res/data4Monocle/coassay_cortex-scNODE-%s-concat_meta_df.csv", modality), row.names=1)
cell_metadata <- as.data.frame(cell_metadata)
rownames(cell_metadata) <- paste0("Cell", 1:dim(cell_metadata)[1])
print(dim(cell_metadata))

print("Constructing gene meta data...")
gene_metadata <- data.frame(gene_short_name = paste0("Gene", 1:50), row.names = paste0("Gene", 1:50))
print(dim(gene_metadata))

print("Constructing CDS...")
rownames(expression_matrix) <- rownames(gene_metadata)
colnames(expression_matrix) <- rownames(cell_metadata)

cds <- new_cell_data_set(expression_matrix, cell_metadata=cell_metadata, gene_metadata=gene_metadata)
print(cds)

# ==================================================
# Set pre-computed umap
# Note: we are not computing pca/umap here, instead we replace them with our precomputed umap.

print("Pre-processing...")
cds <- preprocess_cds(cds, method="PCA")
reducedDims(cds)$PCA <- as.matrix(t(exprs(cds)))
cds <- reduce_dimension(cds, reduction_method = "UMAP", preprocess_method = "PCA")
reducedDims(cds)$UMAP <- umap_matrix

plot_cells(cds, label_groups_by_cluster=FALSE,  color_cells_by = "cell_types", cell_size=0.8, graph_label_size=5.0)
plot_cells(cds, label_groups_by_cluster=FALSE,  color_cells_by = "tps", cell_size=0.8, graph_label_size=5.0)

# ==================================================
# Estimate pseudotime

print("Find partitions...")
cds <- cluster_cells(cds)
plot_cells(cds, color_cells_by = "partition")

print("Constructing trajectory graph...")
cds <- learn_graph(cds)
plot_cells(cds,
           color_cells_by = "cell_types",
           label_groups_by_cluster=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE)

print("Compute pseudotime...")
cds <- order_cells(cds)
plot_cells(cds,
           color_cells_by = "pseudotime",
           label_cell_groups=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE,
           graph_label_size=1.5)

# ==================================================
print("Saving res...")
cell_metadata <- as.data.frame(colData(cds))
pseudotime <- pseudotime(cds)
partitions <- partitions(cds)
cell_metadata$pseudotime <- pseudotime
cell_metadata$partitions <- partitions
write.csv(cell_metadata, sprintf("./res/data4Monocle/coassay_cortex-scNODE-%s-Monocle3_res_df.csv", modality), row.names = FALSE)
