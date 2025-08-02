library(org.Dm.eg.db)
library(clusterProfiler)


df <- read.table("./res/GO/all-genes.csv", sep=",", header=TRUE)
gene_list <- split(df$gene, df$label)

formula_res <- compareCluster(geneCluster=gene_list, fun="enrichGO", OrgDb=org.Dm.eg.db, keyType="SYMBOL", pvalueCutoff = 0.05, pAdjustMethod = "BH", ont="BP")

dotplot(formula_res)

cc_df <- as.data.frame(formula_res)
write.csv(cc_df, "./res/GO/compareCluster_results.csv", row.names = FALSE)