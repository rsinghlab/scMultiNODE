'''
Description:
    Plot GO enrichment results.

Author:
    Jiaqi Zhang <jiaqi_zhang2@brown.edu>
'''
import pandas as pd
import seaborn as sns
import numpy as np
from plotting import *

# -----
# Load the saved results
df = pd.read_csv("./res/GO/compareCluster_results.csv")

# -----
# Replace some GO annotations with abbreviations for better viualization
df['Description'] = df['Description'].apply(lambda x: x.replace("negative regulation", "neg. reg."))
df['Description'] = df['Description'].apply(lambda x: x.replace("positive regulation", "pos. reg."))
df['Description'] = df['Description'].apply(lambda x: x.replace(
    "negative regulation of nucleobase-containing compound metabolic process",
    "neg. reg. of nb cmpd metabolism"
))
df['Description'] = df['Description'].apply(lambda x: x.replace(
    "nucleobase-containing",
    "nb"
))
df['Description'] = df['Description'].apply(lambda x: x.replace(
    "enzyme-linked receptor protein signaling pathway",
    "enzyme-linked receptor signaling"
))
df['Description'] = df['Description'].apply(lambda x: x.replace(
    "cyclin-dependent protein kinase holoenzyme complex",
    "CDK holoenzyme complex"
))

df['GeneRatioValue'] = df['GeneRatio'].apply(lambda x: eval(x))
df['-logp.adj'] = df['p.adjust'].apply(lambda x: -np.log10(x))

# -----
# Plot top 15 GO annotations for each germ layer group
n_terms = 15
df = (
    df.sort_values("p.adjust")   # sort by significance
      .groupby("Cluster")        # group by predicted germ layer label
      .head(n_terms)
)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df,
    x='Cluster',
    y='Description',
    size='GeneRatioValue',
    hue='-logp.adj',
    palette='magma',
    edgecolor='black',
)
plt.xlabel("")
plt.ylabel("")
plt.xticks([0, 1, 2, 3], ["Ecto", "Neu. Ecto", "Meso", "Endo"])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("./res/figs/germ_GO.pdf", dpi=600)
plt.show()
plt.close()