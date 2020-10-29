# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE_MAGIC_CELL
# Automatically replaced inline charts by "no-op" charts
# %pylab inline
import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
import random
from copy import deepcopy
from scipy import interpolate
import numpy as np
from numpy import inf
from dtaidistance import dtw
import matplotlib.pyplot as plt
from _plotly_future_ import v4_subplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import clustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree, fcluster
from sklearn.cluster import AgglomerativeClustering
import os

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read the dataset as a Pandas dataframe in memory
# Note: here, we only read the first 100K rows. Other sampling options are available
dataset_anomaly_readings_pivoted = dataiku.Dataset("readings_long_pivoted")
df = dataset_anomaly_readings_pivoted.get_dataframe()
N_OBS = df.shape[1]-1

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Clustering

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# first, format the input data for clustering
# let's compute the DTW distances for all the pairs among our FRFs
val = np.matrix(df.to_numpy()[:, 1:], dtype=np.double)
labels = df.to_numpy()[:, 0]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# create a distance matrix
# this can be fed to sklearn function
# its lower triangle is empty
ds = dtw.distance_matrix(val)
ds

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# this 1-D distance matrix is used for Scipy
ds_compact = dtw.distance_matrix(val, compact=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# convert a distance matrix into a full-distance matrix
# full-distance matrix has its lower triangle a transposed from the upper
i_lower = np.tril_indices(val.shape[0], -1)
ds_full = ds
ds_full[i_lower] = ds_full.T[i_lower]
ds_full[ds_full == inf] = 0
ds_full

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
dist_df = pd.DataFrame(ds_full, columns=labels, index=labels)
dist_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Using Scipy with DTW

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
Z = linkage(ds_compact, method='complete', metric=dtw.distance_matrix)
Z

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# #### How to interpret Z
# 
# A  by 4 matrix Z is returned. At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster. A cluster with an index less than  corresponds to one of the  original observations. The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.
# 
# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# This Z matrix can be used on the flow in the form of a dataset, to create a chart

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
# try different criterion and compare results
clust = fcluster(Z, 2, criterion='maxclust')
clust

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# see clustered FRFs
zip_iterator = zip(labels, clust)
d = dict(zip_iterator)
d

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
frf_labeled = pd.DataFrame({'FRF':labels, "cluster":clust})
frf_labeled

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Visualize the clustering as a dendogram
fig = plt.figure(figsize=(25, 12))
CUT_THRESHOLD=8 # defines the similarity threshold and separates the color in the dendogram
# plot_dendrogram(model, truncate_mode='level', color_threshold=CUT_THRESHOLD)
dn = dendrogram(Z, orientation='top', color_threshold=CUT_THRESHOLD, labels=labels)
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel("distance ranks")
plt.xlabel("FRFs")
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
folder = dataiku.Folder("HaBimeHi") # points to the folder
folder_path = folder.get_path() # gets the path to the folder
path_fig = os.path.join(folder_path, 'dendogram')
plt.savefig(path_fig)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cuts = cut_tree(Z, n_clusters=[2, 3, 4, 5])
cuts

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
frf_labeled_dtw = pd.DataFrame(np.column_stack([labels, cuts]), columns=["label", "2_clust", "3_clust", "4_clust", "5_clust"])
frf_labeled_dtw

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
py_recipe_output = dataiku.Dataset("readings_dtw")
py_recipe_output.write_with_schema(frf_labeled_dtw)