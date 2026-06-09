import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, ward, single

from sklearn import datasets, mixture, cluster
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn import preprocessing
import seaborn as sbn

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA

from heimat.utils import daten_laden as xdl

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)


_ = xdl.lade_zufalldatenbestand(8, 8, 4, pinf=0.1, n_samples=25000, typ="clf", class_sep=3.5)
x_train, y_train, x_test, y_test, X_mins, X_max = _
ncols = x_train.shape[1]
feature_names = [f"feat_{w}" for w in range(ncols)]
dfref, dftarget = pd.DataFrame(x_train, columns=feature_names), pd.DataFrame(x_test, columns=feature_names)
feat_info = pd.DataFrame({"attribute": feature_names,
                          "information_level": ["pointlike"] * ncols,
                          "type": ["numeric"] * ncols,
                          "missing_or_unknown": [0.0] * ncols})
X = dfref
nclusters = 6

# Clustering mit KMeans
# Clustering mit GMM
# Clustering mit Agglomerative, DBSCAN
X_norm = preprocessing.MaxAbsScaler().fit_transform(X)
X_norm = preprocessing.StandardScaler().fit_transform(X_norm)
kmeans = cluster.KMeans(n_clusters=nclusters, random_state=42).fit(X_norm); cl_kmeans = kmeans.predict(X_norm)
gmm = mixture.GaussianMixture(n_components=nclusters).fit(X_norm); cl_gmm = gmm.predict(X_norm)
ac = cluster.AgglomerativeClustering(n_clusters=nclusters, linkage='ward'); cl_ac = ac.fit_predict(X_norm)
dbscan = cluster.DBSCAN(eps=0.5, min_samples=4); cl_dbscan = dbscan.fit_predict(X_norm)

randids = pd.Series(range(X_norm.shape[0])).sample(1000)
linkage_matrix = ward(X_norm[randids,:]); dendrogram(linkage_matrix); plt.tight_layout(); plt.show()
sbn.clustermap(X_norm[randids, :], figsize=(16, 10), method="ward", cmap='turbo'); plt.tight_layout(); plt.show()

# hue="species", palette=sbn.color_palette("turbo", nclusters)
visdf = dfref[pd.Series(feature_names).sample(5).tolist()]
g = sbn.PairGrid(visdf, vars=visdf.columns)
g.map_diag(sbn.histplot)
g.map_offdiag(sbn.scatterplot)
plt.tight_layout()
plt.show()

if True:
    embed_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50).fit_transform(X_norm)
    embed_pca = PCA(n_components=2).fit_transform(X_norm)
    embed_fica = FastICA(n_components=2).fit_transform(X_norm)

fig, ax = plt.subplots(3, 1, figsize=(14, 8), dpi=125)
ax[0].scatter(x=embed_tsne[:,0], y=embed_tsne[:,1], s=4)
ax[1].scatter(x=embed_pca[:,0], y=embed_pca[:,1], s=4)
ax[2].scatter(x=embed_fica[:,0], y=embed_fica[:,1], s=4)
ax[0].set_ylabel("TSNE")
ax[1].set_ylabel("PCA")
ax[2].set_ylabel("Fast ICA")
plt.tight_layout()
plt.show()
