import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.neighbors import KDTree
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

from flaubert.model.decision_trees import dt

features_train = None
features_names = None
clf = None
feature_labels = None
XDF = None

xDICT_ERKLAERUNG = None


def set_data(ftr, featureNames=None):
    global features_train, features_names
    features_train = ftr.copy()
    if featureNames is not None:
        features_names = featureNames


def get_durchschnittliche_diff_dist(sample_n=0.2):
    df = pd.DataFrame(features_train, columns=features_names)
    liste_a = df.sample(int(sample_n * df.shape[0])).values
    liste_b = df.sample(int(sample_n * df.shape[0])).values
    dist_liste = cdist(liste_a, liste_b, "euclidean").flatten()
    return pd.Series(dist_liste).describe().to_dict()


def get_clustering_stats(clf, features_train_clf):
    nclusters = len(list(set(clf.labels_)))
    if nclusters > 2:
        s_score = silhouette_score(features_train_clf, clf.labels_)
        distribution_cluster_labels = pd.Series(list(filter(lambda x: x != -1, clf.labels_)))
        distribution_data_stat = distribution_cluster_labels.skew()
        noise_data_stat = pd.Series(list(filter(lambda x: x == -1, clf.labels_))).shape[0] / clf.labels_.shape[0]
        return nclusters, s_score, distribution_data_stat, noise_data_stat
    else:
        return nclusters, None, None, None


def get_optim_dbscan_params_kdtree():
    global features_train
    if True:
        tree = KDTree(features_train, leaf_size=int(0.1 * features_train.shape[0]))
        xL = []
        """Info zum EPS (Epsilon) Wert: "
           epsilon ist ein Fixwert für die maximale Distanz zwischen zwei Punkte
           => Zwei Punkte werden als Nachbarn angesehen, wenn die Distanz < EPS
           optim_eps Parameter; Berechnung Stufenweise der Dist[distanzen, kNachbarn=(kStart, kEnd)]
        """
        kStartEnd = (min(1, int(0.01 * features_train.shape[0])),
                     min(int(0.2 * features_train.shape[0]), 10))
        print("(kStart, kEnd):", kStartEnd)
        for kNeighbours in range(*kStartEnd):
            xrec = [tree.query([individ], k=kNeighbours, return_distance=True) for individ in features_train]
            xrec = pd.Series([np.mean(x[0][0][1:kNeighbours]) for x in xrec])
            xL.append([kNeighbours, xrec.quantile(0.1)])
            xL.append([kNeighbours, xrec.quantile(0.9)])
        xLdf = pd.DataFrame(xL)
        xLdf.columns = ["k", "eps"]
        plt.scatter(x=xLdf["k"], y=np.log(xLdf["eps"]), s=5)
        plt.xlabel("k Nachbarn")
        plt.ylabel("np.log(eps) (durchschntl. Distanz, jeweils 10%, 90%)")
        plt.title("Durchschnittliche Distanzen (Intervall [10%, 90%]) ")
        plt.show()
    # addiere Alternativen
    for k in np.arange(2, 5, 0.2):
        xLdf_new = xLdf.copy()
        xLdf_new["eps"] /= k
        xLdf = pd.concat([xLdf, xLdf_new])
        xLdf = xLdf.drop_duplicates()\
                   .dropna()\
                   .reset_index(drop=True)
    dist_stats_dict = get_durchschnittliche_diff_dist(sample_n=0.5)
    return xLdf[xLdf["eps"] >= dist_stats_dict["mean"] / 10.0].reset_index(drop=True)


def get_optim_dbscan_params_grid(features_train_clf, xmetric="euclidean",
                                 min_val=0.01, max_val=15, minsamp=10):
    df_stats = []
    for eps_ in np.linspace(min_val, max_val, 75):
        clf = cluster.DBSCAN(eps=eps_, min_samples=minsamp, metric=xmetric, algorithm="kd_tree", leaf_size=10)
        clf.fit(features_train_clf)
        nclusters, s_score, distribution_data_stat, noise_data_stat = get_clustering_stats(clf, features_train_clf)
        if s_score is not None and s_score > 0 and noise_data_stat <= 0.2:
            record = [eps_, minsamp, nclusters, s_score, distribution_data_stat, noise_data_stat]
            print(record)
            df_stats.append(record)
    df = pd.DataFrame(df_stats, columns=["eps", "minsamp", "nclusters", "s_score",
                                           "distribution_data_stat", "noise_data_stat"])
    print(df)
    return df

def optimale_parametern_suchen(xmetric):
    if True:
        # TODO: Implementierung
        eps_k_df = get_optim_dbscan_params_kdtree()
        xL = []
        for k_, eps_ in eps_k_df[["k", "eps"]].values.tolist():
            clf = cluster.DBSCAN(eps=eps_, min_samples=k_, metric=xmetric, algorithm="kd_tree", leaf_size=10)
            clf.fit(features_train)
            if 1 < len(set(clf.labels_)) < 20:
                # ... s_score, distribution_data, noise_data ...
                xL.append([k_, eps_, len(set(clf.labels_))])
        print("[x] Ergebnis:")
        if len(xL) == 0:
            print("[x] Fehler bei der Suche!")
            return None
        xLdf = pd.DataFrame(xL)
        xLdf.columns = ["k_nachbarn", "epsilon", "clusters", "silhouette_score", "verteilung_counts", "rausch_anteil"]
        xLdf['score'] = [1 / t1 - t2 + t3 for t1, t2, t3 in
                         xLdf[["clusters", "rausch_anteil", "silhouette_score"]].values]
        print(xLdf)
        xLdf.plot.scatter(x="k_nachbarn", y="clusters", c="epsilon")
        plt.show()
    xLdf = xLdf.sort_values(by="score")
    return xLdf


def erklaerung_gruppe(xdfIn, xg):
    xdfInput = xdfIn.copy()
    xn = xdfInput[xdfInput["cdbscan"] == xg].shape[0]
    xm = xdfInput[xdfInput["cdbscan"] != xg].shape[0]
    xm = (xn * 3) if (xm > xn * 3) else xm
    dt.vis_conf_mat = False
    dt.warum(xdfInput,
             xcolumnsSet=[x for x in xdfInput.columns if x not in ["cdbscan"]], xcolTarget="cdbscan",
             xlambdaF=(lambda x: 'g0' if x != xg else 'g1'),
             useOrdered=["g0", "g1"], balancedN=(None if xm <= 3 * xn else {"g0": 3 * xn, "g1": xn}),
             test_size=0.2, max_depth=3, min_samples_split=25, min_samples_leaf=25, criterion="entropy",
             print_stats=True,
             print_dtstruktur=True,
             dt_visualisierung=True, fname_dt=f"gruppe_{xg}")
    return dt.xres_code, dt.clf


def do_model(eps, k=5, xmetric="euclidean"):
    global clf, feature_labels, features_train, features_names, XDF
    global xDICT_ERKLAERUNG

    # optim_param = optimale_parametern_suchen(xmetric=xmetric)
    # k, eps, _, _, _, _, _ = optim_param.iloc[-1].values.tolist()
    clf = cluster.DBSCAN(eps=eps, min_samples=k, metric=xmetric,
                         algorithm="kd_tree",
                         leaf_size=30)
    clf.fit(features_train)

    feature_labels = clf.labels_
    xdt = pd.DataFrame(features_train)
    if features_names is not None:
        xdt.columns = features_names
    xdt["cdbscan"] = feature_labels
    XDF = xdt.copy()

    xlistGruppen = list(set(feature_labels))
    print("\n[x] Gefundene Clusters", len(xlistGruppen) - 1, "(-1 ist Rausch)")
    xDICT_ERKLAERUNG = {}
    if (len(xlistGruppen) - 1) >= 2:
        for xg in xlistGruppen:
            print("\n====================================================================================")
            print("[x] Beispiel Cluster ID", xg, "(entspricht g1)")
            dt_code, dt_clf = erklaerung_gruppe(xdt, xg)
            xDICT_ERKLAERUNG[xg] = (dt_code, dt_clf)
        print("Cluster Größen:")
        for xg in xlistGruppen:
            xn = xdt[xdt["cdbscan"] == xg].shape[0]
            print("[x] Cluster " + str(xg) + ": " + str(xn))
    else:
        xDICT_ERKLAERUNG["-1"] = (None, None)
