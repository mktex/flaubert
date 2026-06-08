import traceback

import numpy as np
import pandas as pd
import json
import sklearn
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn import preprocessing
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn import cluster, datasets

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn import metrics

from flaubert.eda import dfgestalt, dbeschreiben
from flaubert.ersatz import dfersatz
from flaubert.utils import utilpy, daten_laden as xdl
from flaubert.vis import xdiagramme
from flaubert.statistik import stat_op

safe_turn2num = dfersatz.safe_turn2num
from_nans_to_nones = dfgestalt.from_nans_to_nones
simple_dummy_encoding = dfgestalt.simple_dummy_encoding
sort_features = dfgestalt.sort_features
auflistung_werte_in_features = dbeschreiben.auflistung_werte_in_features
get_nulls_rows_stat = dbeschreiben.get_nulls_rows_stat
get_nulls_df_stat = dbeschreiben.get_nulls_df_stat
kateg_werte_liste = dbeschreiben.kateg_werte_liste
frequenz_werte = dbeschreiben.frequenz_werte
xdfmap = utilpy.xdfmap
show_hypothesis = xdiagramme.show_hypothesis
get_diff_mean_by_bootstrap = stat_op.get_diff_mean_by_bootstrap


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)


class GruppenVergleicher:

    feat_info = pd.DataFrame({"attribute": [], "information_level": [], "type": [], "missing_or_unknown": []})
    map_none_replacement = {'feature1': [-1.0, 0.0], 'feature2': [-1.0, 0.0, 9.0]}
    drop_records_by_features_with_null = {
        "Grund für den Rauswurf der Featureliste": ['feature1', 'feature2', 'feature3']
    }
    replacement_vals = {
        "Grund fürs Ersatz mit einem Wert": {"feature4": 0.0, "feature5": -1.0},
        "Einen anderen Grund für den Ersatz": {"feature6": 999}
    }
    replacement_by_rfc = [
        ('alter', lambda r: str(safe_turn2num(r['alter'])), lambda r: int(r['alter']))
    ]

    nsamples = None
    df_referenz, df_target = None, None
    dropped_features = []

    threshold_nullen_in_record = 2
    threshold_percent_nullen_drop = 0.4

    do_plots = True
    do_cluster_sim = True
    use_max_sample = 25000

    dfref_reduced, dftarget_reduced = None, None
    ncomponents = 15
    nclusters = 4
    diskriminanten = []

    def __init__(self, _df_referenz, _df_target, _feat_info,
                 _nsamples=None, _map_none_replacement=None):

        self.df_referenz, self.df_target = _df_referenz, _df_target
        self.map_none_replacement = _map_none_replacement
        self.feat_info = _feat_info

        self.nsamples = _nsamples
        self.setup_environment()

    def take_a_sample(self, df, n=None):
        n = self.use_max_sample if n is None else n
        if df.shape[0] > n:
            return df.sample(n)
        else:
            return df

    def setup_environment(self):
        if self.nsamples is not None:
            self.df_referenz = self.take_a_sample(self.df_referenz, self.nsamples).reset_index(drop=True)
            self.df_target = self.take_a_sample(self.df_target, self.nsamples).reset_index(drop=True)

    def setup_null_values(self, _df):
        for xkey in self.map_none_replacement.keys():
            _df[xkey] = list(map(lambda x: None if safe_turn2num(x) in self.map_none_replacement[xkey] else x,
                                 _df[xkey].values))
        return _df

    def drop_null_features(self, _df, dropping_features):
        _df = _df[list(filter(lambda x: x not in dropping_features, _df.columns))]
        self.update_dropped_features(dropping_features)
        self.feat_info = self.feat_info[list(map(lambda x: x not in dropping_features,
                                                 self.feat_info["attribute"].values))]
        return _df

    def split_ids_by_degree_of_nulls(self, _df_null_stats):
        _low_nulls = _df_null_stats[_df_null_stats['nullen'] <= self.threshold_nullen_in_record]
        _ids_low_nulls = _low_nulls['record_id']
        _high_nulls = list(set(_df_null_stats['record_id'].unique()) - set(_ids_low_nulls))
        return _ids_low_nulls, _high_nulls

    def get_categ_vars(self, _df):
        categs = self.feat_info[self.feat_info['type'] == 'categorical']
        bin_features = []
        for feature in categs['attribute'].values:
            if len(_df[feature].unique()) == 2: bin_features.append(feature)
        return categs

    def transform_cat_features(self, _df, categs, cat_dict=None):
        _df_catenc = _df.copy()
        cat_dict = {} if cat_dict is None else cat_dict
        for cat_feature in categs:
            # TODO: schreibe das erneut
            _df_catenc[cat_feature] = [str(w).replace('.0', '')
                                       for w in _df_catenc[cat_feature].tolist()]
            if cat_feature not in cat_dict.keys():
                cat_features_sorted = [w for w in _df_catenc[cat_feature].unique().tolist()]
                cat_features_sorted.sort()
                cat_features_sorted = ['None'] + list(filter(lambda x: x != 'None', cat_features_sorted))
            else:
                cat_features_sorted = cat_dict[cat_feature]
            _df_catenc = simple_dummy_encoding(_df_catenc, cat_feature, typ_encoding="01",
                                               kateg_werte_sortiert=cat_features_sorted)
            cat_dict[cat_feature] = cat_features_sorted
        return _df_catenc, cat_dict

    def get_features_with_low_representation(self):
        xset = []
        for dfname, _df in [('referenz-dataframe', self.df_referenz),
                            ('target-dataframe', self.df_target)]:
            for feature in _df.columns:
                if len(set(_df[feature])) == 1 or _df[feature].mean() < 0.01:
                    xset.append(feature)
        return list(set(xset))

    def update_dropped_features(self, features):
        self.dropped_features.extend(features)
        self.dropped_features = list(set(self.dropped_features))
        self.dropped_features.sort()

    def clean_data(self, df_input, categs=None, cat_dict=None, pj_cat_dict=None):

        df = df_input.copy()
        df = self.setup_null_values(df)
        df = from_nans_to_nones(df)

        null_stats = get_nulls_df_stat(df)
        null_stats = null_stats[null_stats['percentage'] > self.threshold_percent_nullen_drop]
        if null_stats.shape[0] > 0: df = self.drop_null_features(df, null_stats['feature'].values.tolist())

        df_nulls_records = get_nulls_rows_stat(df)
        df_ids_low_nulls, df_ids_high_nulls = self.split_ids_by_degree_of_nulls(df_nulls_records)
        xdf = df.iloc[df_ids_low_nulls].reset_index(drop=True)

        categs = self.get_categ_vars(xdf) if categs is None else categs
        xdf, cat_dict = self.transform_cat_features(xdf, categs['attribute'].values.tolist(), cat_dict)
        xdf = xdf[list(filter(lambda x: x not in categs['attribute'].values, xdf.columns))]

        return xdf, categs, cat_dict, pj_cat_dict

    def clean_combined(self):
        drfeats = self.get_features_with_low_representation()
        if len(drfeats) == 0: return
        self.df_referenz = self.df_referenz[list(filter(lambda x: x not in drfeats, self.df_referenz.columns))]
        self.df_target = self.df_target[list(filter(lambda x: x not in drfeats, self.df_target.columns))]
        self.update_dropped_features(drfeats)

    def handle_none_drop_or_replace(self, _dfinput):
        _df = _dfinput.copy()
        _df = _df[list(filter(lambda x: x not in self.dropped_features, _df.columns))]
        for reason, feature_list in list(self.drop_records_by_features_with_null.items()):
            for feature in feature_list: _df = _df[~pd.isnull(_df[feature])].reset_index(drop=True)
        for reason, feature_group in list(self.replacement_vals.items()):
            _df = _df.fillna(value=feature_group)
        return _df.reset_index(drop=True)

    def inputation_by_model(self, _dfinput, input_model, features, repl_feature, rfunk_inv):
        dfnonnulls = _dfinput[~pd.isnull(_dfinput[repl_feature])].reset_index(drop=True)
        dfnulls = _dfinput[pd.isnull(_dfinput[repl_feature])].reset_index(drop=True)
        if dfnulls.shape[0] > 0:
            dfnulls[repl_feature] = input_model.predict(dfnulls[features])
            dfnulls[repl_feature] = xdfmap(dfnulls, rfunk_inv, [repl_feature])
        _dfinput = pd.concat([dfnonnulls, dfnulls]).reset_index(drop=True)
        return _dfinput

    def null_handling_by_rfc(self, target_feature, rfunk, _dftrain, n_estimators=250, show_info=True):

        dftrain, dftest = train_test_split(_dftrain, test_size=0.2, random_state=42)
        null_stats_dfref = get_nulls_df_stat(dftrain)
        null_stats_dftarget = get_nulls_df_stat(dftest)

        training_features = null_stats_dfref[null_stats_dfref["percentage"] == 0]['feature'].tolist()
        training_features = list(filter(lambda x: x != target_feature, training_features))
        workon_df = dftrain[training_features + [target_feature]].dropna().copy()
        workon_df[target_feature] = xdfmap(workon_df, rfunk, [target_feature])
        inputation_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        try:
            inputation_clf.fit(workon_df[training_features], workon_df[target_feature])
        except:
            auflistung_werte_in_features(workon_df[training_features + [target_feature]], upto=100)
            traceback.print_exc()
            print("[x] Checkpoint FEHLER in null_handling_by_rfc()..")

        teston_df = self.df_target[training_features + [target_feature]].dropna().copy()
        teston_df[target_feature] = xdfmap(teston_df, rfunk, [target_feature])
        y_true = teston_df[target_feature]
        y_pred = inputation_clf.predict(teston_df[training_features])
        accuracy = metrics.accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=inputation_clf.classes_)
        if show_info:
            print(f"[x] Inputation None's mit RFC Model für: {target_feature}")
            xplain_most = pd.DataFrame(list(zip(training_features, inputation_clf.feature_importances_)),
                                       columns=['feature', 'score']).sort_values(by="score", ascending=False)
            print(xplain_most.head(5))
            print("\n[x] Labels:", inputation_clf.classes_)
            print(classification_report(y_pred, y_true, zero_division=0.0))
            print("\nConfusion Matrix:")
            print(cm)
            print("\nAccuracy:", accuracy)

        return inputation_clf, training_features

    def show_features_distributions(self):
        sample_features = pd.Series(self.df_referenz.columns.tolist()).sample(10)
        for feature in sample_features:
            xdiff = get_diff_mean_by_bootstrap(self.df_referenz, self.df_target, feature, nsample_pr=0.2, nzyklen=100)
            xlow, xhigh = pd.Series(xdiff).quantile(0.025), pd.Series(xdiff).quantile(1 - 0.025)
            null_hypothesis = np.round(xlow, 3) > 0.025 or np.round(xhigh, 3) < -0.025
            if not null_hypothesis: continue
            print(f"[x] Diskriminant: ", feature)
            self.diskriminanten.append(feature)
            fig = plt.figure(figsize=(10, 5), dpi=100)
            self.df_referenz[feature].astype(np.float32).hist(bins=45, density=True, color='black', label="Referenz")
            sbn.kdeplot(data=self.df_referenz[[feature]].astype(np.float32), x=feature, color='black')
            self.df_target[feature].astype(np.float32).hist(bins=45, density=True,
                                                            color='cyan', alpha=0.5, label="Target")
            sbn.kdeplot(data=self.df_target[[feature]].astype(np.float32), x=feature, color='cyan')
            plt.title(f"Comparing histograms and distributions: {feature}")
            plt.tight_layout()
            plt.legend()
            plt.show()
            print()

    def remove_outliers_by_isolation_forest(self, _df):
        iso_forest = IsolationForest(contamination='auto')
        iso_forest.fit(_df)
        _df['outlier'] = iso_forest.predict(_df)
        _df = _df[_df['outlier'] >= 0]
        _df = _df[list(filter(lambda x: x != 'outlier', _df.columns))]
        return _df, iso_forest

    def info_nullen(self, df):
        features_mit_nullen = get_nulls_df_stat(df, False)
        print(features_mit_nullen)
        if features_mit_nullen.shape[0]: auflistung_werte_in_features(df[features_mit_nullen['feature'].values])

    def feature_engineering(self, _dfinput, _clf_based_inputations=None,
                            _xscaler=None, _pca=None, _cluster_model=None):

        fehler_output = [None] * 7
        _dfinput = sort_features(_dfinput)
        _dfinput = self.handle_none_drop_or_replace(_dfinput)
        if len(self.replacement_by_rfc) > 0 and _clf_based_inputations is None:
            _clf_based_inputations = {}
            for repl_feature, rfunk, rfunk_inv in self.replacement_by_rfc:
                _clf_based_inputations[repl_feature] = self.null_handling_by_rfc(repl_feature, rfunk, _dfinput)
        for repl_feature, rfunk, rfunk_inv in self.replacement_by_rfc:
            input_model, features = _clf_based_inputations[repl_feature]
            _dfinput = self.inputation_by_model(_dfinput, input_model, features, repl_feature, rfunk_inv)
        try:
            assert _dfinput.shape[0] == _dfinput.dropna().shape[0]
        except:
            self.info_nullen(_dfinput)
            print('[x] Checkpoint FEHLER: Nulls weiterhin präsent!')
            return fehler_output

        _featureset = _dfinput.columns
        _dfinput, _iso_forest_a = self.remove_outliers_by_isolation_forest(_dfinput)

        if _xscaler is None: _xscaler = preprocessing.RobustScaler().fit(_dfinput)
        X = _xscaler.transform(_dfinput)

        if _pca is None:
            self.run_decomposition_simulation(X, _featureset)
            print("[x] CHECKPOINT: set self.ncomponents..")
            _pca = PCA(n_components=self.ncomponents)
            _pca = _pca.fit(X)
        X_reduced = _pca.transform(X)
        if self.do_plots:
            pca_stats_df = self.pca_results(_featureset, _pca)
            print(pca_stats_df)
            self.scree_plot(_pca)
            _ = [self.show_weights_component(pca_stats_df, k, 5) for k in range(5)]

        x_reduced = pd.DataFrame(X_reduced, columns=[f"PC{w}" for w in range(self.ncomponents)])
        x_reduced.index = _dfinput.index
        x_reduced, _iso_forest_b = self.remove_outliers_by_isolation_forest(x_reduced)
        _dfinput = _dfinput.loc[x_reduced.index]
        if self.do_plots: self.show_pairgrid_firstk_pcs(x_reduced, color_by_cluster=False)

        x_reduced = x_reduced[list(filter(lambda x: x != 'cluster', x_reduced.columns))]
        if _cluster_model is None:
            if self.do_cluster_sim:
                self.run_cluster_simulation(x_reduced)
            print("[x] CHECKPOINT: set self.nclusters..")
            _cluster_model = cluster.KMeans(n_clusters=self.nclusters, random_state=42, init="k-means++", n_init=25)
            _cluster_model = _cluster_model.fit(x_reduced.values)
        x_reduced['cluster'] = _cluster_model.predict(x_reduced.values)
        _dfinput['cluster'] = x_reduced['cluster']
        if self.do_plots:
            self.show_cluster_first_components(x_reduced, _cluster_model)
            self.show_pairgrid_firstk_pcs(x_reduced, color_by_cluster=True)

        cluster_names = list(x_reduced['cluster'].unique())
        cluster_names.sort()
        for xcluster in cluster_names:
            self.xplain_clusterk(_dfinput, clk=xcluster)
            print("=" * 85)

        return x_reduced, _clf_based_inputations, _xscaler, _pca, _cluster_model, _featureset, _dfinput

    def pca_results(self, featureset, pca, uptok=5):
        dimensions = ['Dimension {}'.format(i) for i in range(1, uptok + 1)]
        components = pd.DataFrame(np.round(pca.components_[:uptok], 4), columns=featureset)
        components.index = dimensions
        if len(featureset) > 6:
            keep_features = pd.DataFrame({
                "avg_pc": components.values.mean(axis=0),
                "feature": featureset
            }).sort_values(by="avg_pc", ascending=False)
            components = components[
                keep_features.head(3)["feature"].tolist() +
                keep_features.tail(3)["feature"].tolist()
                ]
        ratios = pca.explained_variance_ratio_.flatten()[:uptok]
        variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance'])
        variance_ratios.index = dimensions
        if self.do_plots:
            fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
            components.plot(ax=ax, kind='bar')
            ax.set_ylabel("Feature Weights")
            ax.set_xticklabels(dimensions, rotation=0)
            for i, ev in enumerate(ratios):
                ax.text(i - 0.40, ax.get_ylim()[1] + 0.05,
                        f"Explained Variance\n          {ev:.4f}",
                        fontsize=10)
            plt.tight_layout()
        return pd.concat([variance_ratios, components], axis=1)

    def scree_plot(self, pca):
        num_components = len(pca.explained_variance_ratio_)
        ind = np.arange(num_components)
        vals = pca.explained_variance_ratio_
        plt.figure(figsize=(15, 5), dpi=150)
        ax = plt.subplot(111)
        cumvals = np.cumsum(vals)
        ax.bar(ind, vals)
        ax.plot(ind, cumvals)
        for i in range(num_components):
            ax.annotate(r"%s%%" % ((str(vals[i] * 100)[:4])), (ind[i] + 0.2, vals[i]),
                        va="bottom", ha="center", fontsize=12)
        ax.xaxis.set_tick_params(width=0)
        ax.yaxis.set_tick_params(width=2, length=12)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Variance-Explained Ratio")
        plt.tight_layout()
        plt.title('Explained Variance Per Principal Component')
        plt.show()

    def show_weights_component(self, pca_stats_df, nk, topk):
        r = pca_stats_df.iloc[nk]
        xcomponent = pd.DataFrame({
            "feature": r.index.values,
            "weight": r.values
        })
        xcomponent = xcomponent.sort_values(by="weight", ascending=False)
        print(xcomponent.head(topk))
        print("...")
        print(xcomponent.tail(topk))
        print()

    def show_pairgrid_firstk_pcs(self, x_reduced, color_by_cluster=False):
        if not color_by_cluster:
            print('[x] First five principal components ')
            g = sbn.PairGrid(self.take_a_sample(x_reduced), vars=x_reduced.columns[:5])
        else:
            print('[x] First five principal components colored by cluster..')
            g = sbn.PairGrid(self.take_a_sample(x_reduced), vars=x_reduced.columns[:5], hue='cluster')
            g.add_legend()
        g.map_diag(sbn.histplot)
        g.map_offdiag(sbn.scatterplot)
        plt.tight_layout()
        plt.show()

    def show_cluster_first_components(self, _dfinput, kmeans_clusterer):

        _df = _dfinput[list(filter(lambda x: x != 'cluster', _dfinput.columns))].copy()
        _df = _df.reset_index(drop=True)
        X = _df.values
        sampled_points = _df.index.values
        if X.shape[0] > 10000: sampled_points = _df.sample(10000).index.values

        centers = kmeans_clusterer.cluster_centers_
        center_names = [f"cl_{w}" for w in range(len(centers))]
        cdf = pd.DataFrame(centers[:, :2], columns=['xdim', 'ydim'])
        cdf['cluster_named'] = center_names

        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(16, 8)
        clabels = kmeans_clusterer.predict(X[sampled_points, :])
        nclusters = len(set(clabels))
        colors = plt.cm.nipy_spectral(clabels.astype(float) / nclusters)
        ax.scatter(X[sampled_points, 0], X[sampled_points, 1],
                   marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
        for i, c in enumerate(centers):
            cdf_record = cdf[list(map(lambda x: x[0] == c[0] and x[1] == c[1],
                                      cdf[['xdim', 'ydim']].values))]
            cluster_name = cdf_record['cluster_named'].iloc[0]
            ax.scatter(c[0], c[1], marker='o', alpha=1, s=150, edgecolors="white", color='black')
            ax.text(c[0] + 0.4, c[1] + 0.4, f"{cluster_name} ({c[0]:.2f}, {c[1]:.2f})", horizontalalignment="center",
                    color="black",
                    fontsize=14, weight='bold', backgroundcolor='white')
        ax.set_title("K-Means clustered data first 2 components")
        plt.show()

    def run_decomposition_simulation(self, X, featureset):
        _pca = PCA(n_components=self.ncomponents)
        _pca = _pca.fit(X)
        if self.do_plots:
            pca_stats_df = self.pca_results(featureset, _pca)
            print(pca_stats_df)
            self.scree_plot(_pca)
            _ = [self.show_weights_component(pca_stats_df, k, 5) for k in range(5)]

    def run_cluster_simulation(self, _df):

        X = self.take_a_sample(_df)
        range_k_clusters, rand_state = list(range(2, 15)), 42
        kmeans_cluster_labels = []
        kmeans_clusterer_list = []

        scores = []
        for k_clusters in range_k_clusters:
            kmeans_clusterer = cluster.KMeans(n_clusters=k_clusters,
                                              random_state=rand_state, init="k-means++", n_init=25)
            kmeans_model = kmeans_clusterer.fit(X)
            cluster_labels = kmeans_model.predict(X)
            kmeans_cluster_labels.append(cluster_labels)
            kmeans_clusterer_list.append(kmeans_clusterer.cluster_centers_)
            score1 = silhouette_score(X, cluster_labels)
            score2 = davies_bouldin_score(X, cluster_labels)
            inertia = kmeans_clusterer.inertia_
            scores.append([score1, score2, np.log(inertia)])
            print(f"[x] KMeans(n_clusters={k_clusters}) done with scores: {score1:.3f}, " + \
                  f"{score2:.3f}, {np.log(inertia):.3f}")

        if self.do_plots:
            fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
            fig.set_size_inches(12, 5)
            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.plot(range_k_clusters, [w[0] for w in scores])
            ax2.plot(range_k_clusters, [w[1] for w in scores])
            ax3.plot(range_k_clusters, [w[2] for w in scores])
            ax1.set_title('silhouette')
            ax2.set_title('davis-bouldin')
            ax3.set_title('inertia')
            plt.tight_layout()
            plt.show()
        return scores, range_k_clusters, kmeans_clusterer_list, kmeans_cluster_labels

    def xplain_clusterk(self, _xplain_on_df, clk=3):
        xplain_on_df = _xplain_on_df.copy().reset_index(drop=True)
        X = xplain_on_df[list(filter(lambda x: x != 'cluster', xplain_on_df.columns))]
        y = [1 if w == clk else 0 for w in xplain_on_df['cluster'].values]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        clf = DecisionTreeClassifier(max_leaf_nodes=4,
                                     min_samples_split=1000, min_samples_leaf=1000,
                                     random_state=42)
        clf.fit(X_train, y_train)
        print(f"\n[x] Xplainer Cluster {clk}..")
        targets, predicted, class_labels = y, clf.predict(X), [0, 1]
        model_cm = confusion_matrix(y_true=targets, y_pred=predicted, labels=class_labels)
        print(classification_report(predicted, targets, zero_division=0.0))
        plt.figure(figsize=(12, 6), dpi=100)
        tree.plot_tree(clf, proportion=True, feature_names=xplain_on_df.columns[:-1])
        plt.tight_layout()
        plt.title(f"Cluster {clk}")
        plt.show()

    def run(self):

        self.df_referenz, categs, cat_dict, pj_cat_dict = self.clean_data(self.df_referenz)
        output = self.clean_data(self.df_target, categs, cat_dict, pj_cat_dict)
        self.df_target = output[0]
        missing_cat_features = set(self.df_referenz.columns) - set(self.df_target.columns)
        assert len(missing_cat_features) == 0
        self.clean_combined()

        if self.do_plots:
            self.show_features_distributions()

        _ = self.feature_engineering(self.df_referenz)
        if _[0] is None: return
        self.dfref_reduced, clf_based_inputations, xscaler, pca, cluster_model, featureset, dfxplain = _

        output = self.feature_engineering(self.df_target, _clf_based_inputations=clf_based_inputations,
                                          _xscaler=xscaler, _pca=pca, _cluster_model=cluster_model)
        self.dftarget_reduced = output[0]

        print("\n[x] Anteile in Referenz:")
        print(frequenz_werte(self.dfref_reduced, xcol="cluster", prozente=True))
        visdf = self.dfref_reduced[list(filter(lambda x: x != 'cluster', self.dftarget_reduced.columns))]
        self.show_cluster_first_components(visdf, cluster_model)
        self.show_pairgrid_firstk_pcs(self.dfref_reduced, color_by_cluster=True)

        print("\n[x] Anteile in Target:")
        print(frequenz_werte(self.dftarget_reduced, xcol="cluster", prozente=True))
        visdf = self.dftarget_reduced[list(filter(lambda x: x != 'cluster', self.dftarget_reduced.columns))]
        self.show_cluster_first_components(visdf, cluster_model)
        self.show_pairgrid_firstk_pcs(self.dftarget_reduced, color_by_cluster=True)

        print("done")


_ = xdl.lade_zufalldatenbestand(8, 8, 4, pinf=0.1, n_samples=50000, typ="clf", class_sep=3.5)
x_train, y_train, x_test, y_test, X_mins, X_max = _
ncols = x_train.shape[1]
feature_names = [f"feat_{w}" for w in range(ncols)]
dfref, dftarget = pd.DataFrame(x_train, columns=feature_names), pd.DataFrame(x_test, columns=feature_names)
feat_info = pd.DataFrame({"attribute": feature_names,
                          "information_level": ["pointlike"] * ncols,
                          "type": ["numeric"] * ncols,
                          "missing_or_unknown": [0.0] * ncols})
# drop_record_id = [False if w != 3 else (True if np.random.rand() < 0.7 else False) for w in y_test]
# dftarget = dftarget[[not w for w in drop_record_id]].reset_index(drop=True)
dftarget = dftarget[[w != 3 for w in y_test]].reset_index(drop=True)

map_none_replacement = {}
drop_records_by_features_with_null = {}
replacement_vals = {}
replacement_by_rfc = []
gv = GruppenVergleicher(
    dfref, dftarget, feat_info, 25000, map_none_replacement
)
gv.map_none_replacement = map_none_replacement
gv.drop_records_by_features_with_null = drop_records_by_features_with_null
gv.replacement_vals = replacement_vals
gv.replacement_by_rfc = replacement_by_rfc
gv.run()