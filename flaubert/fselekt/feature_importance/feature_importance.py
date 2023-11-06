from importlib import reload
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sbn
import traceback

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

from flaubert.model.decision_trees import dt
from flaubert.transform import df_redux, transform_norm as tnorm
from flaubert import einstellungen


class FeatureWichtigkeit():
    model_iter_train = 500
    p_sample_simulation = 0.5

    rfc_model = {}
    rfc_data = None

    nn_model = None
    nn_model_data = None

    def __init__(self, dfcheck, zielvariable, id_col="id", topk=50, nbins=3):
        self.dfcheck = dfcheck
        self.zielvariable = zielvariable
        self.id_col = id_col

        self.X = None

        # kontinuierliche Form der Zielvariable
        self.Y = None
        # kategoriale Form der Zielvariable
        self.y = None

        # Classifier
        self.clf = None

        # Train und Test Daten
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

        # Anzahl der maximal angezeigten Features
        self.topk = topk
        self.nbins = nbins

        # Wichtigkeitsscores aus Klassifizierung
        self.res_importances, self.perm_sorted_idx, self.tree_importance_sorted_idx, self.tree_indices = [None] * 4

        self.wichtige_features, self.unwichtige_features = [None] * 2

    def run(self):
        return self._setup_non_nullen() \
            ._setup_kateg_zielvariable() \
            ._setup_multikolinearitaet() \
            ._zeige_decision_tree_output() \
            ._random_forest_classifier() \
            ._importances_classifier() \
            ._streu_von_weizen() \
            ._zeige_vis_clf_resultat() \
            ._check_wichtigkeit_features_mit_regressor_sim() \
            ._do_sim_input_var() \
            ._check_multikolinearitaet()

    def _setup_non_nullen(self):
        # Nullen simpel einsetzen
        self.dfcheck.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.dfcheck = self.dfcheck.fillna(self.dfcheck.mean())
        return self

    def _setup_kateg_zielvariable(self):
        # Binäre Zielvariable
        self.Y = self.dfcheck[[self.zielvariable]]
        training_features = list(
            filter(lambda feature: feature not in [self.id_col, self.zielvariable], self.dfcheck.columns))
        self.X = self.dfcheck[training_features]
        est = KBinsDiscretizer(n_bins=self.nbins, encode='ordinal', strategy='kmeans')
        estmodel = est.fit(self.Y)
        print("[x] Bins:")
        bins = np.round(np.array(estmodel.bin_edges_[0]), 1)
        self.bins_df = pd.DataFrame(list(zip(range(len(bins)), bins)), columns=["Bin", "Zentrum"])
        print(self.bins_df)
        self.y = estmodel.transform(self.Y).flatten().astype(int)
        return self

    def _setup_multikolinearitaet(self):
        # Entferne Multikolinearität
        Xnmk = df_redux.reduktion_multikolinearitaet(self.X, threshold_corr=0.95,
                                                     target_col=None, id_col=None)
        # Überschreibt X
        self.X = Xnmk
        # Führe die RFC noch einmal
        return self

    def _zeige_decision_tree_output(self):
        for check_warum_gruppe in range(self.nbins):
            print(
                "\n===================================================================================================")
            print("[x] Gruppe (entsprechend umgewandelter Zielvariable in Kategorie):", check_warum_gruppe)
            dtdf = self.X.copy()
            dtdf['_dt_target_check_'] = self.y.copy()
            reload(dt)
            dt.warum(dtdf, xcolumnsSet=filter(lambda x: x not in ["_dt_target_check_"], dtdf),
                     xcolTarget="_dt_target_check_",
                     xlambdaF=lambda x: 'g1' if x == check_warum_gruppe else 'g0',
                     useOrdered=["g0", "g1"], balancedN={"g0": None, "g1": None},
                     test_size=0.7, max_depth=4, min_samples_split=5, min_samples_leaf=5, criterion="entropy",
                     print_stats=True, print_dtstruktur=True, dt_visualisierung=False, fname_dt=None
                     )
            self.rfc_model[check_warum_gruppe] = dt.clf
        return self

    def _random_forest_classifier(self):
        # Der Weg mittels RFC
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42)
        self.clf = RandomForestClassifier(n_estimators=125, random_state=42)
        self.clf.fit(self.X_train, self.y_train)
        self._zeige_model_perf(self.y_test, self.clf.predict(self.X_test),
                               model_typ="RandomForestClassifier", labels_zuordnung=self.clf.classes_)
        self.rfc_model['RFC'] = self.clf
        return self

    def _importances_classifier(self):
        # Berechne die Feature-Wichtigkeitsscores
        self.res_importances = permutation_importance(self.clf, self.X_train, self.y_train, n_repeats=10,
                                                      random_state=42)
        self.perm_sorted_idx = self.res_importances.importances_mean.argsort()  # ascending order
        self.tree_importance_sorted_idx = np.argsort(self.clf.feature_importances_)
        self.tree_indices = np.arange(0, len(self.clf.feature_importances_)) + 0.5
        return self

    def _streu_von_weizen(self):
        # Resultierende Spalten sind:
        features_rfc_imp = self.X.columns[self.tree_importance_sorted_idx][-self.topk:]
        features_perm_imp = self.X.columns[self.perm_sorted_idx][-self.topk:]
        features_diff = list(filter(lambda feature: feature not in features_rfc_imp, features_perm_imp))
        self.wichtige_features = list(features_rfc_imp) + features_diff
        self.wichtige_features.sort()
        self.unwichtige_features = list(self.X.columns[self.tree_importance_sorted_idx][:-self.topk])
        self.unwichtige_features.sort()
        print("\n[x] Unwichtige Features (die Wichtigen werden in den Visualisierungen angezeigt):")
        pprint(self.unwichtige_features)
        print("\n[x] Wichtige Features:")
        pprint(self.wichtige_features)
        return self

    def _zeige_vis_clf_resultat(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_HOCH)
        ax1.barh(self.tree_indices[:self.topk],
                 self.clf.feature_importances_[self.tree_importance_sorted_idx][-self.topk:],
                 height=0.7, color="cyan")
        ax1.set_yticks(self.tree_indices[:self.topk])
        ax1.set_yticklabels(self.X.columns[self.tree_importance_sorted_idx][-self.topk:])
        ax1.set_ylim((0, len(self.clf.feature_importances_[-self.topk:])))
        ax1.xaxis.set_tick_params(labelsize=7)
        ax1.yaxis.set_tick_params(labelsize=7)
        ax1.set_xlabel("Normalized average reduction in Gini impurity", fontsize=7)
        ax2.boxplot(
            self.res_importances.importances[self.perm_sorted_idx[-self.topk:]].T,
            vert=False,
            labels=self.X.columns[self.perm_sorted_idx[-self.topk:]],
        )
        ax2.xaxis.set_tick_params(labelsize=7)
        ax2.yaxis.set_tick_params(labelsize=7)
        ax2.set_xlabel("Average accuracy improvement", fontsize=7)
        fig.tight_layout()
        plt.show()
        return self

    def _zeige_model_perf(self, y_true, y_pred, model_typ="RFC", labels_zuordnung=(0, 1)):
        print(f"\n[x] Performanz Model ({model_typ}):")
        accuracy = metrics.accuracy_score(y_true, y_pred)
        print(classification_report(y_pred, y_true, zero_division=0.0))
        cm = confusion_matrix(y_true, y_pred, labels=labels_zuordnung)
        print("\nConfusion Matrix:")
        print(cm)
        print("\nAccuracy:", accuracy)

    def _check_wichtigkeit_features_mit_regressor_sim(self):
        X_norm, xmin, xmax = tnorm.normalisierung_mehrdimensional(self.X[self.wichtige_features].values,
                                                                  a=0, b=1)
        self.nn_model_data = pd.DataFrame(X_norm, columns=self.wichtige_features)

        print("\n[x] MLPClassifier Training..")
        xNNS = MLPClassifier(hidden_layer_sizes=(8, 8), activation="logistic",
                             solver="adam", learning_rate="adaptive",
                             learning_rate_init=0.001, max_iter=self.model_iter_train, batch_size=30,
                             momentum=0.0, shuffle=True, verbose=False,
                             early_stopping=True, tol=1e-10, n_iter_no_change=350,
                             alpha=0.01, validation_fraction=0.2)
        xNNS.fit(self.nn_model_data.values, self.y)
        self.nn_model = xNNS
        xNNOutputPred = xNNS.predict(self.nn_model_data.values)
        self._zeige_model_perf(self.y, xNNOutputPred,
                               model_typ="MLPClassifier",
                               labels_zuordnung=self.clf.classes_)

        return self

    def _do_sim_input_var(self):
        def get_samples_p_df(nn_model_data):
            r = 0.5 * nn_model_data.std()
            r = pd.DataFrame(list(zip(r.index.values, r.values)), columns=["feature", "wert_0.5_sigma"])
            for i in np.arange(1.0, 3.5, 0.1):
                r[f"wert_{i}_sigma"] = i * nn_model_data.std().values
            r = r.set_index("feature").transpose().reset_index(drop=True)
            return r

        plist_df = get_samples_p_df(self.nn_model_data)
        nsamples = int(self.p_sample_simulation * self.nn_model_data.shape[0])
        datensim = self.nn_model_data.iloc[pd.Series(range(self.nn_model_data.shape[0])).sample(nsamples).values]
        S = Simulate(obs=datensim, var=self.wichtige_features)
        d = S.simulate_increase(model=self.nn_model, plist_df=plist_df)
        S.plot_simulation(d, title='')
        return self

    def _check_multikolinearitaet(self):
        # Multikolinearität
        from scipy.stats import spearmanr
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_HOCH)
            corr = spearmanr(self.X).correlation
            # Ensure the correlation matrix is symmetric
            corr = (corr + corr.T) / 2
            np.fill_diagonal(corr, 1)

            # We convert the correlation matrix to a distance matrix before performing
            # hierarchical clustering using Ward's linkage.
            distance_matrix = 1 - np.abs(np.nan_to_num(corr))
            dist_linkage = hierarchy.ward(squareform(distance_matrix))
            dendro = hierarchy.dendrogram(
                dist_linkage, labels=self.X.columns.tolist(), ax=ax1, leaf_rotation=90
            )
            dendro_idx = np.arange(0, len(dendro["ivl"]))

            ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
            ax2.set_xticks(dendro_idx)
            ax2.set_yticks(dendro_idx)
            ax2.set_xticklabels(dendro["ivl"], rotation=45)
            ax2.set_yticklabels(dendro["ivl"])
            ax1.xaxis.set_tick_params(labelsize=5, rotation=45)
            ax1.yaxis.set_tick_params(labelsize=5)
            ax2.xaxis.set_tick_params(labelsize=5, rotation=45)
            ax2.yaxis.set_tick_params(labelsize=5)
            fig.tight_layout()
            plt.show()

        except:
            traceback.print_exc()

        return self


class Simulate:
    def __init__(self, obs, var):
        self.obs = obs
        self.var = var

    def simulate_increase(self, model, plist_df):
        baseline = model.predict(self.obs.values)
        simdict = {}
        for ivar in self.var:
            X_plus = self.obs.copy()
            werte = X_plus[ivar].values.copy()
            simvalues = []
            for p in plist_df[ivar].values:
                X_plus[ivar] = werte + p
                simvalues.append(model.predict(X_plus.values).mean())
            # plt.scatter(x=simvalues, y=plist_df[ivar].values)
            # plt.title(ivar)
            # plt.show()
            simdict[ivar] = [np.mean(simvalues)]

        b = pd.DataFrame(simdict).T.reset_index()
        b.columns = ["feature_simuliert", "sim_wert"]
        b['baseline'] = [baseline.mean()] * b.shape[0]
        b = b.sort_values(by="sim_wert", ascending=False)

        return b

    @staticmethod
    def plot_simulation(d, **kwargs):
        fig, ax = plt.subplots(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_HOCH)
        sbn.barplot(y='feature_simuliert', x='sim_wert', data=d,
                    palette="light:b", ax=ax)
        ax.axvline(d['baseline'].values[0], color='grey', linestyle='--', linewidth=2)
        ax.plot([-100, -100], [0, 0], color='grey', linestyle='--', linewidth=2, label='baseline')
        maxi = 1.05 * d['sim_wert'].max()
        mini = 0.95 * d['sim_wert'].min()
        ax.set_xlim([mini, maxi])
        ax.set_xlabel('Simulierte Features')
        ax.set_ylabel('Zielvariable')
        ax.legend()
        ax.grid(axis='x', linewidth=.3)
        sbn.despine(offset=10, trim=True)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.show()
