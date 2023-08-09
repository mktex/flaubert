from importlib import reload
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from flaubert.model.decision_trees import dt


class FeatureWichtigket():

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
            ._random_forrest_classifier() \
            ._importances_classifier() \
            ._streu_von_weizen() \
            ._zeige_vis_clf_resultat() \
            ._check_wichtigkeit_features_mit_regressor_sim() \
            ._check_multikolinearitaet()

    def _setup_non_nullen(self):
        # Nullen simpel einsetzen
        self.dfcheck.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.dfcheck = self.dfcheck.fillna(self.dfcheck.mean())
        # Zeige eine Stichprobe
        print(self.dfcheck.sample(4).transpose())
        # input("...")
        return self

    def _setup_kateg_zielvariable(self):
        # Binäre Zielvariable
        self.Y = self.dfcheck[[self.zielvariable]]
        training_features = list(
            filter(lambda feature: feature not in [self.id_col, self.zielvariable], self.dfcheck.columns))
        self.X = self.dfcheck[training_features]
        est = KBinsDiscretizer(n_bins=self.nbins, encode='ordinal', strategy='kmeans')
        self.y = est.fit_transform(self.Y)[:, 0]
        return self

    def _setup_multikolinearitaet(self):
        # Entferne Multikolinearität
        from flaubert.transform import df_redux
        Xnmk = df_redux.reduktion_multikolinearitaet(self.X, threshold_corr=0.7, target_col=None)
        # Überschreibt X
        self.X = Xnmk
        # Führe die RFC noch einmal
        return self

    def _zeige_decision_tree_output(self):
        reload(dt)
        for check_warum_gruppe in range(self.nbins):
            print(
                "\n===================================================================================================")
            print("[x] Gruppe (entsprechend umgewandelter Zielvariable in Kategorie):", check_warum_gruppe)
            dtdf = self.X.copy()
            dtdf['_dt_target_check_'] = self.y.copy()
            dt.warum(dtdf, xcolumnsSet=filter(lambda x: x not in ["_dt_target_check_"], dtdf),
                     xcolTarget="_dt_target_check_",
                     xlambdaF=lambda x: 'g0' if x == check_warum_gruppe else 'g1',
                     useOrdered=["g0", "g1"], balancedN={"g0": None, "g1": None},
                     test_size=0.7, max_depth=4, min_samples_split=5, min_samples_leaf=5, criterion="entropy",
                     print_dtstruktur=True)
        # input("...")
        return self

    def _random_forrest_classifier(self):
        # Der Weg mittels RFC
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42)
        self.clf = RandomForestClassifier(n_estimators=125, random_state=42)
        self.clf.fit(self.X_train, self.y_train)
        print("[x] Accuracy (RandomForestClassifier): {:.2f}".format(self.clf.score(self.X_test, self.y_test)))
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
        self.wichtige_features = list(self.X.columns[self.tree_importance_sorted_idx][-self.topk:])
        self.wichtige_features.sort()
        self.unwichtige_features = list(self.X.columns[self.tree_importance_sorted_idx][:-self.topk])
        self.unwichtige_features.sort()
        print("\n[x] Unwichtige Features (die Wichtigen werden in den Visualisierungen angezeigt):")
        pprint(self.unwichtige_features)
        print("\n[x] Wichtige Features:")
        pprint(self.wichtige_features)
        # input("...")
        return self

    def _zeige_vis_clf_resultat(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.barh(self.tree_indices[:self.topk],
                 self.clf.feature_importances_[self.tree_importance_sorted_idx][-self.topk:], height=0.7)
        ax1.set_yticks(self.tree_indices[:self.topk])
        ax1.set_yticklabels(self.X.columns[self.tree_importance_sorted_idx][-self.topk:])
        ax1.set_ylim((0, len(self.clf.feature_importances_[-self.topk:])))
        ax2.boxplot(
            self.res_importances.importances[self.perm_sorted_idx[-self.topk:]].T,
            vert=False,
            labels=self.X.columns[self.perm_sorted_idx[-self.topk:]],
        )
        ax1.xaxis.set_tick_params(labelsize=7)
        ax1.yaxis.set_tick_params(labelsize=7)
        ax2.xaxis.set_tick_params(labelsize=7)
        ax2.yaxis.set_tick_params(labelsize=7)
        fig.tight_layout()
        plt.show()
        return self

    def _check_wichtigkeit_features_mit_regressor_sim(self):
        from sklearn.neural_network import MLPRegressor
        xNNS = MLPRegressor(hidden_layer_sizes=(8, 8), activation="logistic", solver="adam",
                            learning_rate="adaptive", max_iter=2500, batch_size=30, shuffle=False,
                            verbose=False, early_stopping=False
                            )
        xNNS.fit(self.X[self.wichtige_features], self.Y)
        features_sim = self.wichtige_features
        p = 1.35
        datensim = self.X[self.wichtige_features].iloc[pd.Series(range(self.X.shape[0])).sample(100).values]
        S = Simulate(obs=datensim, var=features_sim)
        d = S.simulate_increase(model=xNNS, p=p)
        S.plot_simulation(d, title='')
        return self

    def _check_multikolinearitaet(self):
        # Multikolinearität
        from scipy.stats import spearmanr
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
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
            ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
            ax2.set_yticklabels(dendro["ivl"])
            ax1.xaxis.set_tick_params(labelsize=7)
            ax1.yaxis.set_tick_params(labelsize=7)
            ax2.xaxis.set_tick_params(labelsize=7)
            ax2.yaxis.set_tick_params(labelsize=7)
            fig.tight_layout()
            plt.show()
        except:
            traceback.print_exc()

        return self


class Simulate:
    def __init__(self, obs, var):
        self.obs = obs
        self.var = var

    def simulate_increase(self, model, p):
        baseline = model.predict(self.obs)
        simdict = {}
        print(self.obs)
        for ivar in self.var:
            X_plus = self.obs.copy()
            # print(X_plus[ivar])
            X_plus[ivar] = X_plus[ivar].values * p
            simdict[ivar] = [model.predict(X_plus).mean()]
        # print(simdict)
        b = pd.DataFrame(simdict).T.reset_index()
        b.columns = ["feature_simuliert", "sim_wert"]
        b['baseline'] = baseline.mean()
        return b

    @staticmethod
    def plot_simulation(d, **kwargs):
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(y='feature_simuliert', x='sim_wert', data=d, palette='deep', ax=ax)
        ax.axvline(d['baseline'].values[0], color='grey', linestyle='--', linewidth=2)
        ax.plot([-100, -100], [0, 0], color='grey', linestyle='--', linewidth=2, label='baseline')
        maxi = 1.05 * d['sim_wert'].max()
        mini = 0.95 * d['sim_wert'].min()
        ax.set_xlim([mini, maxi])
        ax.set_xlabel('Simulation')
        ax.set_ylabel('Variablen')
        # ax.set_title(kwargs.get('title'))
        ax.legend()
        ax.grid(axis='x', linewidth=.3)
        sns.despine(offset=10, trim=True)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.show()
