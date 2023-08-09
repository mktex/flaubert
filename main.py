
# Main EDA
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce
import statsmodels
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from flaubert import einstellungen
from flaubert.eda import dbeschreiben, dfgestalt
from flaubert.ersatz import dfersatz
from flaubert.transform import kateg, df_redux, transform_norm as tnorm
from flaubert.fselekt import feature_scoring, feature_importance
from flaubert.model.neural_networks import mlp, xnn
from flaubert.model.decision_trees import dt
from flaubert.model.clustering import xDBScan
from flaubert.statistik import abtest, stat_op
from flaubert.vis import xdiagramme as xdg
from flaubert.model.outliers import xIQR

if True:
    reload(dbeschreiben)
    reload(dfgestalt)
    reload(einstellungen)
    reload(kateg)
    reload(df_redux)
    reload(feature_scoring)
    reload(feature_importance)
    reload(dt)
    reload(abtest)
    reload(stat_op)
    reload(mlp)
    reload(xnn)
    reload(tnorm)
    reload(xdg)
    reload(dfersatz)
    reload(xIQR)

# ---------------------------------
# Einstellungen
from sklearn.datasets import make_classification
"""
Beispiel:
X, Y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                            n_redundant=0, n_repeated=0,
                            n_classes=2, n_clusters_per_class=1,
                            weights=(0.8, 0.2),
                            flip_y=0.1, class_sep=2.0,
                            hypercube=True, shift=0.5, scale=5,
                            shuffle=True, random_state=42)
"""
X, Y = make_classification(n_samples=5000, n_features=35, n_informative=10,
                            n_redundant=5, n_repeated=3,
                            n_classes=2,
                            n_clusters_per_class=1,
                            weights=(0.8, 0.2),
                            hypercube=True,
                            flip_y=0.1,
                            class_sep=5, shift=5, scale=2.0,
                            shuffle=True, random_state=42)
xdata = pd.DataFrame(np.hstack([X, Y.reshape(-1, 1)]), columns=[f"F{k}" for k in range(X.shape[1])] + ["T"])
xdg.do_pairplot(xdata, xselekt=pd.Series(xdata.columns[:-1]).sample(6).tolist() + ["T"])
xdg.show_summary_scatterplot_farbe_kateg(xdata, 'F15', 'F18', 'T')

# Pandas CSV..
xdata = pd.read_csv("./ngboost/data/surv/flchain.csv", encoding="utf-8")
xdg.do_pairplot(xdata, xselekt=pd.Series(xdata.columns[:-2]).sample(6).tolist() + ["death"])
xdg.show_summary_scatterplot_farbe_kateg(xdata, 'age', 'creatinine', 'death')

# einstellung ID und Zielvariable
id_col = "id"
target_col = "T"
zielvariable_dict = {
    "Yes": 1, "No": 0
}
zielv_dict_inv = dict(list(map(lambda x: (x[1], x[0]), zielvariable_dict.items())))
features = list(filter(lambda x: x not in [target_col, id_col], xdata.columns))
entferne_features = ["Over18", "EmployeeCount", "StandardHours", "recently_promoted",
                     "tenure", "filed_complaint", 'Unnamed: 83',
                     'encounter_id', 'patient_id', 'hospital_id', 'chapter']
features = list(filter(lambda feature: feature not in entferne_features, features))

# Erzeuge ID und entferne feature
xdata["id"] = list(range(xdata.shape[0]))
xdata = xdata[[id_col] + features + [target_col]]

# Sampling
# xdata = xdata.sample(10000).reset_index(drop=True)

# dbeschreiben
if True:
    dbeschreiben.zeige_kateg_features(xdata, target_col, id_col)
    xdg.histogram_stack_kateg(xdata, None, target_col)

# Struktur: id | .. | Zielvariable
xdata = xdata[[id_col] +
              list(filter(lambda feature: feature not in [id_col, target_col], xdata.columns)) +
              [target_col]
]


# ---------------------------------
# Behandlung kategorialen Variablen

# Tranformation der ordinal- kategorialen Features zu numerische Features
ersatz_dict = {
    'BusinessTravel': {'Travel_Rarely': 1, 'Travel_Frequently': 2, 'Non-Travel': 0},
    'OverTime': {'No': 0, 'Yes': 1},
    'salary': {'low': 0, 'medium': 1, 'high': 2}
     # target_col: zielvariable_dict
}
for feature in list(ersatz_dict.keys()):
    if feature in xdata.columns:
        xdata[feature] = [ersatz_dict[feature][wert] for wert in xdata[feature].values]

# dichtediagramm eines Features
xdg.histogram_dichte_diagramm(xdata, target_col, titel="", alpha=0.025)

# dfgestalt
dfnum, dfcat, num_cols, cat_cols = dfgestalt.num_cat_trennen_desc(xdata, target_col, id_col)

# ---------------------------------
# Einsetzen der Outlier-Werte als NaN
outliers_dict = {}
for feature in dfnum.columns[:-2]:
    outliers = xIQR.run(dfnum[feature], return_iqr_params=False, ident_residuen=0)
    if outliers.shape[0] > 0:
        print(feature)
        outidx = outliers.index.tolist()
        outliers_dict[feature] = outidx
        xdg.boxplot(dfnum, feature, target_col)
        xdg.boxplot(dfnum.loc[~dfnum.index.isin(outidx)], feature, target_col)

# None Werte für Outliers
print(outliers_dict.keys())
for feature in outliers_dict.keys():
    for id in outliers_dict[feature]:
        dfnum.at[id, feature] = None


# ---------------------------------
# Behandlung Nullwerte
nullendf = dbeschreiben.prozent_nullen_in_df(dfnum)
print(nullendf[nullendf[1] > 0])

# Nullen mit Clustering ersetzen (funktioniert nur für numerische Features)
dfnum = dfersatz.get_dataframe_filled(dfnum, nullendf[nullendf[1] > 0][0].tolist(), id_col, target_col)

# Nullen mit Mode für kategoriale Features
dfcat = dfersatz.ersatz_nullwerte_durch_mode(dfcat)
# dfcat['chapter'] = ['NAWert' if 'nan' == str(x) else x for x in dfcat['chapter'].values.tolist()]

# Nullen behandeln mit NN Modelle
# -- Modelaufbau für ein numerisches Feature
if True:
    feature_target = "creatinine"
    dfnum, origwerte = dfersatz.setze_nullen_zuffall(dfnum, feature_target, n=50)
    xclude = [target_col, id_col]
    dftrain_feature = pd.merge(dfnum[num_cols].copy(),
                               kateg.kateg2dummy(dfcat[cat_cols], sep=None),
                               left_index=True, right_index=True) if dfcat.shape[1] > 2 else dfnum[num_cols].copy()
    dftrain_feature = dftrain_feature[list(filter(lambda x: x not in xclude, dftrain_feature.columns))]
    datenpaket = xnn.input_numeric_nn(dftrain_feature,
                                 xcolTarget=feature_target, hidden_layer_sizes=(16, 16), verbose=True, aktivF="logistic",
                                 skalierung=(0, 1), standardisierung=True, entferneKorr=True, thresholdKorr=0.01,
                                 namedLearner=f"model_nn_{feature_target}")
    idmap, model_nn, transform_dict, model_namen = datenpaket
    # Anwendung auf dfnum => fehlende Werte wurden ersetzt
    dftrain_feature = xnn.apply_modell_nn_fuer_feature(dftrain_feature, feature_target, model_namen, ersatz_in_df=True)
    dfnum = pd.merge(dftrain_feature[num_cols], dfnum[xclude], left_index=True, right_index=True)

# -- Modelaufbau für ein kategoriales Feature
dfcat['ist_LifeSciences'] = list(map(lambda x: 'Ja' if x == 'Life Sciences' else 'Nein', dfcat['EducationField'].values))
feature_target = 'ist_LifeSciences' # "EducationField" # "JobRole"
dfcat, origwerte = dfersatz.setze_nullen_zuffall(dfcat, feature_target, n=50)
xclude = [target_col, id_col, 'EducationField']
dfcat_filtered = dfcat[list(filter(lambda x: x not in xclude, cat_cols))]
dftrain_feature = pd.merge(dfnum[num_cols].copy(),
                           kateg.kateg2dummy(dfcat_filtered, sep=None),
                           left_index=True, right_index=True)
dftrain_feature = pd.merge(dftrain_feature, dfcat[[feature_target]], left_index=True, right_index=True)
datenpaket = xnn.input_class_nn(dftrain_feature, xcolTarget=feature_target, xcolID=None, showStuff=True,
                   hidden_layer_sizes=(16, 16), aktivF="logistic",
                   skalierung=(0, 1), standardisierung=True,
                   namedLearner=f"model_nn_{feature_target}")
# Erklärung mit einem DT
dbeschreiben.frequenz_werte(dftrain_feature, group_by_feature=feature_target, prozente=True, sep=None, id_col=id_col)
dtmodel_feature = dt.warum(dfersatz.ersatz_nullwerte_durch_mean(dftrain_feature),
                           xcolumnsSet=filter(lambda x: x != feature_target, dftrain_feature.columns),
                           xcolTarget=feature_target,
                           xlambdaF=lambda x: 'g0' if x == 'Nein' else 'g1',
                           useOrdered=["g0", "g1"], balancedN=None,  # {'g0': 237*3, 'g1': 237},
                           test_size=0.2, max_depth=4, min_samples_split=5,
                           min_samples_leaf=25, criterion="entropy",
                           print_stats=True, print_dtstruktur=True,
                           dt_visualisierung=True, fname_dt=f"dt_model_feature_{feature_target}")

idmap, model_nn, transform_dict, model_namen = datenpaket
# Anwendung auf dfnum => fehlende Werte wurden ersetzt
dfnum = xnn.apply_modell_nn_fuer_feature(dfnum, feature_target, model_namen, ersatz_in_df=True)


# ---------------------------------
# Visualisierungen und Hypothesentests

# Balkendiagramme (für kategoriale Features) und Histogramme mit Dichtefunktionen (für numerische Variablen)
if True:
    for feature in cat_cols: xdg.show_counts_kateg(dfcat, feature, target_col)
    for feature in num_cols: xdg.histogram_dichte_kateg(dfnum, feature, target_col)

if True:
    feature_cat = "sex"
    feature_num = "lambda"
    xdg.histogram_stack_kateg(dfcat, feature_cat, target_col, returnBinData=False)
    xdg.histogram_dichte_diagramm(dfnum, feature_num, titel="", alpha=0.025)

# Zielvariable pro kategoriale Variablen
if True:
    print(einstellungen.TRENNLINIE_TYP_II)
    print("[x] Durchschnittswerte für kategoriale Variablen:")
    print(cat_cols)
    for feature in cat_cols:
        print(einstellungen.TRENNLINIE_TYP_I)
        print(f"[x] {feature}:")
        dbeschreiben.durchschnittswerte_pro_kateg(dfcat, feature, target_col)

# Führe Hypothesentests auf kategorialen und numerischen Features
if True:
    feature_cat = 'gender'
    werta, wertb = "M", "F"
    dbeschreiben.frequenz_werte(dfcat, group_by_feature=feature_cat, prozente=True, sep=None, id_col=id_col)
    xdg.show_counts_kateg(dfcat, feature_cat, target_col)
    stat_op.statistik_test(dfcat, xcol=feature_cat, werta=werta, wertb=wertb, target_col=target_col)

if True:
    feature_num = 'JobInvolvement'
    dbeschreiben.zeige_korrelationen(dfnum, target_col, id_col, threshold_corr=0.01)
    gr_a = dfnum[dfnum[target_col] == 0][feature_num]
    gr_b = dfnum[dfnum[target_col] == 1][feature_num]
    mu1, sd1, mu2, sd2 = gr_a.mean(), gr_a.std(), gr_b.mean(), gr_b.std()
    stat_op.simulation_ttest(mu1, sd1, mu2, sd2)


# ---------------------------------
# Transformation 1-hot
if dfcat.shape[1] > 2:
    xdfcat_dummy = kateg.kateg2dummy(dfcat[cat_cols], sep=None)
    xdfcat_dummy = pd.merge(xdfcat_dummy, dfcat[[target_col, id_col]],
                            left_index=True, right_index=True)
    print(xdfcat_dummy)


# ---------------------------------
# Reduktion der Dimensionalität
if dfcat.shape[1] > 2:
    dfcat_redux = df_redux.reduktion_multikolinearitaet(xdfcat_dummy, threshold_corr=0.8,
                                                        target_col=target_col,
                                                        id_col=id_col)
    dfcat_scored = feature_scoring.features_reduzierung(dfcat_redux, target_col, id_col, 15, doPlot=True)

# Reduzierung Dimensionalität durch Multikolinearität
dfnum_redux = df_redux.reduktion_multikolinearitaet(dfnum, threshold_corr=0.8,
                                                    target_col=target_col,
                                                    id_col=id_col)
dfnum_scored = feature_scoring.features_reduzierung(dfnum_redux, target_col, id_col, 15, doPlot=True)

# Zusammenstellung Datenbestand
if True:
    df = pd.merge(dfcat_scored, dfnum_scored,
                  left_on=[id_col, target_col], right_on=[id_col, target_col]) if dfcat.shape[1] > 2 else dfnum_scored
    df = df[list(filter(lambda feature: feature != id_col, df.columns))]
    df = df[list(filter(lambda feature: feature != target_col, df.columns)) + [target_col]]
    print("[x] Feature Importances")
    fw = feature_importance.FeatureWichtigket(df, target_col, id_col=id_col, topk=25, nbins=2).run()
    # Hier SPEICHERN


# ---------------------------------
# Durchcheck Transformationen
num_features_verblieben = list(filter(lambda x: x in dfnum_redux.columns and x not in [id_col, target_col],
                                      dfnum.columns))
for feature in list(filter(lambda x: x not in [target_col, id_col], num_features_verblieben)):
    tnorm.zeige_transform_effekt_I(dfnum_redux, feature)

# ---------------------------------
# Visualisierungen Suche nach Clusters
if True:
    xdg.show_hist_lmres(df[fw.wichtige_features])
    xdg.show_heatmap(df, target_col, list(df.columns))
    xdg.do_pairplot(df, xselekt=pd.Series(fw.wichtige_features).sample(6).tolist() + [target_col])
    xdg.show_summary_scatterplot_farbe_kateg(df, 'F30', 'F15', 'T')

# Durchchecke Anzahl 0 vs 1
dbeschreiben.frequenz_werte(df, group_by_feature=target_col, prozente=False)
nklase_1 = 2169

# ---------------------------------
# Clustering Model: DBScan
if True:
    # Iterativ, suche den besten eps
    # tcl = 1 # [df[target_col] == tcl]
    features_clf = df[fw.wichtige_features].copy() # features_train_norm # [np.where(labels_train_norm == 0)[0],:]
    X_norm, xmin, xmax = tnorm.normalisierung_mehrdimensional(features_clf.values, a=0, b=1)
    features_clf = pd.DataFrame(X_norm, columns=features_clf.columns)
    reload(xDBScan)
    xDBScan.set_data(features_clf, featureNames=fw.wichtige_features)
    clust_stats_dict = xDBScan.get_durchschnittliche_diff_dist(sample_n=0.5)
    maxv = clust_stats_dict['mean']
    dfcl = xDBScan.get_optim_dbscan_params_grid(features_clf, xmetric="euclidean", min_val=0.01, max_val=maxv, minsamp=5)

if True:
    reload(xDBScan)
    xDBScan.set_data(features_clf, featureNames=fw.wichtige_features)
    xDBScan.do_model(eps=0.814046, k=5, xmetric="euclidean")
    clusters = list(xDBScan.xDICT_ERKLAERUNG.keys())
    cluster_features = []
    print("")
    for cluster_klasse in clusters:
        if cluster_klasse != -1:
            print(f"[x] Cluster Klasse {cluster_klasse}:")
            print(xDBScan.xDICT_ERKLAERUNG[cluster_klasse][0])
            ist_cluster_k = xDBScan.xDICT_ERKLAERUNG[cluster_klasse][1].predict(features_clf)
            clfeature = f"cl_{cluster_klasse}" # {tcl}_
            df[clfeature] = ist_cluster_k
            cluster_features.append(clfeature)


model_df_train = df[fw.wichtige_features + cluster_features + [target_col]]
print(model_df_train)

if True:
    datapaket = dfgestalt.split_data_groups(model_df_train,
                                            feature_liste=fw.wichtige_features + cluster_features,
                                            target_col=target_col,
                                            lambdafunk=lambda x: 'g0' if x < 1.0 else 'g1',
                                            use_ordered=["g0", "g1"], balanced_N={'g0': nklase_1*2, 'g1': nklase_1})
    X, y, dframe, labels, xlabels_basis = datapaket
    X_norm, xmin, xmax = tnorm.normalisierung_mehrdimensional(X, a=0, b=1)
    datenpaket = train_test_split(X_norm, labels, test_size=0.2)
    features_train_norm, features_test_norm, labels_train_norm, labels_test_norm = datenpaket


# TRAIN Datenbestand
if True:
    dt_data = pd.DataFrame(np.hstack([features_train_norm, labels_train_norm.reshape(-1, 1)]),
                               columns=(fw.wichtige_features + cluster_features + [target_col]))
    dbeschreiben.frequenz_werte(dt_data, group_by_feature=target_col, prozente=False)


# DT Model
if True:
    train_nklase_g1 = 1700
    dt.vis_conf_mat = True
    dtmodel = dt.warum(dt_data,
                       xcolumnsSet=filter(lambda x: x != target_col, dt_data.columns),
                       xcolTarget=target_col,
                       xlambdaF=lambda x: 'g0' if x < 1.0 else 'g1',
                       useOrdered=["g0", "g1"], balancedN={'g0': train_nklase_g1*2, 'g1': train_nklase_g1},
                       test_size=0.2, max_depth=6, min_samples_split=5,
                       min_samples_leaf=25, criterion="entropy",
                       print_stats=True, print_dtstruktur=True,
                       dt_visualisierung=True, fname_dt="dt_model")
    clfmodel = dtmodel
    # dt_datenpaket = dt.features_train, dt.features_test, dt.labels_train, dt.labels_test
    # features_train, features_test, labels_train, labels_test = dt_datenpaket

# Optimierung
if True:
    feature_names = list(filter(lambda x: x != target_col, dt_data.columns))
    dt.do_model_optim()
    dt.clf = dt.clf_optim
    clfmodel = dt.clf
    print("\n", dt.print_decision_tree(dt.clf, feature_names=feature_names), "\n")
    dt.export_dt_vis(feature_names, fname_dt="dt_optim")


# ---------------------------------  Optimierung des Schwellenwertes ---------------------------------
if True:
    RocCurveDisplay.from_estimator(clfmodel, features_test_norm, labels_test_norm); plt.show()
    labels_test_pred = clfmodel.predict_proba(features_test_norm)
    fpr, tpr, thresholds = roc_curve(labels_test_norm, labels_test_pred[:,1])
    labels_sorted = clfmodel.classes_

if True:
    dt_stats = []
    for t in list(filter(lambda x: 0 <= x <= 1, thresholds)):
        labels_test_pred = (clfmodel.predict_proba(features_test_norm)[:, 1] >= t).astype(int)
        accuracy = accuracy_score(labels_test_norm, labels_test_pred)  # (y_pred == y_true).mean()
        cm = confusion_matrix(labels_test_norm, labels_test_pred, labels=labels_sorted)
        precision, recall, f1_score = stat_op.get_confusion_matrix_stats(cm, 1)
        dt_stats.append([t, accuracy, precision, recall, f1_score])
    dt_stats = pd.DataFrame(dt_stats, columns=["threshold", "accuracy", "precision", "recall", "f1_score"])
    dt_stats = dt_stats.sort_values(by="threshold", ascending=True).reset_index(drop=True)
    print(dt_stats)
    for feature, farbe in [("accuracy", "red"), ("precision", "blue"), ("recall", "black"), ("f1_score", "orange")]:
        plt.plot(dt_stats["threshold"], dt_stats[feature], c=farbe, label=feature)
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.show()

if True:
    best_threshold = 0.53
    labels_test_pred = (clfmodel.predict_proba(features_test_norm)[:, 1] >= best_threshold).astype(int)
    # labels_test_pred = stat_op.apply_clf_model(features_test, clfmodel, best_threshold, 1)
    stat_op.print_clf_statistics(clfmodel, features_train_norm, labels_train_norm,
                                 features_test_norm, labels_test_norm, best_threshold, 1,
                                 vis_confusion_mat=True)

# ROC mit Threshold
if True:
    fpr, tpr, thresholds = roc_curve(labels_test_norm, clfmodel.predict_proba(features_test_norm)[:, 1])
    auc_score = auc(fpr, tpr)
    label=f"Model: AUC: {auc_score}"
    plt.plot(fpr, tpr, '-', linewidth=4, label="Model NN/DT")
    plt.legend(loc="upper right")
    plt.title("Receiver Operating Characteristic")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

if True:
    from flaubert.model.decision_trees import xbayes
    N = features_train_norm.shape[0]
    cm = confusion_matrix(labels_test_norm, labels_test_pred, labels=labels_sorted)
    x0 = cm[1, :].sum() # Test == True
    x1 = cm[1, 1] # Test == True, K == True => sollte so hoch wie möglich sein
    x4 = cm[0, 1] # Test == False, K == True => sollte so niedrig wie möglich sein
    xbayes.set_data(N, x0, x1, x4)
    xbayes.confusion_matrix()
    xbayes.stats_basic()


# ---------------------------------
# Model NN
if True:
    model_nn_clf = mlp.get_mlp_clf(features_train_norm, labels_train_norm,
                                   ist_train_test_packet=False,
                                   lernrate=0.01, zyklen=1250, struktur=(16, len(clusters), 2),
                                   batch_size=50, early_stopping=True, alpha=0.0,
                                   verbose=False)
    stat_op.stats_perf_clf(features_test_norm, labels_test_norm, clf=model_nn_clf, vis_confusion_mat=True)
    # Für Optimierung mit Threshold
    clfmodel = model_nn_clf




# ---------------------------------
# NGBoost Methode

# NGBoost Methode
from ngboost import NGBClassifier, NGBSurvival
from ngboost.distns import k_categorical, LogNormal
from scipy.stats import lognorm

# ngboost classif
model_ngb_clf = NGBClassifier(Dist=k_categorical(2))
model_ngb_clf.fit(X, y.flatten())
y_pred = model_ngb_clf.predict(X)
# y_pred_dist = model_ngb_clf.predict_proba(X)
y_true = y.flatten()
stat_op.stats_perf_clf(X, y_true, clf=model_ngb_clf, labels_sorted=[0, 1])

# ngboost survival
X_train = df[list(filter(lambda x: x != "Survival", fw.wichtige_features))].values
y_train = df["Status"]
T_train = df[["Survival"]].values
E_train = df["Status"].values
ngb = NGBSurvival(Dist=LogNormal, n_estimators=1000).fit(X_train, T_train, E_train)
Y_preds = ngb.predict(X_train)
Y_dists = ngb.pred_dist(X_train)

# test Mean Squared Error
plt.scatter(x=T_train, y=Y_preds, s=4); plt.show()

def testme(k=None):
    x = list(set(T_train.flatten()))
    x.sort()
    x = np.array(x)
    locs = Y_dists.loc
    scales = Y_dists.scale
    if k is None:
        k = pd.Series(list(range(T_train.shape[0]))).sample().values[0]
    s = scales[k]
    loc = locs[k]
    print(f"[x] lognormal mit s={s} und scale=np.exp({loc})")
    # x = np.linspace(lognorm.ppf(0.01, s), lognorm.ppf(0.99, s), 100)
    rv = lognorm(s=s, scale=np.exp(loc))
    plt.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    plt.title(f"Voraussage: {np.round(rv.mean(), 2)} IST:{T_train[k]}")
    plt.show()

print(df[df[target_col] == 0])
testme(21)

if True:
    plt.hist(Y_dists.mean(), bins=30, alpha=0.5, label="Pred")  # range=(0, 10)
    plt.hist(T_train, bins=30, alpha=0.5, label="True")  # range=(0, 10)
    plt.legend()
    plt.show()
