import numpy as np
import pandas as pd
from sklearn import cluster

from flaubert.eda import dbeschreiben, dfgestalt
from flaubert.utils import signal_verarbeitung as signaldv
from flaubert.model.neural_networks import xnn
from flaubert.statistik import stat_op
from flaubert.utils import zahlen_checks

ersatz_nullwerte_durch_mean = lambda xdf_in: xdf_in.fillna(xdf_in.mean().iloc[0])
ersatz_nullwerte_durch_median = lambda xdf_in: xdf_in.fillna(xdf_in.median().iloc[0])
ersatz_nullwerte_durch_mode = lambda xdf_in: xdf_in.fillna(xdf_in.mode().iloc[0])


def ersetze_alle_na_werte_mit_none(xdt):
    """ Falls die Situation vorkommt, dass None, NaN und Inf gemischt sind """
    print(" ... nan mit None ...")
    for feature in xdt.columns:
        xdt[feature] = list(map(lambda x:
                                None if dbeschreiben.ist_ungueltig(x, ()) else x,
                                xdt[feature].values
        ))
    print(" ... fertig ...")
    return xdt


def ersatz_mit_knn(xdf_input, xcol="CareerSatisfaction", ignoriere_spalten=("id",),
                   reduziere_cols_mit_nans=False, threshold_null_werte_cluster=0.7, nclust=3,
                   verbose=False, ersatz_nans_mit_none=False):
    """ Für jede Zeile, die NAN Werte hat, findet diejenigen Datensätze,
        die in einem Cluster zusammengehören
        threshold_null_werte_cluster:
            Feature für Clustering nicht verwenden, wenn die Nullwerte mehr als so viel Prozent betragen
        Achtung: es funktioniert nicht, wenn alle Werte None sind
    """
    print("=============================================================================================")
    print("[x] Feature:", xcol)
    ignoriere_spalten = list(filter(lambda x: x is not None, ignoriere_spalten))
    if ersatz_nans_mit_none: xdf_input = ersetze_alle_na_werte_mit_none(xdf_input.copy())
    xdf = xdf_input[list(filter(lambda x: x not in ignoriere_spalten, xdf_input.columns))].copy()
    if '__cluster__' in xdf.columns:
        if verbose: print("[x] Intern wird __cluster__ Feld verwendet, dieses Feld darf in Voraus nicht existieren!")
        return xdf_input, None

    xdf_vorhanden, xdf_fehlend, xnwa, xnwb = dfgestalt.datenbestand_spalten(xdf, xcol,
                                                                            all_nan_stats=reduziere_cols_mit_nans)
    if xdf_vorhanden.shape[0] == 0:
        if verbose: print("[x] WARNUNG: für den Feature |{}| kann die Ersatz-Mit-KNN Methode nicht funktionieren, "
                          "da alle Werte None sind!")
        return xdf_input, None

    # es sollten für clustering keine Features, die sehr viele NaNs haben
    xcols_cluster = xdf_vorhanden.describe().columns.tolist()
    xcols_cluster = list(filter(lambda x: x != xcol, xcols_cluster))
    if reduziere_cols_mit_nans:
        xcols_cluster = list(filter(lambda x:
                                    xnwa[x] <= threshold_null_werte_cluster and xnwb[x] <= threshold_null_werte_cluster,
                         xcols_cluster))
    np.random.seed()
    np.random.shuffle(xcols_cluster)

    xdf_vorhanden = ersatz_nullwerte_durch_mean(xdf_vorhanden[xcols_cluster])
    xdf_fehlend = ersatz_nullwerte_durch_mean(xdf_fehlend[xcols_cluster])

    print(".. cluster training ..")
    anzahl_cluster = nclust
    clf = cluster.KMeans(n_clusters=anzahl_cluster)  # random_state=42
    clf.fit(xdf_vorhanden[xcols_cluster].values)
    xdf_vorhanden['__cluster__'] = clf.labels_

    if verbose:
        print("\n[x] Beispiel Resultat:")
        print(xdf_vorhanden[xcols_cluster + ["__cluster__"]].sample(20))
    xdf_clustering_data = ersatz_nullwerte_durch_mean(xdf[xcols_cluster])
    xdf["__cluster__"] = clf.predict(xdf_clustering_data)

    if verbose:
        print("\n[x] Datensätze klassifiziert:")
        print(xdf[[xcol] + xcols_cluster + ['__cluster__']])
        print("\n[x] Frequenz der Datensätze im jeweiligen Cluster:")
        _ = dbeschreiben.frequenz_werte(xdf, group_by_feature="__cluster__", prozente=True)

    cluster_means = xdf.groupby("__cluster__").mean()[xcol].to_dict()
    if verbose:
        print("\n[x] Durchschnitte im Cluster: ", cluster_means)

    xgruppen = []
    for cl in range(anzahl_cluster):
        xgruppen.append(
            xdf[xdf["__cluster__"] == cl].fillna(value={xcol: cluster_means[cl]})
        )
    xdf_res = pd.concat(xgruppen)

    print("\n [x] Wenn alles gut gelaufen ist, haben sich die statistischen Maßen nicht wesentlich geändert:")
    print("\t VORHER:")
    print(xdf_input[xcol].describe())
    print("\n\t NACHHER:")
    print(xdf_res[xcol].describe())

    xdf_res = xdf_res[list(filter(lambda x: x != '__cluster__' and x != (xcol + "_istNull"), xdf_res.columns))]
    for xc in ignoriere_spalten:
        xdf_res[xc] = xdf_input[xc]

    return xdf_res, clf


def get_dataframe_filled(xdf_input, num_cols, id_col, target_col):
    """
        Ersatz mit KNN iterativ für fehlende Werte
    """
    _xdf_input = xdf_input.copy()
    ignoriere_spalten = list(filter(lambda x: x is not None, [id_col, target_col]))
    _xdf_input = ersetze_alle_na_werte_mit_none(_xdf_input)
    if num_cols is None:
        num_cols = dbeschreiben.get_liste_features_mit_nullen(_xdf_input, num_cols=None, target_col=target_col)
        if len(num_cols) == 0:
            return xdf_input
    xdt, _ = ersatz_mit_knn(_xdf_input, xcol=num_cols[0], ignoriere_spalten=ignoriere_spalten,
                            reduziere_cols_mit_nans=False, threshold_null_werte_cluster=0.7,
                            nclust=15, verbose=False, ersatz_nans_mit_none=False)
    if len(num_cols) > 1:
        for xc in filter(lambda x: x not in [id_col, target_col, num_cols[0]], num_cols):
            xdt, _ = ersatz_mit_knn(xdt, xcol=xc, ignoriere_spalten=ignoriere_spalten,
                                    reduziere_cols_mit_nans=False, threshold_null_werte_cluster=0.7,
                                    nclust=15, verbose=False, ersatz_nans_mit_none=False)
    return xdt


def setval(dfin, xcol, xval, xval_new):
    """
        Für kategoriale Werte die einer Zuordnung unterliegen, können numerische Werte assoziert werden
        eg "Masters" < "Doctorate"
    """
    for ik in range(dfin.shape[0]):
        xw = str(dfin.at[ik, xcol])
        if xw == xval:
            dfin.at[ik, xcol] = xval_new
    return dfin


"""
Ersatz von verschachtelten Listen mit String-Elemente wenn sep als Parameter vorhanden:
Beispiel:
    xs = "blah;blup;bleep"
    sep = ";"
    repl_strlist(xs, 'blup', 'blap')
    => 'blah;blap;bleep'
"""
stringlist = lambda xs, sep: list(map(lambda x: x.strip(), str(xs).split(sep)))
repl_strlist = lambda xstr_sep, xval, xval_new, sep: \
    sep.join(
        list(map(lambda x: x if x != xval else xval_new, stringlist(xstr_sep, sep)))
    )


def repl_df_dict(df, xcol, dict_repl, repl_strings=True, sep=";"):
    """ Gegeben ein Dict-Objekt für eine kategoriale Variable ersatz der Werte entsprechend Werte für Keys.
    df: data frame
    dict_repl: replacement dictionary
    repl_strings: whether the replacements values are numbers or strings
    """
    dfvalues = df[xcol].values.tolist()
    for xkey in dict_repl.keys():
        if repl_strings:
            dfvalues = [repl_strlist(t, xkey, dict_repl[xkey], sep) for t in dfvalues]
        else:
            dfvalues = [t if xkey != t else dict_repl[xkey] for t in dfvalues]
    return dfvalues


def setze_nullen_at_index(xdt, xcol, listIndeces):
    original_werte = []
    for xnum in listIndeces:
        original_werte.append(xdt.at[xnum, xcol])
        xdt.at[xnum, xcol] = None
    return xdt, list(zip(listIndeces, original_werte))


def setze_nullen_zuffall(xdt, xcol, n=10):
    xdt = xdt.copy()
    xdt = xdt.reset_index(drop=True)
    listIndeces = np.random.randint(0, xdt.shape[0] - 1, n)
    xdt, original_werte = setze_nullen_at_index(xdt, xcol, listIndeces)
    return xdt, original_werte


def generiere_modell_nn_numeric_fuer_feature(xdfInput, xclude=None,
                                             feature_target=None, xthreshold=0.1):
    """ Beispiel xclude: (xcolTarget, xcolID)
        feature_target wird gleich den Namen des Modells sein
    """
    from flaubert.model.neural_networks import xnn
    xdfInput = xdfInput.copy()
    xtemp = xdfInput[[x for x in xdfInput.columns if x not in xclude]].copy()
    xtemp, origwerte = setze_nullen_zuffall(xtemp, feature_target, n=10)
    xcolsKorr = stat_op.get_corrlist(xtemp, feature_target)
    xcolsKorr = xcolsKorr[[np.abs(x) > xthreshold for x in xcolsKorr.values]]
    print("[x] Korrelationen für", feature_target + ":")
    print(xcolsKorr)
    xcols = [x for x in xcolsKorr.index.values if x != feature_target]
    if len(xcols) == 0:
        print("[x] Fehler in Filter nach Korrelationen. Verblienene Liste Features ist 0!")
        return
    datenpaket = xnn.input_numeric_nn(xtemp[xcols + [feature_target]],
                             xcolTarget=feature_target, hidden_layer_sizes=(16, 16), verbose=True, aktivF="logistic",
                             skalierung=(0, 1), standardisierung=True, entferneKorr=False, thresholdKorr=None,
                             namedLearner=f"model_nn_{feature_target}")
    idmap, model_nn, transform_dict, model_namen = datenpaket
    return idmap, model_nn, transform_dict, model_namen


def ersatz_mit_nn(xdotrain, inputkette, xcolTarget, xcolID, mode="train"):
    """ Gegeben eine Kette von Features, baut NN Modelle aus, um iterativ Werte zu ersetzen
        Beispiel: INPUTKETTE = ["distmod", "hostgal_photoz", "hostgal_specz"]
        mode: train, apply
    """
    xdt = xdotrain.copy()
    print(xdt[inputkette].sample(10))
    if mode == "train":
        for kModell in range(len(inputkette)):
            feature_learner = inputkette[kModell]
            xcolsTrain = [x for x in xdotrain.columns if x not in inputkette[kModell:]] + [feature_learner]
            print("[x]", xcolsTrain)
            xdotrainNN = xdt[[not np.isnan(x) for x in xdt[feature_learner].values]][xcolsTrain].copy()
            generiere_modell_nn_numeric_fuer_feature(xdotrainNN,
                                                     xclude=(xcolTarget, xcolID),
                                                     feature_target=feature_learner, xthreshold=0.01)
            xdotrainNeu = xnn.apply_modell_nn_fuer_feature(xdt[xcolsTrain], target_col=feature_learner,
                                                           modelnamen=feature_learner, ersatz_in_df=True)
            xdt = xdt[[x for x in xdt.columns if x not in [feature_learner]]].merge(
                xdotrainNeu[[xcolID, feature_learner]], left_on=xcolID, right_on=xcolID)
        xdotrain = xdt
    elif mode == "apply":
        for kModell in range(len(inputkette)):
            feature_learner = inputkette[kModell]
            xdotrainNeu = xnn.apply_modell_nn_fuer_feature(xdotrain, target_col=feature_learner,
                                                           modelnamen=feature_learner, ersatz_in_df=True)
            xdt = xdt[[x for x in xdt.columns if x not in [feature_learner]]].merge(
                xdotrainNeu[[xcolID, feature_learner]], left_on=xcolID, right_on=xcolID)
        xdotrain = xdt
    return xdotrain
