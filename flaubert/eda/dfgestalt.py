import numpy as np
import pandas as pd

from flaubert.eda import dbeschreiben


def melting(xdf_input, id_features, werte_features):
    """
    Quelle: https://pandas.pydata.org/docs/reference/api/pandas.melt.html
    Melting-Operation ist das Gegenteil von Pivot
    Gegeben ein Datenbestand der Form:
        df
               A  B  C
            0  a  1  2
            1  b  3  4
            2  c  5  6
    Dann resultiert durch Melt:
        pd.melt(df, id_features=['A'], werte_features=['B', 'C'])
           A variable  value
        0  a        B      1
        1  b        B      3
        2  c        B      5
        3  a        C      2
        4  b        C      4
        5  c        C      6
    Zweck: damit es klar ist was das Ding tut
    """
    xdf_input = xdf_input.copy()[id_features + werte_features]
    df_res = pd.melt(xdf_input, id_vars=id_features, value_vars=werte_features, ignore_index=False)
    return df_res


def pivot(xdf_input, index_feature, pivot_features, werte_features):
    xdf_input = xdf_input.copy()[[index_feature] + pivot_features + werte_features]
    df_res = xdf_input.pivot(index=index_feature, columns=pivot_features, values=werte_features)
    df_res[index_feature] = df_res.index.values
    df_res = df_res.reset_index(drop=True)
    return df_res


def num_cat_trennen_desc(xdata, target_col, id_col):
    """
        Gegeben DataFrame xdata und target_col und id_col, ergibt kategoriale und numerische Features
        Verwendet describe() um die Trennung durchzuühren
    :return:
    """
    # separat numerische und kategoriale Variablen
    cols_filter = list(filter(lambda x: x is not None, [target_col, id_col]))
    num_cols = list(filter(lambda x: x not in cols_filter, xdata.describe().columns))
    cat_cols = list(filter(lambda x: x not in num_cols and x not in cols_filter, xdata.columns))

    print("[x] Numerisch:")
    print(num_cols, "\n")

    print("[x] Kategorial (ordinal und nominal):")
    print(cat_cols, "\n")

    dfnum = xdata[num_cols + cols_filter]
    dfcat = xdata[cat_cols + cols_filter]

    return dfnum, dfcat, num_cols, cat_cols


def num_cat_trennen_dtypes(xdf_input):
    """ Spalte den Datenbestand vertikal in kategoriale und numerische Teile
        Verwendet select_dtypes()
        Beispiel:
            xdf_num, xdf_cat = dbeschreiben.num_cat(dfres)
    """
    xdf_cat = xdf_input.select_dtypes(include=['object']).copy()
    xdf_num = xdf_input[list(filter(lambda xc: xc not in xdf_cat.columns, xdf_input.columns))]
    print("\nKategoriale Felder: {}".format(xdf_cat.columns))
    print("\nNumerische Felder: {}".format(xdf_num.columns))
    return xdf_num, xdf_cat


def splitdata_train_test(features, targets, ptrain=0.8):
    """ Aufteilung Daten (features / targets) entsprechend ptrain; ptrain: 0.8 """
    kSize = int(ptrain * features.shape[0])
    kids = list(range(0, targets.shape[0]))
    np.random.shuffle(kids)
    train_features = features[kids[:kSize], :]
    train_targets = targets[kids[:kSize]]
    test_features = features[kids[kSize:], :]
    test_targets = targets[kids[kSize:]]
    return train_features, train_targets, test_features, test_targets


def datenbestand_spalten(xdf_input, xcol, all_nan_stats=False, verbose=False):
    """ Entsprechend den Nullwerte in einem Feature xcol,
        spalte den Datenbestand in 2 Gruppen:
        xnullwerte_a und xnullwerte_b
    """
    xdf_temp = xdf_input.copy()
    xdf_temp[xcol + "_istNull"] = [np.isnan(t) if 'float' in str(type(t)) else False for t in
                                   xdf_temp[xcol].values.tolist()]
    if verbose:
        print(xdf_temp[[xcol, xcol + "_istNull"]])
        print(type(xdf_temp[xcol + "_istNull"].values[3]), xdf_temp[xcol + "_istNull"].values[3] is False)

    xdf_vorhanden = xdf_temp[list(map(lambda x: bool(x) is False, xdf_temp[xcol + "_istNull"].values))]
    xdf_fehlend = xdf_temp[list(map(lambda x: bool(x) is True, xdf_temp[xcol + "_istNull"].values))]

    if verbose:
        print("[x] Unterschied zwischen den zwei Gruppen:")
        print("\n Gruppe ohne fehlenden Daten (Gruppe A) in {}".format(xcol))
        print(xdf_vorhanden.describe())
        print("\n Gruppe mit fehlenden Daten (Gruppe B) in {}".format(xcol))
        print(xdf_fehlend.describe())
        print("[x] Prozent der Nullwerte pro Feature:")

    xcols_cluster = [xcol] if not all_nan_stats else xdf_vorhanden.describe().columns.tolist()
    xnullwerte_a = [(xc, dbeschreiben.prozent_nullen(xdf_vorhanden, xc)) for xc in xcols_cluster]
    xnullwerte_a = pd.Series(list(map(lambda x: x[1], xnullwerte_a)),
                             index=list(map(lambda x: x[0], xnullwerte_a))).to_dict()
    xnullwerte_b = [(xc, dbeschreiben.prozent_nullen(xdf_fehlend, xc)) for xc in xcols_cluster]
    xnullwerte_b = pd.Series(list(map(lambda x: x[1], xnullwerte_b)),
                             index=list(map(lambda x: x[0], xnullwerte_b))).to_dict()

    if verbose:
        print("\n Gruppe A:")
        print(xnullwerte_a)
        print("\n Gruppe B:")
        print(xnullwerte_b)

    return xdf_vorhanden, xdf_fehlend, xnullwerte_a, xnullwerte_b


def encode_label(xsLabelFeature, useOrdered=None):
    if useOrdered:
        xlabel_basis = useOrdered
    else:
        xlabel_basis = sorted(set(xsLabelFeature))  # ["kateg1", "kateg2", "kateg3"]
    xlabels = np.array([xlabel_basis.index(x) for x in xsLabelFeature])
    return xlabels, xlabel_basis


def addiere_umwandlung_numfeld2kateg(xdfInput, xcolNum="target", anzahlBins=4):
    """ umwandelt eine numerische Variable in Bins """
    from scipy import stats
    xdfInput = xdfInput.copy()
    bin_means, bin_edges, binnumber = stats.binned_statistic(xdfInput[xcolNum], xdfInput[xcolNum],
                                                             statistic='mean',
                                                             bins=anzahlBins)
    xdfInput[xcolNum + "_bin"] = [str(x) for x in binnumber]
    return xdfInput


def split_data_groups(data_input, feature_liste, target_col, lambdafunk, use_ordered=None, balanced_N=None):
    """ Datenbestand data_input wird anhand lambda Funktion lambdafunk aufgeteilt
        Ein Balance-Faktor sorgt dafür, dass in Situationen mit hohen Unterschieden (m:n, mit m>>n)
        die Gruppe die größer ist auch mehr Daten aus dem Pool bekommt.
    """
    data = data_input.copy()
    data["group"] = [lambdafunk(x) for x in data[target_col]]
    if balanced_N is not None:
        print("\n", data.groupby("group")[target_col].describe(), "\n")
        xL = []
        for xg in set(data.group.values):
            if balanced_N[xg] is not None:
                xL.append(data[data.group == xg].sample(balanced_N[xg]))
            else:
                xL.append(data[data.group == xg])
        data = pd.concat(xL)
        data = data.reset_index(drop=True)
    dtFrame = data[[x for x in feature_liste if x != target_col]]
    print(data.sample(10))
    print("xcolTarget:", target_col)
    print("[x] Gruppen:", set(data["group"].values))
    X = dtFrame.values
    y = [lambdafunk(x) for x in data[target_col].values]
    xlabels, xlabels_basis = encode_label(y, useOrdered=use_ordered)
    print(list(zip(xlabels_basis, list(range(len(xlabels_basis))))))
    return X, y, dtFrame, xlabels, xlabels_basis


def zeige_fortschritt(idx, maxlen):
    if idx % int(0.1 * maxlen) == 0:
        print("[x] Fortschritt: ", str(round(100. * (idx + 0.0) / maxlen, 2)) + "%")
        return True
    return False


def dataframe_spaltung_zum_X_Y(xdfInput, xcolID=None, xcolTarget=None):
    """ Spaltet einen Datenbestand in X (Training-Features) und Zielvariable """
    xdf = xdfInput.copy()
    if xcolID is not None:
        xdf = xdf[[x for x in xdf.columns if x != xcolID]]
    xdf = xdf[[x is not None for x in xdf[xcolTarget].values]]
    xfeatures = [x for x in xdf.columns if x != xcolTarget]
    X = xdf[xfeatures].values
    y = xdf[xcolTarget].values
    return X, y, xfeatures


def targetFeatureSplit(data, posLabel):
    """
        return targets and features as separate lists
        posLabel == 1 :: Ende der Liste
        posLabel == 0 :: Anfang der Liste
    """
    target = []
    features = []
    for item in data:
        if posLabel == 0:
            target.append(item[0])
            features.append(item[1:])
        elif posLabel == 1:
            target.append(item[-1])
            features.append(item[:-1])
    return target, np.array(features)
