from functools import reduce
import numpy as np
import pandas as pd

from flaubert.utils import zahlen_checks


def gibts_ungueltige_werte_in_series(xs, werte_liste):
    """Beispiel: gibts_ungueltige_werte_in_series([nan, 1, 2, 3, 9999], [9999]) hat 2 ungültige Werte"""
    return zahlen_checks.checkNone(xs) \
           or zahlen_checks.checkNaN(xs) \
           or zahlen_checks.checkInf(xs) \
           or (True in [x in werte_liste for x in xs])


def ist_ungueltig(x, liste_werte):
    return zahlen_checks.checkNone([x]) \
           or zahlen_checks.checkNaN([x]) \
           or zahlen_checks.checkInf([x]) \
           or x in liste_werte


# Anzahl oder Prozente der Nullen in einer Spalte
def nullen(df, xcol): return np.sum([ist_ungueltig(x, (np.nan, None)) for x in df[xcol].values])


def prozent_nullen(df, xcol): return (nullen(df, xcol) / df.shape[0]) if df.shape[0] != 0 else 0


def nullen_in_df(xdf): return pd.DataFrame([(xcol, nullen(xdf, xcol)) for xcol in xdf.columns])


def prozent_nullen_in_df(xdf): return pd.DataFrame([(xcol, prozent_nullen(xdf, xcol)) for xcol in xdf.columns])


# Ergibt die Durchschnittswerte einer kontinuierlicher Variable pro Wert aus einer Kategorialer Variable
def durchschnittswerte_pro_kateg(df, xkateg, xwert):
    return df[[xkateg, xwert]].groupby(xkateg).mean().sort_values(by=xwert)


def get_liste_features_mit_nullen(df, num_cols, target_col):
    """ Gegeben ein DataFrame und eine Liste num_cols in df.columns
        ergibt die Liste der Features mit Nullen
        num_cols und target_cols dürfen None sein
    """
    _df = df.copy()
    res_cols = []
    _df = _df[num_cols] if num_cols is not None else _df
    if num_cols is None:
        features_mit_nullen_df = pd.DataFrame(
            [(feature, gibts_ungueltige_werte_in_series(_df[feature].values, (np.nan, None)))
             for feature in _df.columns])
        features_mit_nullen_df[1].astype(np.int32)
        if target_col is not None:
            features_mit_nullen_df = features_mit_nullen_df[features_mit_nullen_df[0] != target_col]
        features_mit_nullen_df = features_mit_nullen_df[features_mit_nullen_df[1] == 1]
        if features_mit_nullen_df.shape[0] == 0:
            print("[x] Datenbestand hat keine Nullen")
        else:
            print(features_mit_nullen_df)
            res_cols = features_mit_nullen_df[0].tolist()
    return res_cols


def kateg_werte_liste(xdf_input, xcol, sep=None):
    """ Ergibt die Liste der Werte in kategorialer Variable
        Bleibt der sep None, dann sind die Werte nichts anderes als ein set(xL)
    """
    xliste = list(map(lambda x: str(x), xdf_input[xcol].values.tolist()))
    xliste = list(filter(lambda x: x is not np.nan, xliste))
    if sep is not None:
        xliste = list(map(lambda x: list(set(x.split(sep))), xliste))
        xliste = list(reduce(lambda a, b: a + b, xliste))
    xliste = [str(t).strip() for t in xliste]
    return xliste


def frequenz_werte(xdf_input, group_by_feature="CousinEducation",
                   prozente=False, sep=None, id_col="id"):
    """ Anzahl der Datensätze mit xcol == {Wert} ODER Wert in xcol.split(sep)
    """
    df = xdf_input.copy()
    xl = kateg_werte_liste(df, group_by_feature, sep=sep)
    xs = pd.DataFrame({group_by_feature: xl})
    xs[id_col] = xs.index.values.tolist()
    xres = xs.groupby(group_by_feature, as_index=True).count()
    xres = xres.sort_values(by=id_col, ascending=False)
    if prozente:
        xres[id_col] = [t / xdf_input.shape[0] for t in xres['id'].values.tolist()]
    return xres


def zeige_kateg_features(xdata, target_col, id_col):
    # Zeige die Situation der kategorialen Variablen, welche Werte steckt drin?
    cols_filter = list(filter(lambda x: x is not None, [target_col, id_col]))
    num_cols = list(filter(lambda x: x not in cols_filter, xdata.describe().columns))
    cat_cols = list(filter(lambda x: x not in num_cols and x not in cols_filter, xdata.columns))
    for xcol in cat_cols:
        print("\n==========================")
        print("COL: {}".format(xcol))
        xr = frequenz_werte(xdata, group_by_feature=xcol, prozente=False, sep=";")
        print(xr.index.values.tolist())


def zeige_korrelationen(xdf_input, target_col, id_col, threshold_corr=0.01):
    # Korrelationen stärker als threshold_corr (oder niedriger als -threshold_corr)
    cols_filter = list(filter(lambda x: x != id_col, xdf_input.columns))
    xtemp_corr = xdf_input[cols_filter].corr()[target_col]
    xcorr2 = xtemp_corr[list(map(lambda x: np.abs(x) > threshold_corr, xtemp_corr.values))]
    xcorr2df = pd.DataFrame({
        'feature': xcorr2.index.values,
        f'Korr_{target_col}': xcorr2.values
    })
    dfcorr = xcorr2df
    dfcorr = dfcorr.sort_values(by=[f'Korr_{target_col}'], ascending=False)
    return dfcorr
