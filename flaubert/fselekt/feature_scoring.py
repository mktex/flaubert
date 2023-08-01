from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np

from flaubert.eda import dfgestalt


def result_dataframe(xfeatures, xres, sig=None, qnt=None, prfx=None, xrichtung=1, doexpand=True):
    """
        xrichtung: 1 scoreNamen oder pwert "<" Threshold
        xrichtung: -1 scoreNamen oder pwert ">" Threshold
    """
    xL = list(zip(xfeatures, xres))
    if doexpand:
        xL = [[x[0], round(x[1][0], 4), round(x[1][1], 4)] for x in xL]
    xLDF = pd.DataFrame(xL)
    scoreNamen = "gewichtung_" + prfx
    xLDF.columns = ["feature", scoreNamen] if qnt is not None else ["feature", "gewichtung_" + prfx, "pwert_" + prfx]
    xLDF = xLDF.sort_values(by=scoreNamen, ascending=False)
    xLDF = xLDF.reset_index(drop=True)
    print(xLDF)
    print(xLDF.describe())
    if qnt is not None:
        if xrichtung == 1:
            print(f"\n[x] Entfernt alle die unter Quantile (qnt={qnt}) liegen:")
            selekt_out = xLDF[xLDF[scoreNamen] < xLDF[scoreNamen].quantile(qnt)].feature.values.tolist()
        elif xrichtung == -1:
            print(f"\n[x] Entfernt alle die über Quantile (qnt={qnt}) liegen:")
            selekt_out = xLDF[xLDF[scoreNamen] > xLDF[scoreNamen].quantile(qnt)].feature.values.tolist()
        else:
            print("[x] Unbekannter Wert in xrichtung")
            return
    elif sig is not None:
        if xrichtung == 1:
            print(f"\n[x] Entfernt alle die über den Signifiganzniveau ({sig}) liegen:")
            selekt_out = xLDF[xLDF["pwert_" + prfx] < sig].feature.values.tolist()
        elif xrichtung == -1:
            print(f"\n[x] Entfernt alle die unter den Signifiganzniveau ({sig}) liegen:")
            selekt_out = xLDF[xLDF["pwert_" + prfx] > sig].feature.values.tolist()
        else:
            print("[x] Unbekannter Wert in xrichtung")
            return
    selekt_in = [x for x in xLDF.feature.values if x not in selekt_out]
    print("\n[x] OUT:", selekt_out)
    print("\n[x] IN:", selekt_in, "\n")
    return selekt_in, selekt_out, xLDF


def gewichtungen_mutual_info_classif(xdfInput, xcolID=None, xcolTarget=None, qnt=0.25):
    """
    Quelle: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif
    Beispiel:
        selekt_in, selekt_out, xLDF = gewichtungen_mutual_info_classif(xdotrain, xcolID="object_id", xcolTarget="target")
    """
    X, y, xfeatures = dfgestalt.dataframe_spaltung_zum_X_Y(xdfInput, xcolID, xcolTarget)
    xres = mutual_info_classif(X, y)
    return result_dataframe(xfeatures, xres, sig=None, qnt=qnt, prfx="mi", xrichtung=1, doexpand=False)


def gewichtungen_f_classif(xdfInput, xcolID=None, xcolTarget=None, sig=0.1):
    """
    Quelle: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif
    Beispiel:
        selekt_in, selekt_out, xLDF = gewichtungen_f_classif(xdotrain, xcolID="object_id", xcolTarget="target", sig=0.1)
    """
    X, y, xfeatures = dfgestalt.dataframe_spaltung_zum_X_Y(xdfInput, xcolID, xcolTarget)
    F, p = f_classif(X, y)
    xres = list(zip(F, p))
    return result_dataframe(xfeatures, xres, sig=sig, qnt=None, prfx="fscore", xrichtung=1, doexpand=True)


def gewichtungen_chi2(xdfInput, xcolID=None, xcolTarget=None, sig=0.1):
    """
    Quelle: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
    Beispiel:
        selekt_in, selekt_out, xLDF = gewichtungen_chi2(xdotrain, xcolID="object_id", xcolTarget="target", sig=0.1)
    """
    xdf = xdfInput.copy()
    for xcol in [x for x in xdf.columns if x not in [xcolID, xcolTarget]]:
        xm = xdf[xcol].min()
        if xm < 0:
            xdf[xcol] = [x + np.abs(xm) for x in xdf[xcol].values]
    X, y, xfeatures = dfgestalt.dataframe_spaltung_zum_X_Y(xdf, xcolID, xcolTarget)
    Chi2, p = chi2(X, y)
    xres = list(zip(Chi2, p))
    return result_dataframe(xfeatures, xres, sig=sig, qnt=None, prfx="chi2", xrichtung=1, doexpand=True)


def features_classification(xdfInput, xcolID=None, xcolTarget=None, sig=0.1, qnt=0.25):
    """ Klassifiziert alle Features vs Zielvariable anhand Scores wie MI, Chi2 """
    selekt_in_mi, selekt_out_mi, xLDF_mi = gewichtungen_mutual_info_classif(xdfInput, xcolID, xcolTarget, qnt)
    selekt_in_fscore, selekt_out_fscore, xLDF_fscore = gewichtungen_f_classif(xdfInput, xcolID, xcolTarget, sig)
    selekt_in_chi2, selekt_out_chi2, xLDF_chi2 = gewichtungen_chi2(xdfInput, xcolID, xcolTarget, sig)
    xLDF = xLDF_mi.merge(xLDF_fscore, left_on="feature", right_on="feature")
    xLDF = xLDF.merge(xLDF_chi2, left_on="feature", right_on="feature")
    scores = ["mi", "fscore", "chi2"]
    # Achtung: das ist nur dann korrekt, wenn höhere Werte besser sind als niedrigen Werte.
    for xcol in scores:
        xLDF = xLDF.sort_values(by="gewichtung_" + xcol, ascending=True)
        xLDF = xLDF.reset_index(drop=True)
        xLDF[xcol] = list(range(xLDF.shape[0]))
    xLDF["cscore"] = [np.sum(x) / (3. * xLDF.shape[0]) for x in xLDF[scores].values]
    xLDF = xLDF.sort_values(by="cscore")
    xLDF = xLDF.reset_index(drop=True)
    return xLDF


def features_reduzierung(xdotrain, xcolTarget, xcolID,
                         keep_n_features=5,
                         doPlot=True):
    """ Reduziert ein Featureset mit scores """
    xdotrainS = xdotrain[[x not in [] for x in xdotrain[xcolTarget].values]].copy()
    xLDF = features_classification(xdotrainS, xcolID=xcolID, xcolTarget=xcolTarget, sig=0.2, qnt=0.05)
    xLDF = xLDF.dropna()
    print(xLDF)
    xcolsSelekt = xLDF.feature.values[-keep_n_features:].tolist()
    xdotrainS = xdotrainS[[xcolID] + pd.Series(xcolsSelekt).tolist() + [xcolTarget]]
    if doPlot:
        from flaubert.vis import xdiagramme as xdg
        xdg.show_corrplot(xdotrainS[xcolsSelekt + [xcolTarget]],
                          zeigeUeberschriften=True)
    print("[x] Ausgewählte Features:", xcolsSelekt)
    print(set(xdotrainS[xcolTarget].values))
    return xdotrainS
