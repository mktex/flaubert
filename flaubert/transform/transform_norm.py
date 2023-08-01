
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import skew
from scipy.stats import boxcox
from scipy.special import inv_boxcox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalisierung_mehrdimensional(_X, _xmin=None, _xmax=None, a=0, b=1):
    """ Führt eine Min-Max normalisierung in Intervall [a, b]
        Xnorm, xmin, xmax = tnorm.normalisierung_mehrdimensional(X, None, None, a=0, b=1)
        xmin, xmax sind Min und Max Werte per Feature in X.
    """
    X = _X.copy()
    X_mins = _xmin.copy() if _xmin is not None else np.min(X, axis=0)
    X_max = _xmax.copy() if _xmax is not None else np.max(X, axis=0)
    X_mins = np.repeat(X_mins.reshape(-1, 1), X.shape[0], axis=1).transpose()
    X_max = np.repeat(X_max.reshape(-1, 1), X.shape[0], axis=1).transpose()
    X_norm = (b - a) * (X - X_mins) / (X_max - X_mins) + a
    xmins = X_mins[0, :]
    xmax = X_max[0, :]
    if _xmin is not None and _xmax is not None:
        return X_norm
    else:
        return X_norm, xmins, xmax


def normalisierung_mehrdimensional_inv(_Xnorm, _xmin, _xmax, a=0, b=1):
    assert _xmin.shape == (_Xnorm.shape[1],)
    assert _xmax.shape == (_Xnorm.shape[1],)
    Xnorm = _Xnorm.copy()
    xmin = _xmin.copy()
    xmax = _xmax.copy()
    X_mins = np.repeat(xmin.reshape(-1, 1), Xnorm.shape[0], axis=1).transpose()
    X_max = np.repeat(xmax.reshape(-1, 1), Xnorm.shape[0], axis=1).transpose()
    X = (Xnorm - a) * (X_max - X_mins) / (b - a) + X_mins
    return X


def skalierung(xs, intervall=(0, 1), min_max_intervall=None, return_paramset=False):
    """ skaliert Series xs im neuen Intervall
        intervall: (untere_grenze, obere_grenze)
    """
    if min_max_intervall is None:
        min_max_intervall = (xs.min(), xs.max())
    if not return_paramset:
        return np.interp(xs, min_max_intervall, intervall)
    else:
        paramset_skalierung_dict = {
            "min_max_original_werte": min_max_intervall,
            "interpolationsintervall": intervall
        }
        return np.interp(xs, min_max_intervall, intervall), paramset_skalierung_dict


def skalierung_inv(xs, paramset_skalierung_dict):
    """ skaliert Series xs im neuen Intervall
        intervall: (untere_grenze, obere_grenze)
    """
    intervall = paramset_skalierung_dict['interpolationsintervall']
    min_max_original_werte = paramset_skalierung_dict['min_max_original_werte']
    return np.interp(xs, intervall, min_max_original_werte)


def skalierung_paramdict(xs, paramset_skalierung_dict):
    """ skaliert Series xs im neuen Intervall gegeben in parameter Dict Objekt
        intervall: (untere_grenze, obere_grenze)
    """
    min_max_original_werte = paramset_skalierung_dict['min_max_original_werte']
    intervall = paramset_skalierung_dict['interpolationsintervall']
    return np.interp(xs, min_max_original_werte, intervall)


"""
def min_max_normalisierung(xs, in_intervall=None):
    feature_werte = xs.copy()
    if in_intervall is not None:
        xmin, xmax = xs.min(), xs.max()
        MIN, MAX = in_intervall
        feature_werte = [((MAX - MIN) * (t - xmin) / (xmax - xmin) + MIN) for t in feature_werte]
    return feature_werte

def min_max_normalisierung_inv(xs, ):
    MIN, MAX = in_intervall
    minmax_xmin, minmax_xmax = xs.min(), xs.max()
    feature_werte = pd.Series([(( (x - MIN) * (minmax_xmax - minmax_xmin) ) / (MAX - MIN) + minmax_xmin)
                               for x in feature_werte
                               ])
"""


def xfuncISTNormal(xs, doPrint=False):
    if xs.shape[0] <= 3:
        if doPrint: print("[x] WARNUNG: Anfrage xfuncISTNormal() mit zu wenigen (<3) Werte!")
        return 0.0, 0.0, 0.0
    x1, x2, x3 = (skew(xs), shapiro(xs)[1], kstest(xs, 'norm').pvalue)
    if doPrint:
        print("[x] Heteroskedastizität:", x1)
        print("[x] Shapiro-Test Normalität:", x2)
        print("[x] KS-Test Normalität:", x3)
    return x1, x2, x3


def transform_serie_normal_by_boxcox(xs):
    """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html#scipy.stats.boxcox
    """
    from scipy import stats
    xtemp, xlambda = stats.boxcox(xs.values)
    xtemp = pd.Series(xtemp)
    xtemp.index = xs.index
    return xtemp, xlambda


def transform_serie_normal_by_boxcox_inv(xs, xlambda):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html#scipy.stats.boxcox
    Definition
    y =
        :: (x ** lmbda - 1) / lmbda, for lmbda > 0
        :: log(x), for lmbda = 0
        => x : np.power(x' * lmbda + 1, 1.0/lambda)
    """
    from scipy.stats import boxcox
    from scipy.special import inv_boxcox
    xretS = pd.Series(inv_boxcox(xs, xlambda))
    xretS.index = xs.index
    return xretS


def transform_serie_normal_by_log(xs, returnParams=False):
    """
        Beispiel Anwendung:
            xtemp = transform_serie_normal_by_log(xs)
    """
    xtemp = pd.Series([np.log(x) for x in xs.values])
    xtemp.index = xs.index
    return xtemp


def transform_serie_normal_by_log_inv(xtemp):
    """
        xfINV = lambda x: transform_serie_normal_by_log_inv([x])
    """
    xfINV = lambda x: np.exp(x)
    xretS = pd.Series([xfINV(x) for x in xtemp])
    xretS.index = xtemp.index
    return xretS


def transform_serie(xs, returnParams=False, verwendeTransform="boxcox", verwendeFinalTransformLog=False):
    """
        xs: ein Pandas Series
        Transformationskette:
            Normalisierung(X) -> BoxCox(X) -> Standardisierung
        Beispiel Anwendung:
            xdictparams = tnorm.transform_serie_normal(xs, returnParams=True, verwendeTransform="boxcox", verwendeFinalTransformLog=False)
            # xtemp, xresNorm, xs_min, xs_max, xlambda, xtempMean, xtempStd
        verwendeTransform: boxcox oder log
    """
    print(xs)
    xtemp = (xs - xs.min() + 0.0) / (xs.max() - xs.min() + 0.0)  # Normalisierung
    if verwendeTransform == "boxcox":
        xtemp, xlambda = boxcox(xtemp + 1.0)  # BoxCox
    elif verwendeTransform == "log":
        xtemp = transform_serie_normal_by_log(xtemp)
        xlambda = None
    else:
        print("[x] unbekannter Wert in Parameter verwendeTransform!")
        return None
    xtemp = pd.Series(xtemp)
    xtemp.index = xs.index
    xtempMean = xtemp.mean()
    xtempStd = xtemp.std()
    xtemp = (xtemp - xtempMean) / xtempStd  # Standardisierung
    if verwendeFinalTransformLog:
        xLogFinMin = np.abs(xtemp.min()) + 0.0001
        xtemp = transform_serie_normal_by_log(xtemp + xLogFinMin)
    else:
        xLogFinMin = 0.0
    if verwendeTransform == "boxcox":
        xformel = "Y = (BoxCox((X - xsmin) / (xsmax - xsmin) + 1.0, xlambda) - xtempMean) / xtempStd \n"
    elif verwendeTransform == "log":
        xformel = "Y = Log((X - xsmin) / (xsmax - xsmin), xlambda) - xtempMean) \n"
    if verwendeFinalTransformLog:
        xformel += "Z = Log(Y + ABS(Min(Y)) + 0.0001)"
    if returnParams == False:
        return xtemp
    xDictParams = {
        "xtemp": xtemp,
        "istNormal": xfuncISTNormal(xtemp),
        "xsmin": xs.min() + 0.0,
        "xsmax": xs.max() + 0.0,
        "xlambda": xlambda,
        "xtempMean": xtempMean,
        "xtempStd": xtempStd,
        "xLogFinMin": xLogFinMin,
        "formel": xformel
    }
    return xDictParams


def apply_transform(xs, xdictParams, debug=False):
    """ wende eine Transformation auf ein Series anhand eines Dict
        Beispiel Dict:
        {
        'x8': { 'formel': 'Y = (BoxCox((X - xsmin) / (xsmax - xsmin) + 1.0, xlambda) - '
                'xtempMean) / xtempStd \n',
                'xLogFinMin': 0.0,
                'xlambda': 1.288098681356535,
                'xsmax': 2.966287515417295,
                'xsmin': -2.8937988118283187,
                'xtempMean': 0.5637171359576898,
                'xtempStd': 0.10520829203612445}
        }
    """

    def printout(xs, debug, msg):
        if debug:
            print(msg)
            print(xs)

    xtemp = xs.copy()
    try:
        xsmin = xdictParams["xsmin"] + 0.0
        xsmax = xdictParams["xsmax"] + 0.0
        verwendeTransform = "boxcox" if xdictParams["xlambda"] is not None else "log"
        xlambda = xdictParams["xlambda"]
        verwendeFinalTransformLog = True if xdictParams["xLogFinMin"] != 0 else False
        xLogFinMin = xdictParams["xLogFinMin"]
        xtempMean = xdictParams["xtempMean"]
        xtempStd = xdictParams["xtempStd"]
        if xtempStd == 0:
            print("[x] WARNUNG: apply_transform() hat std == 0; es wird zu NaN Werte führen!")
        xtemp = pd.Series((xs - xsmin) / (xsmax - xsmin))
        printout(xtemp, debug, "[x] A. normalisiert:")
        if verwendeTransform == "boxcox":
            if xlambda != 0:
                xtemp = (np.power(xtemp.values + 1.0, xlambda) - 1) / xlambda  # boxcox(xtemp.values + 1.0, xlambda)
            else:
                xtemp = np.log(xtemp),  # boxcox(xtemp.values + 1.0, xlambda)
        elif verwendeTransform == "log":
            xtemp = transform_serie_normal_by_log(xtemp)
        printout(xtemp, debug, "[x] B. BoxCox/Log:")
        xtemp = pd.Series((xtemp - xtempMean) / xtempStd)
        printout(xtemp, debug, "[x] C. standardisiert:")
        if verwendeFinalTransformLog:
            xtemp = transform_serie_normal_by_log(xtemp + xLogFinMin)
    except:
        import traceback
        traceback.print_exc()
    return xtemp


def transform_serie_inv(xtempIN, xsMin, xsMax, xlambda, xtempMean, xtempStd,
                        xLogFinMin=0.0, verwendeTransform="boxcox",
                        verwendeFinalTransformLog=False):
    """
        xfINV = lambda x: transform_serie_normal_inv([x], xsMin, xsMax, xlambda, xtempMean, xtempStd, "boxcox") # oder "log")
    """
    xtemp = xtempIN.copy()
    xIndx = xtemp.index
    if verwendeFinalTransformLog:
        xtemp = transform_serie_normal_by_log_inv(xtemp) - xLogFinMin
    xtemp = xtemp * xtempStd + xtempMean
    if verwendeTransform == "boxcox":
        xtemp = inv_boxcox(xtemp, xlambda)
    elif verwendeTransform == "log":
        xtemp = [np.exp(x) for x in xtemp.values]
    else:
        print("[x] unbekannter Wert in Parameter verwendeTransform!")
        return
    xtemp = (xtemp - 1.0) * (xsMax - xsMin) + xsMin
    xtemp = pd.Series(xtemp)
    xtemp.index = xIndx
    return xtemp


def zeige_transform_effekt_I(xdfIN, xcolName="acm_wsk_kreinigung", doLogTransformFinal=False):
    """
        Visualsierung der Resultat der Transformation mithilfe der Kette Normalisierung -> BoxCox -> Standardisierung
    """
    xdf = xdfIN.copy()
    xs = xdf[xcolName]
    print(xs.describe())
    xDictParams = transform_serie(xs, returnParams=True, verwendeTransform="boxcox",
                                  verwendeFinalTransformLog=doLogTransformFinal)
    xret1 = xDictParams["xtemp"]  # _ISTNormal, xsmin, xsmax, xlambda, xtempMean, xtempStd, xLogFinMin

    print("[x] Stichprobe Resultat:")
    print(xret1.sample(10))
    plt.close()
    xfig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(xdf[xcolName], 100, alpha=0.5)
    ax[1].hist(xret1, 100, alpha=0.5)
    ax[0].set_title("(hist) %s (vor)" % xcolName)
    ax[1].set_title("(hist) %s (nach)" % xcolName)
    plt.tight_layout()
    plt.show()


def zeige_transform_effekt_II(xdfIN, xcolName1="acm_wsk_kreinigung", xcolName2="acm_wsk_haushalt",
                              doTransformCols=(True, True),
                              doLogTransformFinal=(False, False)):
    """
        Gleich wie in zeige_transform_effekt_I(), nur für 2 Variablen. Dazu noch die resultierende Streudiagramm
    """
    xdf = xdfIN.copy()
    doLT1, doLT2 = doLogTransformFinal
    doT1, doT2 = doTransformCols
    print("===================================================")
    print("[x] Variable:", xcolName1)
    xs = xdf[xcolName1]
    print(xs.describe())
    xret1 = xs.copy()
    if doT1:
        xDictParams = transform_serie(xs, returnParams=True, verwendeTransform="boxcox",
                                      verwendeFinalTransformLog=doLT1)
        xret1 = xDictParams["xtemp"]  # _ISTNormal, xsmin, xsmax, xlambda, xtempMean, xtempStd, xLogFinMin
    print("[x] Stichprobe Resultat:")
    print(xret1.sample(10))
    print("===================================================")
    print("[x] Variable:", xcolName2)
    xs = xdf[xcolName2]
    print(xs.describe())
    xret2 = xs.copy()
    if doT2:
        xDictParams = transform_serie(xs, returnParams=True, verwendeTransform="boxcox",
                                      verwendeFinalTransformLog=doLT2)
        xret2 = xDictParams["xtemp"]  # , _ISTNormal, xsmin, xsmax, xlambda, xtempMean, xtempStd, xLogFinMin
    print("[x] Stichprobe Resultat:")
    print(xret2.sample(10))
    plt.close()
    xfig, ax = plt.subplots(3, 2, figsize=(10, 8))
    ax[0, 0].hist(xdf[xcolName1], 100, alpha=0.5)
    ax[0, 1].hist(xret1, 100, alpha=0.5)
    ax[1, 0].hist(xdf[xcolName2], 100, alpha=0.5)
    ax[1, 1].hist(xret2, 100, alpha=0.5)
    ax[2, 0].scatter(x=xdf[xcolName1], y=xdf[xcolName2], s=3, alpha=0.5)
    ax[2, 1].scatter(x=xret1, y=xret2, s=3, alpha=0.5)
    ax[0, 0].set_title("(hist) %s (vor)" % xcolName1)
    ax[0, 1].set_title("(hist) %s (nach)" % xcolName1)
    ax[1, 0].set_title("(hist) %s (vor)" % xcolName2)
    ax[1, 1].set_title("(hist) %s (nach)" % xcolName2)
    ax[2, 0].set_title("Streudiagramm (vor)")
    ax[2, 1].set_title("Streudiagramm (nach)")
    ax[2, 0].set_ylabel("%s" % xcolName2)
    ax[2, 0].set_xlabel("%s" % xcolName1)
    ax[2, 1].set_ylabel("%s" % xcolName2)
    ax[2, 1].set_xlabel("%s" % xcolName1)
    plt.tight_layout()
    plt.show()


def get_stat_minmax(X):
    return X.min(), X.max()


def normalize_zum_0_255(X):
    xmin, xmax = get_stat_minmax(X)
    return 255 * (X - xmin) / (xmax - xmin), xmin, xmax


def normalize_zum_0_255_inv(Xinv, xmin, xmax):
    return ((Xinv * (xmax - xmin)) / 255.) + xmin


def normalize_like_image(X):
    """
    Normalize the given dataset X
    Args:
        X: ndarray, dataset
    Returns:
        (Xbar, mean, std): tuple of ndarray, Xbar is the normalized dataset
        with mean 0 and standard deviation 1; mean and std are the
        mean and standard deviation respectively.
    Note:
        You will encounter dimensions where the standard deviation is
        zero, for those when you do normalization the normalized data
        will be NaN. Handle this by setting using `std = 1` for those
        dimensions when doing normalization.
    """
    xmubar = np.mean(X, axis=0)
    xstdbar = np.std(X, axis=0)
    std_filled = xstdbar.copy()
    std_filled[xstdbar == 0] = 1.
    Xbar = (X - xmubar) / std_filled
    Xbar, xminbar, xmaxbar = normalize_zum_0_255(Xbar)
    return Xbar, xmubar, xstdbar, xminbar, xmaxbar


def normalize_like_image_inv(XbarInv, xmubar, xstdbar, xminbar, xmaxbar):
    _XbarInv = normalize_zum_0_255_inv(XbarInv, xminbar, xmaxbar)
    return _XbarInv * xstdbar + xmubar
