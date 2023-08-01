import sys
import numpy as np
import pandas as pd
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("./flaubert/utils")

xfGetFeatures = lambda xcolTarget, xcolID, xdotrain: [x for x in xdotrain.columns.values if
                                                      x not in [xcolTarget, xcolID]]
xfGetSizeObj = lambda xObj: sys.getsizeof(xObj) / 1024 / 1024.
xfGetBuckets = lambda objekteIDs: list(cbook.pieces(np.array(list(range(len(objekteIDs)))), 3 * 9))


def removeUnnamed(xdf):
    xdf = xdf[[x for x in xdf.columns if "Unnamed" not in x]]
    return xdf


def applyRegelFilter(xdfInput, xListeInputOutlierByRegel):
    """ Wendet eine Regelliste auf xdfInput an
        Beispiel einer solcher Liste:
            xListeInputOutlierByRegel = [
                "KitchenAbvGr == 3.0",
                "BsmtFinSF1 > 4000",
                "OpenPorchSF > 400 and BedroomAbvGr == 5",
                "target_bin == 4"
            ]
    """

    def getOK(xrecord, xregel):
        xkeyList = list(xrecord.keys())
        for xkey in xkeyList:
            locals()[xkey] = xrecord[xkey]
        ok = eval(xregel)
        return ok

    xregelOKs = {}
    for xregel in xListeInputOutlierByRegel:
        okListe = [getOK(xdfInput.iloc[k].to_dict(), xregel) for k in range(xdfInput.shape[0])]
        xregelOKs[xregel] = okListe
    xL = []
    for xkey in list(xregelOKs.keys()):
        xL.append(xregelOKs[xkey])
    xLdf = pd.DataFrame(xL).transpose()

    regel_dict = xregelOKs
    regel_selekt = xLdf.values
    return regel_dict, regel_selekt


def filter_by_lambda_list(df, lambdafunk_liste, xcol):
    """
        Filter DataFrame aufgrund einer Liste Lambda-Funktionen für ein Feature
        Verwendet eine Liste Lambda Filters auf einem Feature
        xfFList = [lambda x: x != 0]
    """
    for xF in lambdafunk_liste:
        df = df[[xF(x) for x in df[xcol].values]]
    return df


def apply_lambda(xdfVis, xcolID, xcolTarget, xF):
    # Anwendung einer Lambda-Funktion für alle Spalten
    sFields = xdfVis.columns.values
    for xc in [x for x in sFields if x not in [xcolID, xcolTarget]]:
        if xc is not xcolTarget and xc is not xcolID:
            xdfVis[xc] = xF(xdfVis[xc])
    return xdfVis


def freq2period(w):
    if w != 0:
        return 2.0 * np.pi / w
    else:
        return 0.0


def period2freq(T):
    if T != 0:
        return 2.0 * np.pi / T
    else:
        return 0.0


def get_periodogram(xdf_input=None, wmin=2.0 * np.pi / 30, wmax=2.0 * np.pi / 30 * 6,
                    doplot=False, vollepackung=False):
    """
        nparray_in eine Zeitreihe mit t Zeitachse und y Werte
    """
    import scipy.signal as signal
    import numpy as np
    if xdf_input is None:
        t = 100 * np.random.rand(100)
        A = 1.
        w = 2 * np.pi
        phi = 0.
        y = A * np.sin(w * t + phi) + 0.1 * np.random.randn(100)
    else:
        if xdf_input.shape[1] == 2:
            t = xdf_input[:, 0].flatten()
            y = xdf_input[:, 1].flatten()
        else:
            t = np.arange(0, xdf_input.shape[0])
            y = xdf_input
    wList = np.linspace(wmin, wmax, 100)
    pgram = signal.lombscargle(t, y, wList, normalize=True)
    pgram = np.array([freq2period(w) for w in pgram])
    if doplot:
        pd.Series(pgram).plot()
        plt.title("Periodogram: Frequenz in Periode umgewandelt")
        plt.xlabel("Periode T")
        plt.grid()
        plt.show()

    if not vollepackung:
        return pgram
    else:
        return pgram, wList, t, y


def get_signal_minmax_locations(xsignal):
    """ Irgendwo gefunden.. sehr effiziente Identifizierung von Min-Max """
    doublediff = np.diff(np.sign(np.diff(xsignal)))
    peak_locations = np.where(doublediff == -2)[0] + 1
    doublediff2 = np.diff(np.sign(np.diff(-1 * xsignal)))
    through_locations = np.where(doublediff2 == -2)[0] + 1
    return peak_locations, through_locations


def interpolate_points(xl, nsize):
    """ Interpolation mehreren Punkte """
    from scipy import interpolate
    tck = interpolate.splrep(x=list(range(len(xl))), y=xl, k=1)
    ynew = interpolate.splev(np.arange(0, len(xl), 1.0 / nsize), tck, der=0)
    return ynew