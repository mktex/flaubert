import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from flaubert import einstellungen


def fit_exp(xpoints, ypoints, show_stuff=True):
    def func(x, a, b, c):
        return a * np.exp(- b * x) + c

    try:
        popt, pcov = curve_fit(func, xpoints, ypoints)
        if show_stuff: print('a = %s , b = %s, c = %s' % (popt[0], popt[1], popt[2]))
        return pd.Series([func(x, *popt) for x in xpoints])
    except:
        pass
    return pd.Series(ypoints)


def run(xs, return_iqr_params=False, ident_residuen=0, show_vis=False):
    """
        Implementierung der IQR Methode mit standardisierte Schwellenwerte
    """
    if len(set(xs)) <= 2:
        print("[x] Fehler: len(set(xs)) <= 2!")
        return pd.Series([]) if return_iqr_params is False else (pd.Series([]), None, None, None, None)
    if show_vis:
        print("<pre>")
    if 'Series' not in str(type(xs)):
        xs = pd.Series(xs)
    xs = xs[[not x for x in xs.isna().values]]
    if xs.shape[0] == 0 or len(set(xs)) <= 1:
        print("[x] Fehler: xs.shape[0] == 0 or len(set(xs)) <= 1")
        return pd.Series([]) if return_iqr_params is False else (pd.Series([]), None, None, None, None)
    xtemp = xs
    Q1 = xtemp.quantile(0.25)
    Q3 = xtemp.quantile(0.75)
    k = 0.01
    while Q1 == Q3 and (Q1 >= 0 or Q3 <= 1) and k < 0.25:
        Q1 = xtemp.quantile(0.25 - k)
        Q3 = xtemp.quantile(0.75 + k)
        k += 0.01
    iqr_wert = Q3 - Q1
    if iqr_wert < 0.001:
        print("[x] Fehler: iqr_wert < 0.001")
        return pd.Series([]) if return_iqr_params is False else (pd.Series([]), None, None, None, None)
    if show_vis:
        print('Q1=%.2f;' % Q1, 'Q3=%.2f; ' % Q3, 'IQR=%.2f; ' % iqr_wert)
    xtemp = xtemp.tolist()
    funk_selekt = lambda ik: xs[[x <= Q1 - ik * iqr_wert or x >= Q3 + ik * iqr_wert for x in xtemp]]
    xp = 0.1
    xrange_liste = np.arange(1.0, 25.0, 0.05)
    xres_iqr = [len(funk_selekt(x)) for x in xrange_liste]
    try:
        fitted_exp_list = fit_exp(xrange_liste, xres_iqr, show_stuff=show_vis)
        tck = interpolate.splrep(xrange_liste, fitted_exp_list, k=2)
        xres = pd.DataFrame(interpolate.spalde(xrange_liste, tck))
        xinklination = np.abs(np.degrees([np.arctan(x) for x in xres[1].values]))
    except:
        import traceback
        traceback.print_exc()
        xinklination = []
    xgI = [x for x in xinklination if x != 0 and x <= 60]
    if len(xgI) != 0:
        gesuchte_inklination = max(xgI)
        if show_vis:
            print('[x] Inklination der Ableitungskurve: ')
            print(xinklination)
            print('[x] Gesuchte Inklination: ', gesuchte_inklination)
        x_iqr_parameter = xrange_liste[xinklination.tolist().index(gesuchte_inklination)]
    else:
        if show_vis:
            print('[x] Anpassung IQR fehlgeschlagen, Verwendung standard Wert 1.5')
        x_iqr_parameter = 1.5
    if show_vis:
        print('[x] IQR Parameter: ', x_iqr_parameter)
        print("</pre>")
    xlist = funk_selekt(x_iqr_parameter)
    if ident_residuen == -1:
        xres = xlist[xlist < 0]
    if ident_residuen == 1:
        xres = xlist[xlist > 0]
    if ident_residuen == 0:
        xres = xlist
    if show_vis:
        fig = plt.figure(figsize=einstellungen.FIGSIZE_BREIT)
        ax = fig.add_subplot(111)
        ax.scatter(x=xrange_liste, y=xres_iqr, s=3)
        ax.plot(xrange_liste, fitted_exp_list, color='red')
        ax.set_title('Optimierung IQR: [%.2f - x * %.2f, %.2f + x * %.2f]' % (
            Q1, x_iqr_parameter, Q3, x_iqr_parameter))
        ax.set_xlabel('x')
        ax.set_ylabel('N (Anzahl markiert als Ausreisser)')
        plt.show()
    if not return_iqr_params:
        return xres
    else:
        return xres, pd.DataFrame({'x': xrange_liste, 'y': xres_iqr}), Q1, Q3, x_iqr_parameter
