
from importlib import reload
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection
from matplotlib import cm
import seaborn as sbn
import numpy as np
import pandas as pd
import time

from sklearn.metrics import r2_score, mean_squared_error

from flaubert import einstellungen
from flaubert.statistik import xcorrs


def update(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([12])


# brg, bone, copper, cool, plasma, cubehelix
COLOR_STR = "brg"
xfCOLOR = lambda xarg: plt.cm.brg(xarg)
plt.rcParams['font.size'] = 9

VERWENDE_DIR = None

REDUZIERE = False
MAXIMAL = 1250

MAX_BINS = 75
MAX_KATEGORIEN = 4
SCATTER_SPARAM = 28

jitter_on = (1, 1)  # jeweils für Koordinate x und y


def reduziere_dataframe(xdfin):
    if xdfin.shape[0] > MAXIMAL:
        # print(f"[x] WARNUNG: für Visualisierung wird das DataFrame reduziert, maximal {MAXIMAL}")
        return xdfin.sample(MAXIMAL)
    else:
        return xdfin


def rand_jitter(arr):
    _arr = [int(t) for t in arr]
    stdev = .015 * (np.max(_arr) - np.min(_arr))
    return np.array(_arr) + np.random.randn(len(_arr)) * stdev


def jitter(x, y, s=15, c='b', marker='o', cmap=None, norm=None,
           vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    global jitter_on
    return plt.scatter(rand_jitter(x) if jitter_on[0] == 1 else x,
                       rand_jitter(y) if jitter_on[1] == 1 else y,
                       s=s, c=c, marker=marker, cmap=cmap, norm=norm,
                       vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)


def jitter_feature(xdfinput, featurex, featurey):
    xdfinput = xdfinput.copy()
    try:
        if jitter_on[0] == 1:
            xdfinput[featurex] = rand_jitter(xdfinput[featurex])
        if jitter_on[1] == 1:
            xdfinput[featurey] = rand_jitter(xdfinput[featurey])
    except:
        import traceback
        traceback.print_exc()
        print(f"[x] FEHLER: Jittering für Features {featurex}, {featurey} fehlgeschlagen!")

    return xdfinput


def kateg_aus_numerischer_variable(xs):
    global MAX_BINS
    from scipy import stats
    if len(set(xs)) > MAX_KATEGORIEN:
        bin_means, bin_edges, binnumber = stats.binned_statistic(xs, xs, statistic='mean', bins=MAX_KATEGORIEN)
        return [str(x) for x in binnumber]
    else:
        return [str(x) for x in xs]


def show_summary_gridplots(xdfInput, xkateg, xcolID=None):
    """
        Pairplot für alle Features in Input-Dataframe.
        Es dürfen nur numerische Werte in xdfInput geben (außer xkateg und eventuell xcolID)
        Beispiel:
            xdiagramme.show_summary_gridplots(xdfVis[[u'target', u'g_avg'] + xcorrDict[u'g_avg']], xkateg="target", xcolID="object_id")
        Nutzung: Korrelationen
    """
    import seaborn as sbn

    xdfInput = reduziere_dataframe(xdfInput.copy())

    if xkateg not in xdfInput.columns:
        print("[x] FEHLER: Aufruf show_summary_gridplots mit xkateg nicht existent:", xkateg)
        print("[x] Vorhandene Features:", xdfInput.columns)
        return
    if len(set(xdfInput[xkateg])) > 20:
        print(
            "[x] Anzahl der unterschiedlichen Werte in " + xkateg + " ist > 20, wahrscheinlich keine kategorialer Variable gegeben")
        print("[x] show_summary_gridplots ist nicht möglich")
        return
    xdfInput = xdfInput.copy()
    xdfInput = xdfInput[[x for x in xdfInput.columns if x != xcolID]]
    xdfInput = xdfInput.dropna()

    sbn.pairplot(xdfInput, vars=[x for x in xdfInput.columns if x != xkateg],
                 hue=xkateg, height=1.5, aspect=2,
                 plot_kws=dict(s=10, edgecolor="b", linewidth=1, alpha=0.8))

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(None, VERWENDE_DIR, fname="xdg_show_summary_gridplots")


def show_summary_plot(xdfInput, xcol, ycol, type=0):
    """
        Wert type:
            0: cubehelix
            1: jointplot
            2: mit histogram, dichtediagramm und Regression
            3: zusätzlich sbn.kdeplot einsatz
        Beispiel:
            xdg.show_summary_plot(xdfVis, xcol=xCol1, ycol=xCol2, type=0)
    """
    import seaborn as sbn

    xdfInput = jitter_feature(xdfInput.copy(), xcol, ycol)
    xdfInput = reduziere_dataframe(xdfInput)

    sbn.set_style("white")
    if type == 0:
        cmap = sbn.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
        sbn.kdeplot(xdfInput[xcol], xdfInput[ycol], cmap=cmap, n_levels=10, shade=True)
    elif type == 1:
        g = sbn.jointplot(x=xcol, y=ycol, data=xdfInput, kind="kde", color="m", alpha=0.8)
        g.plot_joint(plt.scatter, c="black", s=SCATTER_SPARAM, linewidth=1, marker="+", alpha=0.8)
        g.ax_joint.collections[0].set_alpha(0)
        g.set_axis_labels(xcol, ycol)
    elif type == 2:
        g = sbn.jointplot(x=xcol, y=ycol, data=xdfInput, kind="reg")
        g.plot_joint(plt.scatter, c="cyan", s=SCATTER_SPARAM, linewidth=1, marker="+", alpha=0.3)
    elif type == 3:
        g = sbn.jointplot(x=xcol, y=ycol, data=xdfInput, s=SCATTER_SPARAM, alpha=0.0, kind="scatter")
        g.plot_joint(sbn.kdeplot, zorder=0, n_levels=4, linewidth=2)
        g.plot_joint(plt.scatter, c="cyan", s=SCATTER_SPARAM, linewidth=1, marker="+", alpha=0.3)

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(None, VERWENDE_DIR, fname="xdg_" + "show_summary_plot")


def show_summary_scatterplot_farbe_kateg(xdfInput, xcol, ycol, xkateg, xtitle=None):
    """
        Beispiel:
        https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
    """

    xdfInput = jitter_feature(xdfInput.copy(), xcol, ycol)
    xdfInput = reduziere_dataframe(xdfInput)

    if xtitle is None:
        xtitle = 'Streudiagramme und Histogramme'
    print("[x] %s:" % xcol)
    print(xdfInput[xcol].describe())
    print("[x] %s:" % ycol)
    print(xdfInput[ycol].describe())
    fig = plt.figure(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NIEDRIG)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
    categories = np.unique(xdfInput[[x is not None for x in xdfInput[xkateg].values]][xkateg])
    colors = [xfCOLOR(i / float(len(categories) - 1)) for i in range(len(categories))]
    for i, category in enumerate(categories):
        ax_main.scatter(xcol, ycol, s=SCATTER_SPARAM, c=np.array(colors[i]).reshape(1, -1), alpha=.7,
                        data=xdfInput.loc[xdfInput[xkateg] == category, :][[xcol, ycol]],
                        cmap="tab10", edgecolors='gray', linewidths=.5,
                        label=str(category))  # xdfInput[xkateg].astype('category').cat.codes
    ax_main.legend(handler_map={PathCollection: HandlerPathCollection(update_func=update)},
                   fontsize=einstellungen.FIG_FONTZISE)
    ax_bottom.hist(xdfInput[xcol], 40, histtype='stepfilled', orientation='vertical', color='cyan')
    ax_bottom.invert_yaxis()
    ax_right.hist(xdfInput[ycol], 40, histtype='stepfilled', orientation='horizontal', color='cyan')
    ax_main.set(title=xtitle, xlabel=xcol, ylabel=ycol)
    ax_main.title.set_fontsize(16)
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(12)
    xlabels = ax_main.get_xticks().tolist()
    ax_main.set_xticklabels(xlabels)
    plt.tight_layout()
    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(None, VERWENDE_DIR, fname="xdg_" + "show_summary_scatterplot_farbe_kateg")


def show_histogram_by_target_quantiles(xdfInput, xcol, xcolTarget, qFirst=0.1, qLast=0.9):
    """ Dasselbe nur mit Quantilen
        xcolTarget muss schon in quantilen
    """

    if xcol == "target_bin" or xcol == xcolTarget:
        return

    xdfInput = reduziere_dataframe(xdfInput.copy())

    x_var = xcol
    groupby_var = xcolTarget
    xdfInput = xdfInput[[x is not None for x in xdfInput[xcolTarget].values]].copy()
    xWerteTarget = list(set(xdfInput[xcolTarget]))
    xWerteTargetUnique = [x for x in xWerteTarget if len(set(xdfInput[xdfInput[xcolTarget] == x][xcol])) <= 2]
    if len(xWerteTargetUnique) != 0:
        print("[x] WARNUNG: einige der Werte in der kategorialer Variable tauchen zu wenig auf!:")
        print(xWerteTargetUnique)
        print("[x] Diese werden in Histogram nicht aufgenommen")
        print("[x] xcol:", xcol)
        print("[x] xcolTarget:", xcolTarget)
    xdfInput = xdfInput[[x not in xWerteTargetUnique for x in xdfInput[xcolTarget]]]
    df_agg = xdfInput.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [xdf[x_var].values.tolist() for i, xdf in df_agg]
    xlimS = pd.Series(reduce(lambda a, b: a + b, vals))
    xlim = (xlimS.quantile(qFirst), xlimS.quantile(qLast))
    if xlim[0] == xlim[1]:
        print("[x] Fehler: Quantilen müssen unterschiedlich sein", xlim)
        return
    vals = [[y for y in x if xlim[0] < y < xlim[1]] for x in vals]
    plt.figure(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NIEDRIG)
    colors = [xfCOLOR(i / float(len(vals) - 1)) for i in range(len(vals))]
    xdictlegend = {group: col for group, col in zip(np.unique(xdfInput[groupby_var]).tolist(), colors[:len(vals)])}
    n, bins, patches = plt.hist(vals, bins=25, stacked=True, density=False, color=colors[:len(vals)])
    xN = max([x.max() for x in n])
    plt.legend(xdictlegend, loc='upper right')
    plt.title("Histogramm %s Farbe nach %s" % (x_var, groupby_var), fontsize=einstellungen.FIG_FONTZISE)
    plt.xlabel(x_var)
    plt.ylabel("Anzahl")
    plt.xticks(bins[::3], [round(b, 1) for b in bins[::3]])
    plt.ylim(0, 1.1 * xN)
    plt.tight_layout()

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(None, VERWENDE_DIR, fname="xdg_" + "show_histogram_by_target_quantiles")


def histogram_stack_kateg(xdfin, xcol, xcolTarget, returnBinData=False):
    """
        Histogram einer kategorialer Variable xcol
        mit Stack per einer anderer kateg Variable
        xcol: wenn == None resultiert in einem Barplot nach target,
              wenn != None dann Histogramm
    """
    xdfInput = reduziere_dataframe(xdfin.copy())

    x_var = xcol
    groupby_var = xcolTarget
    xdfInput = xdfInput[[x is not None for x in xdfInput[xcolTarget].values]].copy()
    xdfInput[xcolTarget] = kateg_aus_numerischer_variable(xdfInput[xcolTarget])

    if x_var is not None:
        df_agg = xdfInput.loc[:, [x_var, groupby_var]].groupby(groupby_var)
        vals = [df[x_var].values.tolist() for i, df in df_agg]
    else:
        df_agg = xdfInput.loc[:, [groupby_var]].groupby(groupby_var)
        vals = [df.shape[0] for i, df in df_agg]

    ids = [i for i, df in df_agg]
    fig = plt.figure(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NIEDRIG)
    if len(vals) > 2:
        colors = [xfCOLOR(i / float(len(vals) - 1)) for i in range(len(vals))]
        xdictlegend = {group: col for group, col
                       in list(zip(np.unique(xdfInput[groupby_var]).tolist(), colors[:len(vals)]))
                       }
    else:
        colors = None
        xdictlegend = None

    if x_var is not None:
        # xdfInput[x_var].unique().__len__()
        if len(vals) > 2:
            n, bins, patches = plt.hist(vals, 5, stacked=True, density=False, color=colors[:len(vals)])
        else:
            n, bins, patches = plt.hist(vals, 5, stacked=True, density=False)
        xN = max([x.max() for x in n])
        plt.xticks(bins[::1], [round(b, 1) for b in bins[::1]])
        plt.xlabel(x_var)
        """
        if xdictlegend is not None:
            plt.legend(xdictlegend, loc='upper right')
        else:
            plt.legend(loc='upper right')
        """
        plt.legend(loc='upper right')
        plt.title("Histogramm %s Farbe nach %s" % (x_var, groupby_var), fontsize=einstellungen.FIG_FONTZISE)
        _ = (n, bins, patches)
    else:
        _ = plt.bar(x=range(len(vals)), height=vals, tick_label=ids)
        xN = max(vals)
        plt.xlabel("Kategorie")
        plt.title("Barplot %s" % (xcolTarget), fontsize=einstellungen.FIG_FONTZISE)

    plt.ylabel("Anzahl")
    plt.ylim(0, 1.1 * xN)
    plt.tight_layout()

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(fig, VERWENDE_DIR, fname="xdg_" + "histogram_stack_kateg")

    if returnBinData:
        return _


def histogram_dichte_diagramm(xdfInput, xcol, titel="", alpha=0.05):
    """ Histogramme und Dichtediagramme """
    import seaborn as sbn
    import scipy.stats as stats
    from flaubert.statistik import stat_op
    sbn.set_style("white")
    plt.figure(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NIEDRIG)
    ax = sbn.histplot(xdfInput[xcol], bins=45,
                      kde=True, stat="density", kde_kws=dict(cut=3))
    meanc, varc, stdc = stats.bayes_mvs(xdfInput[xcol], alpha=.95)
    deltay = 0.05 * (ax.viewLim.y1 - ax.viewLim.y0)
    kdef, xL, xH = stat_op.kde_fit(xdfInput[xcol], alpha=alpha, minWert=0.0, doPlot=False)
    print(f"[x] xL={xL}, xH={xH}")
    ax_lim_y = ax.viewLim.y1
    ax.vlines(x=meanc.statistic, ymin=ax.viewLim.y0, ymax=0.9 * ax_lim_y, linewidth=4, color="black",
              linestyles="dashed")
    ax.vlines(x=xL, ymin=ax.viewLim.y0, ymax=ax_lim_y, linewidth=2, color="cyan", linestyles="dashed")
    ax.vlines(x=xH, ymin=ax.viewLim.y0, ymax=ax_lim_y, linewidth=2, color="cyan", linestyles="dashed")
    ax.text(meanc.statistic, ax.viewLim.y1 - 3 * deltay,
            "  mu = " + str(np.round(meanc.statistic, 2)) +
            " mit CI 95% ({}, {})".format(*np.round(meanc.minmax, 2)))
    ax.text(meanc.statistic, ax.viewLim.y1 - 4 * deltay, "  std = " + str(np.round(stdc.statistic, 2)))
    ax.text(xL, ax.viewLim.y1 - deltay, "  min für P(x<min) <= {}: ".format(alpha / 2.) + str(np.round(xL, 2)))
    ax.text(xH, ax.viewLim.y1 - deltay, "  max für P(x>max) <= {}: ".format(alpha / 2.) + str(np.round(xH, 2)))
    plt.title(titel)
    plt.tight_layout()
    plt.show()


def histogram_dichte_kateg(xdfInput, xcol, xcolTarget):
    """ Histogramme und Dichtediagramme
        xcol: numerisch
        xcolTarget: ist kategorial
    """
    import seaborn as sbn
    from flaubert.statistik import stat_op as statistik

    xdfInput = reduziere_dataframe(xdfInput.copy())

    fig = plt.figure(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NIEDRIG)
    xdfInput = xdfInput.copy()
    xs = pd.Series(statistik.standardize(xdfInput[xcol]))
    xs, xm = statistik.make_series_positiv(xs)
    xdfInput[xcol] = xs
    xdfInput = xdfInput[[x is not None for x in xdfInput[xcolTarget]]]
    xcolTargetWerte = list(set(xdfInput[xcolTarget]))
    colors = [xfCOLOR(i / float(len(xcolTargetWerte) - 1)) for i in range(len(xcolTargetWerte))]
    xYMaxCount = 1
    for i, xcolTW in enumerate(xcolTargetWerte):
        xplotData = xdfInput.loc[xdfInput[xcolTarget] == xcolTW, xcol]
        xplotData = xplotData.dropna()
        if xYMaxCount < xplotData.max():
            xYMaxCount = xplotData.max()
        if xplotData.shape[0] < 5:
            print("[x] WARNUNG: zu wenige Daten für", xcolTW)
        else:
            sbn.histplot(xplotData, color=colors[i], label=xcolTW,
                         kde=True, stat="density", kde_kws=dict(cut=3))
            # hist_kws={'alpha': .5}, kde_kws={'linewidth': 3, "lw": 3})
    # plt.ylim(0, xYMaxCount)
    plt.title('Dichtediagramm %s grouppiert nach %s' % (xcol, xcolTarget), fontsize=einstellungen.FIG_FONTZISE)
    plt.legend()
    plt.tight_layout()

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(fig, VERWENDE_DIR, fname="xdg_" + "histogram_dichte_kateg")


def show_counts_kateg(xdfIn, xcol, xcolTarget):
    """ zeige counts einer kategorialer Variable """

    xdfInput = reduziere_dataframe(xdfIn.copy())
    xdfInput[xcolTarget] = [str(w) for w in xdfInput[xcolTarget].values]

    if len(set(xdfInput[xcolTarget])) > 25:
        print("[x] xcol soll kategorial sein!")
        return

    if xcolTarget not in xdfInput.columns:
        print("[x] xcol ({}) nicht in xdfInput ({})".format(xcolTarget, xdfInput.columns))
        return

    sbn.countplot(x=xcol, hue=xcolTarget, data=xdfInput)
    plt.tight_layout()

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(None, VERWENDE_DIR, fname="xdg_" + "show_counts_kateg")


def show_corrplot(xdfInput, zeigeUeberschriften=False):
    """ zeige die Korrelationsdiagramme (nur numerisch) """
    xcorr = xdfInput.corr()
    sbn.set(style="white")
    mask = np.zeros_like(xcorr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=einstellungen.FIGSIZE_BREIT)
    cmap = sbn.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
    sbn.heatmap(xcorr, mask=mask, cmap=cmap, annot=zeigeUeberschriften, vmax=.5, center=0, square=True, linewidths=1,
                cbar_kws={"shrink": .5}, annot_kws={"size": einstellungen.FIG_FONTZISE})
    plt.xticks(rotation=0, fontsize=einstellungen.FIG_FONTZISE, alpha=.7)
    plt.yticks(rotation=0, fontsize=einstellungen.FIG_FONTZISE, alpha=.7)
    plt.title("Korrelationen")

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(None, VERWENDE_DIR, fname="xdg_" + "show_corrplot")


def show_scatter_kateg(xdfInput, xcol, ycol, xkateg, xlim=None, ylim=None, xtitle=None):
    """
        Streudiagramm für ein X=xcol, Y=ycol gefärbt nach xkateg Werte, alle Werte in selber Diagramm
        Beispiel:
            xdiagramme.show_scatter_kateg(xdotrain, xcol="r_lZII", ycol="hostgal_photoz", xkateg="target")
        xlim, ylim in Format (0, 90000)
        Inspiration: https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
        Nutzung: Überblick Ausreißer
    """

    if xcol == "dfPlotK" or ycol == "dfPlotK":
        return

    xdfInput = jitter_feature(xdfInput.copy(), xcol, ycol)
    xdfInput = reduziere_dataframe(xdfInput)

    if xtitle is None:
        xtitle = "Streudiagram " + xcol + " vs " + ycol
    categories = np.unique(xdfInput[[x is not None for x in xdfInput[xkateg].values]][xkateg])
    colors = [xfCOLOR(i / float(len(categories) - 1)) for i in range(len(categories))]
    plt.figure(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NORM, facecolor='w', edgecolor='k')
    for i, category in enumerate(categories):
        plt.scatter(xcol, ycol,
                    data=xdfInput.loc[xdfInput[xkateg] == category, :][[xcol, ycol]],
                    s=SCATTER_SPARAM, c=np.array(colors[i]).reshape(1, -1), label=str(category), alpha=0.5)

    if xlim is not None and ylim is not None:
        plt.gca().set(xlim=xlim, ylim=ylim, xlabel=xcol, ylabel=ycol)
    plt.xticks(fontsize=einstellungen.FIG_FONTZISE)
    plt.yticks(fontsize=einstellungen.FIG_FONTZISE)
    plt.gca().set_xlabel(xcol)
    plt.gca().set_ylabel(ycol)
    plt.title(xtitle, fontsize=einstellungen.FIG_FONTZISE)
    plt.legend(handler_map={PathCollection: HandlerPathCollection(update_func=update)},
               fontsize=einstellungen.FIG_FONTZISE)

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(None, VERWENDE_DIR, fname="xdg_" + "show_scatter_kateg")


def show_one_scatterplot_per_kateg_lm_model(xdfInput, xcol, ycol, xkateg, xlim=None, ylim=None):
    """
        Zeigt die Streudiagrammen der Datensätze pro Wert der xkateg nebeneinander.
        Expandiert show_scatter per kategs; Liste Plots mit den Scatterplts für jeden Wert in xKateg
        Im Prinzip das hier ist die expandierter Form der Methode show_scatter_kateg()
        Beispiel:
            xdiagramme.show_scatterplots_kateg(xdotrain, xcol="hostgal_photoz", ycol="u_avg", xkateg="target")
        https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
        Nutzung: Überblick Ausreißer
    """
    import seaborn as sbn

    if xcol == "dfPlotK" or ycol == "dfPlotK":
        return

    xdfInput = jitter_feature(xdfInput.copy(), xcol, ycol)

    xWerteUnique = list(set(xdfInput[xkateg]))
    xm = int(len(xWerteUnique) % 5)
    xn = int(len(xWerteUnique) / 5)
    if xn >= 1:
        xBucketsWerte = [xWerteUnique[(i * 5):(i * 5 + 5)] for i in range(xn)]
        if xm != 0:
            xBucketsWerte[-1].extend(xWerteUnique[(xn * 5):(xn * 5 + xm)])
    else:
        xBucketsWerte = [xWerteUnique]
    for xB in xBucketsWerte:
        xdfPlot = xdfInput[[x in xB for x in xdfInput[xkateg].values]]
        xdfPlot = reduziere_dataframe(xdfPlot.copy())
        sbn.set_style("white")
        gridobj = sbn.lmplot(x=xcol, y=ycol,
                             data=xdfPlot,
                             robust=False,
                             palette=COLOR_STR,
                             col=xkateg,
                             scatter_kws=dict(s=10, linewidths=.7, edgecolors='black'))
        if xlim is not None and ylim is not None:
            gridobj.set(xlim=xlim, ylim=ylim)
        plt.show()


def show_scatterplot_hist_one_feature(xdfInput, xcol, ycol):
    if xcol == "dfPlotK" or ycol == "dfPlotK":
        return

    xdfInput = jitter_feature(xdfInput.copy(), xcol, ycol)
    xdfInput = reduziere_dataframe(xdfInput)

    fig, ax = plt.subplots(2, 2, figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NORM)
    ax[0, 0].hist(xdfInput[xcol], bins=25)
    ax[0, 0].set_xlabel(xcol)
    ax[0, 0].set_ylabel("Anzahl")
    ax[0, 1].scatter(xdfInput[xcol], xdfInput[ycol], s=SCATTER_SPARAM, marker='+', alpha=0.8)
    ax[0, 1].set_xlabel(xcol)
    ax[0, 1].set_ylabel(ycol)
    ax[1, 0].hist(xdfInput[ycol], bins=25)
    ax[1, 0].set_xlabel(ycol)
    ax[1, 0].set_ylabel("Anzahl")
    ax[1, 1].scatter(xdfInput[ycol], xdfInput[xcol], s=SCATTER_SPARAM, marker='+', alpha=0.8)
    ax[1, 1].set_xlabel(ycol)
    ax[1, 1].set_ylabel(xcol)
    plt.tight_layout()
    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(fig, VERWENDE_DIR, fname="xdg_" + "show_scatterplot_hist_one_feature")


def show_scatterplot_xy_size_color(df, xcol=6, ycol=7, value_feature=4, scale_factor=5, color_feature=16):
    """
        df: input DataFrame
        xcol, ycol: die x und y achsen
        value_feature: verwendet als "size" Parameter s
        scale_factor: skaliert den Parameter size
        color_feature: die Farbe der Bobbels
    """
    df.plot(kind="scatter", x=xcol, y=ycol, alpha=0.5, s=df[value_feature] * scale_factor,
            label=value_feature,
            figsize=(10, 7),
            c=color_feature,
            cmap=plt.get_cmap("jet"), colorbar=True, )
    plt.legend()
    plt.show()


def boxplot(xdf_input, xcol, target_col):
    """ Eine Liste Boxplots pro Kategorie """
    import seaborn as sbn

    xdf_input = reduziere_dataframe(xdf_input.copy())
    xdf_input[target_col] = [str(x) for x in xdf_input[target_col].values.tolist()]

    fig = plt.figure(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NORM)
    sbn.boxplot(x=target_col, y=xcol, data=xdf_input, notch=False)

    def add_n_obs(df, group_col, y):
        medians_dict = {grp[0]: grp[1][y].median() for grp in df.groupby(group_col)}
        xticklabels = [x.get_text() for x in plt.gca().get_xticklabels()]
        n_obs = df.groupby(group_col)[y].size().values
        for (x, xticklabel), n_ob in zip(enumerate(xticklabels), n_obs):
            plt.text(x, medians_dict[xticklabel], "#obs : " + str(n_ob), horizontalalignment='center',
                     fontdict={'size': 14, "weight": 'bold'}, color='black')

    add_n_obs(xdf_input, group_col=target_col, y=xcol)
    plt.title('Boxplot %s gruppiert nach %s, Labels befinden sich am Median' % (xcol, target_col),
              fontsize=einstellungen.FIG_FONTZISE)
    # plt.ylim(10, 40)
    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(fig, VERWENDE_DIR, fname="xdg_" + "boxplot")


def show_divergent_bars(xdfInput, xcol, xkateg, avgFunk="avg", xlim=None, xtitle=None):
    """
        xcol: numerisches Feature
    """

    xdfInput = reduziere_dataframe(xdfInput.copy())

    if avgFunk == "avg":
        xdfInput = xdfInput.groupby(xkateg).mean()
    elif avgFunk == "sum":
        xdfInput = xdfInput.groupby(xkateg).mean()
    elif avgFunk == "sum":
        xdfInput = xdfInput.groupby(xkateg).sum()
    elif avgFunk == "median":
        xdfInput = xdfInput.groupby(xkateg).median()
    xdfInput[xkateg] = xdfInput.index.values
    x = xdfInput.loc[:, [xcol]]
    xnewcol = xcol + '_divbar'
    xdfInput[xnewcol] = (x - x.mean()) / x.std()
    xdfInput['colors'] = ['red' if x < 0 else 'brown' for x in xdfInput[xnewcol]]
    xdfInput.sort_values(xnewcol, inplace=True)
    xdfInput.reset_index(drop=True)

    fig = plt.figure(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NORM)
    plt.hlines(y=xdfInput.index, xmin=0, xmax=xdfInput[xnewcol], linewidth=10, color=xdfInput.colors)
    for x, y, tex in zip(xdfInput[xnewcol], xdfInput.index, xdfInput[xnewcol]):
        t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left',
                     verticalalignment='center', fontdict={'color': 'red' if x < 0 else 'brown', 'size': 12})
    plt.yticks(xdfInput.index, xdfInput[xkateg], fontsize=einstellungen.FIG_FONTZISE)
    plt.ylabel(xkateg)
    if xtitle is None:
        plt.title("Daten gruppiert (%s) pro %s, dann normalisiert: " % (avgFunk, xcol), fontdict={'size': 16})
    else:
        plt.title(xtitle, fontdict={'size': 16})
    plt.grid(linestyle='--', alpha=0.8)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(fig, VERWENDE_DIR, fname="xdg_" + "show_divergent_bars")


def show_area_chart(xdfInput, xcol, ycol, doRet=True, doDiffs=False,
                    markOutliers=True, xlim=None, ylim=None, xtitle=None):
    import numpy as np
    from flaubert.statistik import stat_op
    from flaubert.model.outliers import xIQR

    xdfInput = jitter_feature(xdfInput.copy(), xcol, ycol)
    xdfInput = reduziere_dataframe(xdfInput)

    xdfInput = xdfInput.reset_index(drop=True)
    x = np.arange(xdfInput.shape[0])
    xnewcol = ycol + "_st"
    xdfInput[xnewcol] = stat_op.standardize(xdfInput[ycol])

    xdfInput["y_returns"] = xdfInput[ycol].values
    if doRet:
        xdfInput["y_returns"] = (xdfInput[xnewcol].diff().fillna(0) / xdfInput[xnewcol].shift(1)).fillna(0) * 100
    elif doDiffs:
        xdfInput["y_returns"] = xdfInput[xnewcol].diff().fillna(0)
    else:
        xdfInput["y_returns"] = xdfInput[xnewcol]

    xL = []
    if markOutliers:
        xoutliers = xIQR.run(xdfInput["y_returns"])
        if xoutliers.shape[0] > 0:
            print("[x] Ausreißern:")
            print(xoutliers)
            for k in range(len(xdfInput["y_returns"])):
                if xdfInput.index[k] in xoutliers.index.values:
                    xL.append([k, round(xdfInput["y_returns"].iloc[k], 2)])

    print(xdfInput)
    y_returns = xdfInput["y_returns"]

    fig = plt.figure(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NORM)
    if doRet:
        plt.fill_between(x[1:], y_returns[1:], 0, where=y_returns[1:] >= 0, facecolor='black', interpolate=True,
                         alpha=0.7)
        plt.fill_between(x[1:], y_returns[1:], 0, where=y_returns[1:] <= 0, facecolor='red', interpolate=True,
                         alpha=0.7)
    else:
        plt.fill_between(x, y_returns, 0, where=y_returns >= 0, facecolor='black', interpolate=True, alpha=0.7)
        plt.fill_between(x, y_returns, 0, where=y_returns <= 0, facecolor='red', interpolate=True, alpha=0.7)

    print("[x] Ausreißern:", xL)
    for xlPeak in xL:
        xkoord, ykoord = xlPeak
        plt.annotate('Ausreißer: ' + str(xlPeak), xy=(xkoord, ykoord), xytext=(xkoord, ykoord),
                     bbox=dict(boxstyle='square', fc='firebrick'),
                     arrowprops=dict(facecolor='steelblue', shrink=0.05), fontsize=einstellungen.FIG_FONTZISE,
                     color='white')

    xtickvals = np.round(np.array(xdfInput[xcol].values.tolist()), 4)
    plt.gca().set_xticks(x[::6])
    plt.gca().set_xticklabels(xtickvals[::6], rotation=45,
                              fontdict={'horizontalalignment': 'center', 'verticalalignment': 'baseline'})
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if xtitle is not None:
        plt.title(xtitle, fontsize=einstellungen.FIG_FONTZISE)
    plt.ylabel(ycol)
    plt.xlabel(xcol)
    plt.grid(alpha=0.5)

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    # diagramme_zum_ordner_weiterleiten(fig, VERWENDE_DIR, fname="xdg_" + "show_area_chart")
    plt.show()


def do_periodogram(xsIN=None, wmin=0.01, wmax=10, showStuff=False):
    """
        xsIN: ein numpy array mit Spalte 0 Zeitstempel und 1 Werte
    """
    from flaubert.utils import signal_verarbeitung as xsv
    pgram, wList, t, y = xsv.get_periodogram(xsIN, wmin, wmax, False, vollepackung=True)
    if showStuff:
        fig = plt.subplot(2, 1, 1)
        plt.scatter(t, y, marker='+')
        plt.subplot(2, 1, 2)
        plt.plot(wList, pgram)
        diagramme_zum_ordner_weiterleiten(None, VERWENDE_DIR, fname="xdg_" + "do_periodogram")
    return pgram


def show_zeitreihe(xdfInput, xcolZeitreihe, xcolDatumTag, xtitle=None, doTicksByStep=None, doMinMaxByStep=None,
                   doRound=3,
                   showPlot=True, verwende_dir=None, arbeitspaketnamen="", fname_suffix=""):
    """   """
    from flaubert.utils import signal_verarbeitung as dfU
    xdfInput = xdfInput.copy()
    xdfInput = xdfInput.reset_index(drop=True)
    data = xdfInput[xcolZeitreihe].values
    peak_locations, through_locations = dfU.get_signal_minmax_locations(data)
    ymin = xdfInput[xcolZeitreihe].values.min()
    ymin = 0.8 * ymin if ymin > 0 else 1.2 * ymin
    ymax = xdfInput[xcolZeitreihe].values.max()
    ymax = 1.2 * ymax if ymax > 0 else 0.8 * ymax
    # print("[x] Min/Max Werte:", ymin, ymax)
    xdeltaAbstand = (ymax - ymin) / 18.

    fig = plt.figure(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NORM)
    plt.plot(xcolDatumTag, xcolZeitreihe, data=xdfInput, color='tab:blue', label=xcolZeitreihe)
    plt.scatter(xdfInput[xcolDatumTag].loc[peak_locations], xdfInput[xcolZeitreihe].loc[peak_locations], marker="+",
                color='tab:green', s=SCATTER_SPARAM, label='Höhen')
    plt.scatter(xdfInput[xcolDatumTag].loc[through_locations], xdfInput[xcolZeitreihe].loc[through_locations],
                marker="o", color='tab:red', s=SCATTER_SPARAM, label='Tiefen')
    zipListe = list(zip(through_locations, peak_locations)) if doMinMaxByStep is None else list(
        zip(through_locations[::doMinMaxByStep], peak_locations[::doMinMaxByStep]))

    for t, p in zipListe:
        plt.text(xdfInput[xcolDatumTag].iloc[p], xdeltaAbstand + xdfInput[xcolZeitreihe].iloc[p],
                 round(xdfInput[xcolDatumTag].iloc[p], doRound), horizontalalignment='center', color='darkgreen')
        plt.text(xdfInput[xcolDatumTag].iloc[t], -xdeltaAbstand + xdfInput[xcolZeitreihe].iloc[t],
                 round(xdfInput[xcolDatumTag].iloc[t], doRound), horizontalalignment='center', color='darkred')
    stepTicks = 5 if doTicksByStep is None else doTicksByStep
    xtick_location = xdfInput[xcolDatumTag].tolist()[::stepTicks]
    xtick_labels = xdfInput[xcolDatumTag].tolist()[::stepTicks]
    plt.xticks(xtick_location, xtick_labels, rotation=45, fontsize=einstellungen.FIG_FONTZISE, alpha=.7)
    plt.title("Zeitreihe Höhen/Tiefen %s" % xcolZeitreihe, fontsize=einstellungen.FIG_FONTZISE)
    plt.ylim(ymin - 2 * xdeltaAbstand, ymax + 2 * xdeltaAbstand)
    plt.yticks(fontsize=einstellungen.FIG_FONTZISE, alpha=.7)
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=.3)
    if xtitle is not None:
        plt.title(xtitle + "({})".format(arbeitspaketnamen), fontsize=einstellungen.FIG_FONTZISE)

    diagramme_zum_ordner_weiterleiten(fig, verwende_dir=verwende_dir,
                                      fname="xdg_peak_locations_{}".format(fname_suffix))


def zeitreihe_pcf(xdfInput, xcolZeitreihe):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NORM)
    plot_acf(xdfInput[xcolZeitreihe], ax=ax1)  # lags=50
    plot_pacf(xdfInput[xcolZeitreihe], ax=ax2)  # lags=20
    ax1.spines["top"].set_alpha(.3)
    ax2.spines["top"].set_alpha(.3)
    ax1.spines["bottom"].set_alpha(.3)
    ax2.spines["bottom"].set_alpha(.3)
    ax1.spines["right"].set_alpha(.3)
    ax2.spines["right"].set_alpha(.3)
    ax1.spines["left"].set_alpha(.3)
    ax2.spines["left"].set_alpha(.3)
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(fig, VERWENDE_DIR, fname="xdg_" + "zeitreihe_pcf")


def zeitreihe_pro_kateg(xdfInput, xcolZeitreihe, xcolDatum,
                        xcolKateg, doSampleN=1000,
                        xtitle=None, shortenDatumLambda=None, defaultPlotColumnsN=4):
    """
        Gegeben ein DataFrame mit einer kategorialer Variable (xcolKategWert) und ein Feature Zeitreihe (xcolZeitreihe)
        gibt eine Visualisierung aller Plots pro Kategorie
        shortenDatumLambda = lambda xstr: xstr[:7]
    """
    xdfInput = xdfInput[[x is not None for x in xdfInput[xcolKateg]]]
    xdfInput = xdfInput.reset_index(drop=True)
    if doSampleN and xdfInput.shape[0] > doSampleN:
        xdfInput = xdfInput.sample(doSampleN)
        xdfInput = xdfInput.sort_values(by=xcolDatum)
        xdfInput = xdfInput.reset_index(drop=True)
    x = np.array(xdfInput[xcolDatum].values.tolist())
    xstepby = int(0.1 * len(x))
    xlistKateg = list(set(xdfInput[xcolKateg].values))
    if len(xlistKateg) > 1:
        mycolors = [xfCOLOR(i / float(len(xlistKateg) - 1)) for i in range(len(xlistKateg))]
        nplotsRows = [k for k in range(6) if (defaultPlotColumnsN * k) > (len(xlistKateg) - 1)][0]
    else:
        mycolors = "cyan"
        nplotsRows = 1
    fig, ax = plt.subplots(nplotsRows, defaultPlotColumnsN, figsize=einstellungen.FIGSIZE_BREIT,
                           dpi=einstellungen.FIGDPI_NORM)
    # print nplotsRows, defaultPlotColumnsN
    for i in range(nplotsRows):
        for j in range(defaultPlotColumnsN):
            if i * defaultPlotColumnsN + j < len(xlistKateg):
                if (nplotsRows == 1 and defaultPlotColumnsN == 1):
                    xAX = ax
                elif (nplotsRows == 1 and defaultPlotColumnsN > 1):
                    xAX = ax[j]
                elif ((nplotsRows > 1 and defaultPlotColumnsN == 1)):
                    xAX = ax[i]
                else:
                    xAX = ax[i, j]
                ywerte = xdfInput[xdfInput[xcolKateg] == xlistKateg[i * defaultPlotColumnsN + j]][xcolZeitreihe]
                xwerte = xdfInput.loc[ywerte.index][xcolDatum].values.tolist()
                xAX.plot(xwerte, ywerte.values.tolist(), label=xlistKateg[i * defaultPlotColumnsN + j],
                         color=mycolors[i * defaultPlotColumnsN + j], linewidth=2.5)
                xAX.legend(loc='best', fontsize=einstellungen.FIG_FONTZISE)
                if shortenDatumLambda is None:
                    xAX.set_xticks(xwerte[::xstepby])
                    xAX.set_xticklabels(np.array(xwerte)[::xstepby])
                else:
                    xAX.set_xticks(xwerte[::xstepby])
                    xAX.set_xticklabels([shortenDatumLambda(a) for a in np.array(xwerte)[::xstepby].tolist()])
    if xtitle is not None:
        ax.set_title(xtitle, fontsize=einstellungen.FIG_FONTZISE)
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(fig, VERWENDE_DIR, fname="xdg_" + "zeitreihe_pro_kateg")


def extract_coords(xL):
    xs = [xL[k][0] for k in range(len(xL))]
    ys = [xL[k][1] for k in range(len(xL))]
    zs = [xL[k][2] for k in range(len(xL))]
    return xs, ys, zs


def plot_xL_3d(xL, P=None, Q=None, xKoordinaten=None, xLOutliers=None, x3dlabels=("x0", "x1", "z")):
    """
        xL: Liste der Koordinaten aller 3D-Vektoren
        Linie(P: 3d Vektor Start, Q:3D-Vektor Ende)
        xKoordinaten: In Fall einer NN-Funktion, xKoordinaten sind die vorausgesagten Koordinaten (mit Abstand Delta) zwischen P und Q
        xLOutliers: Liste Koordinaten von Vektoren, die als Ausreißern gelten
        Beispiel:
            - gegeben xdf Pandas DataFrame
            from flaubert.vis import xdiagramme as xdg
            xdg.plot_xL_3d(xdf[["x0", "x1", "z"]].values, x3dlabels=["x0", "x1", "z"])
    """
    ax = plt.axes(projection='3d')
    xs, ys, zs = extract_coords(xL)
    xcolors = (np.array(zs) - np.min(zs)) / (np.max(zs) - np.min(zs)) * 255
    ax.scatter3D(xs, ys, zs, s=10, alpha=0.7, c=np.array(xcolors).reshape(1, -1), cmap=cm.viridis)
    ax.set_xlabel(x3dlabels[0])
    ax.set_ylabel(x3dlabels[1])
    ax.set_zlabel(x3dlabels[2])
    if xLOutliers is not None:
        for elemOutlierList, farbeStr in xLOutliers:
            xsOut, ysOut, zsOut = extract_coords(elemOutlierList)
            ax.scatter3D(xsOut, ysOut, zsOut, s=15, c=farbeStr, edgecolors='w')
    if P is None:
        xL_x0_pos = [x for x in xL if x[0] > 0 and x[1] > 0]
        xL_x0_neg = [x for x in xL if x[0] < 0 and x[1] < 0]
        if len(xL_x0_pos) == 0 or len(xL_x0_neg) == 0:
            print("[x] keine negativen && positiven Werte in einer der Variablen")
            kRandom1 = np.random.choice(range(len(xL)))
            kRandom2 = np.random.choice(range(len(xL)))
            xL_x0_neg = xL
            xL_x0_pos = xL
        else:
            kRandom1 = np.random.choice(range(len(xL_x0_pos)))
            kRandom2 = np.random.choice(range(len(xL_x0_neg)))
        xrandomP = xL_x0_pos[kRandom1]
        xrandomQ = xL_x0_neg[kRandom2]
        print("[x] P:", xL_x0_pos[kRandom1])
        print("[x] Q:", xL_x0_neg[kRandom2])
        ax.plot3D([xrandomP[0], xrandomQ[0]], [xrandomP[1], xrandomQ[1]], [xrandomP[2], xrandomQ[2]], c="red")
    if xKoordinaten is not None:
        for k in range(1, len(xKoordinaten)):
            Pnn = xKoordinaten[k - 1]
            Qnn = xKoordinaten[k]
            ax.plot3D([Pnn[0], Qnn[0]], [Pnn[1], Qnn[1]], [Pnn[2], Pnn[2]], c="black")
        print("[x] P-Q Koordinaten:", [P[0], Q[0]], [P[1], Q[1]], [P[2], P[2]])
        ax.plot3D([P[0], Q[0]], [P[1], Q[1]], [P[2], Q[2]], c="red")
    plt.show()


def plot_xdf_3d(xdfInput, P=None, Q=None, xKoordinaten=None, uVektor=None, zFeature="nnOutput"):
    print("[x] Visualiserung 3D:")
    print(xdfInput.head(3))
    xyx0 = "x0"
    xyx1 = "x1"
    if len(uVektor) > 2:
        print("[x] Wähle Feature1 (zB x0), Feature2 (zB x1):")
        xyx0 = input("feature 1: ")
        xyx1 = input("feature 2: ")
    xt = xdfInput.columns.tolist()
    indx = [xt.index(xyx0), xt.index(xyx1), xt.index(zFeature)]
    # xL, P=None, Q=None, xKoordinaten=None, xLOutliers=None, x3dlabels=("x0", "x1", "z")
    plot_xL_3d(xL=xdfInput[[xyx0, xyx1, zFeature]].values.tolist(),
               P=P[indx].tolist() if P is not None else None,
               Q=Q[indx].tolist() if Q is not None else None,
               xKoordinaten=[np.array(x)[indx] for x in xKoordinaten] if xKoordinaten is not None else None,
               x3dlabels=[xyx0, xyx1, zFeature])


def showPlotFile(xfName="", prfx="./xpydm.zip/data/"):
    with open(prfx + xfName, 'rb') as f:
        data_uri = f.read().encode('base64').replace('\n', '')
    img_tag = '%html <img src="data:image/png;base64,{0}">'.format(data_uri)
    print(img_tag)


def show_heatmap(xdf_input, target_col, xselekt_cols):
    f, ax = plt.subplots(figsize=einstellungen.FIGSIZE_BREIT)
    _ = sbn.heatmap(xdf_input[[target_col] + xselekt_cols].corr())
    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(None, VERWENDE_DIR, fname="xdg_" + "show_heatmap")


def show_corrplot_df(xABC, xcmap=plt.cm.BrBG, xrotation=0.0, yrotation=0.0,
                     xtitle="Korrelationen"):
    fig = plt.figure(dpi=einstellungen.FIGDPI_NORM)  # figsize=(12, 10)
    data = xABC.corr()
    plt.matshow(data, fignum=fig.number, cmap=xcmap)  #
    plt.xticks(list(range(xABC.shape[1])), xABC.columns, fontsize=einstellungen.FIG_FONTZISE, rotation=xrotation)
    plt.yticks(list(range(xABC.shape[1])), xABC.columns, fontsize=einstellungen.FIG_FONTZISE, rotation=yrotation)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title(xtitle)
    for (i, j), z in np.ndenumerate(data):
        plt.text(j, i, '{:0.1f}'.format(z))

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(fig, VERWENDE_DIR, fname="xdg_" + "show_corrplot_df")


def show_gridhist(xdf, xby=(5, 4)):
    """ xby wie groß sollte der Grid sein (5,4) => 5 Zeilen, 4 Spalten"""
    xcolumns = xdf.columns
    print("[x] Histogramme für:", xcolumns.tolist())
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    if xcolumns.shape[0] > (xby[0] * xby[1]):
        print("[x] xby zu niedrig eingestellt")
        return
    fig, ax = plt.subplots(xby[1], xby[0], figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NORM)
    fig.tight_layout()
    for i in range(xby[1]):
        for j in range(xby[0]):
            k = i * xby[0] + j
            if k < xcolumns.shape[0]:
                ax[i, j].hist(xdf[xcolumns[k]])
                ax[i, j].set_title(xcolumns[k])

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(fig, VERWENDE_DIR, fname="xdg_" + "show_gridhist")


def do_pairplot(xdf, xselekt):
    """
        # Pairplot für enge Auswahl Features
    """
    from matplotlib.artist import setp
    fig, ax = plt.subplots(1, figsize=einstellungen.FIGSIZE_BREIT)
    xstrMap = lambda xstr: [x for x in xstr]
    xstrPad = lambda xstr: ''.join(([' '] * (10 - len(xstr))) + xstrMap(xstr)) if len(xstr) < 10 else xstr
    axs = pd.plotting.scatter_matrix(xdf[xselekt].dropna(),
                                     figsize=einstellungen.FIGSIZE_BREIT, ax=ax,
                                     marker='o', hist_kwds={'bins': 35}, s=8, alpha=.5)
    for row in axs:
        for j in range(len(row)):
            subplot = row[j]
            setp(subplot.get_xticklabels(), rotation=0)
            setp(subplot.get_yticklabels(), rotation=0)
            ylabel = subplot.get_ylabel()
            ylabel = ylabel[:10] + (".." if len(ylabel) >= 10 else "")
            xlabel = subplot.get_xlabel()
            yticks = [str(item) for item in subplot.get_yticks()]
            if len([x for x in yticks if x != ""]) != 0:
                subplot.set_yticklabels([xstrPad(str(np.round(float(lbl), 2))) for lbl in yticks])
            subplot.set_ylabel(ylabel, rotation=90, fontdict={'fontsize': 10, 'fontweight': 'bold'})
            subplot.set_xlabel(xlabel, rotation=0, fontdict={'fontsize': 10, 'fontweight': 'bold'})
            # subplot.grid(visible=True)

    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(fig, VERWENDE_DIR, fname="xdg_" + "do_pairplot")


def show_hist_lmres(xdf_cols_stats):
    """
        Gegeben ein DataFrame plot Histogrammen
    :param xdf_cols_stats:
    :return:
    """
    xdf_cols_stats.hist(bins=25, grid=False)
    plt.autoscale(enable=True, tight=True)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    # fig, axs = plt.subplots(1, figsize=(16, 8))
    diagramme_zum_ordner_weiterleiten(fig, VERWENDE_DIR, fname="xdg_" + "show_hist_lmres")


def regression_ist_vs_pred(ist_werte, voraussage, target_col, jitter_on=(1, 1)):
    """
        Regressionsresultate in Streudiagramm visualisieren
    :return:
    """

    if True:
        r2_perf = r2_score(ist_werte, voraussage)
        rmse_perf = np.sqrt(mean_squared_error(ist_werte, voraussage))
        print("R^2 {}; RMSE {}".format(r2_perf, rmse_perf))
        f, ax = plt.subplots(figsize=einstellungen.FIGSIZE_BREIT)
        # plt.scatter(x=df_voraussage[target_col].values, y=xpred, s=4, alpha=0.3)
        jitter(x=ist_werte, y=voraussage, s=5, alpha=0.2)
        plt.xlabel(target_col)
        plt.ylabel("Voraussage ({})".format(target_col))
        plt.title("IST-Werte vs Voraussage mit Jitter")
        # fig, axs = plt.subplots(1, figsize=(16, 8))
        diagramme_zum_ordner_weiterleiten(None, VERWENDE_DIR, fname="xdg_" + "regression_ist_vs_pred")


def diagramm_weiterleiten(ax=None,
                          xfig=None,
                          titel='', x_label='', y_label='', fName='',
                          xtickLabels=None, suptitle=None, xn=10,
                          zielordner="./data/",
                          diagramme_speichern=True):
    """
        Speichert eine Diagramm unter eingegebenen Zielordner
        oder zeigt einfach ein Plot
    Beispiel Verwendung:
        diagramm_weiterleiten(ax=ax, titel='FREQUENZ aus FFT (bester Auswahl)', x_label='', y_label='',
                              fName='freq_from_fft_optim', xtickLabels=None)
    :param ax:
    :param titel:
    :param x_label:
    :param y_label:
    :param fName:
    :param xtickLabels: in Format (Zahl, Label)
    :param suptitle:
    :return:
    """
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    def format_fn(tick_val, tick_pos):
        if int(tick_val) in [int(x[0]) for x in xtickLabels]:
            xstr = [x for x in xtickLabels if int(x[0]) == int(tick_val)][0]
            return str(xstr[1])  # + " (" + str(xstr[0]) + ")"
        return ''

    if xtickLabels is not None:  # and xtickVals is not None:
        xtickLabels = list(xtickLabels)
    if ax is not None or xfig is not None:
        if ax is not None:
            if 'axes' in dir(ax):
                ax = ax.axes
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(titel)
            xfig = ax.get_figure()
        xfig.set_size_inches(15, 10, forward=True)
        if suptitle is not None:
            plt.suptitle(suptitle)
        if xtickLabels is not None:  # and xtickVals is not None:
            ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
            ax.xaxis.set_major_locator(MaxNLocator(xn, integer=True))
    plt.tight_layout()
    if diagramme_speichern and xfig is not None:
        xfig.savefig(zielordner + '/out_Visualisierung_' + fName + '.png', dpi=einstellungen.FIGDPI_NORM)
    elif diagramme_speichern is False:
        plt.show()
    plt.close()


def zeige2d(Y_star, Y_norm):
    pd.Series(Y_norm.flatten()).plot(color="red")
    pd.Series(Y_star.flatten()).plot(color="black")
    plt.title("Prognose (Rot) vs IST (Schwarz)")
    plt.show()


def zeige3d(X_norm, Y_star):
    """ 3D Visualisierung mit den originalen 2 Features (3tes Feature ist GP)
            gegen Prognose Y_star (oder Y_norm) """
    L_star = np.zeros((X_norm.shape[0], 3))
    L_star[:, 0] = X_norm[:, 0].flatten()
    L_star[:, 1] = X_norm[:, 1].flatten()
    L_star[:, 2] = Y_star[:, 0].flatten()
    plot_xL_3d(L_star, P=None, Q=None, xKoordinaten=None, xLOutliers=None, x3dlabels=("x0", "x1", "z"))


def diagramme_zum_ordner_weiterleiten(fig, verwende_dir, fname):
    plt.tight_layout()
    if verwende_dir is not None:
        if fig is not None:
            fig.savefig(verwende_dir + '/{}.png'.format(fname), dpi=einstellungen.FIGDPI_NORM)
        else:
            plt.savefig(verwende_dir + '/{}.png'.format(fname), dpi=einstellungen.FIGDPI_NORM)
        print("[x] Diagramme unter {}/{}.png gespeichert".format(verwende_dir, fname))
    else:
        plt.show()
    plt.close()


def plot(xs, xtitel="", xachse=None):
    if xachse is None:
        plt.plot(xs)
    else:
        plt.plot(xachse, xs)
    plt.title(xtitel);
    plt.show()


def plot2s(xs1, xs2, xtitel="", xachse=None):
    """ Zeigt Plots für beiden Zahlenreihen """
    if xachse is None:
        pd.Series(xs1).plot(c="black")
        pd.Series(xs2).plot(c="cyan")
    else:
        plt.plot(xachse, xs1, c="black")
        plt.plot(xachse, xs2, c="cyan")
    plt.title(xtitel)
    plt.show()


def plot3s(xs1, xs2, xs3, xtitle="", diagramme_speichern=False, zielordner="./data/tempdata/", fname="visualisierung"):
    # s1 = expandiere_zahlenreihe(xseries, faktor=5.0)
    fig = plt.figure(figsize=(16, 8), dpi=einstellungen.FIGDPI_NORM)
    plt.plot(xs1, c="black")
    plt.plot(xs2, c="cyan")
    plt.plot(xs3, c="orange")
    plt.title(xtitle)
    if diagramme_speichern and fig is not None:
        fpath = zielordner + f'/{fname}.png'
        fpath = fpath.replace("//", "/")
        fig.savefig(fpath, dpi=einstellungen.FIGDPI_NORM)
    elif diagramme_speichern is False:
        plt.show()
    plt.close()


def imshow(pixels, ticklabels=None, titel="regel 1, realintv"):
    fig, ax = plt.subplots()
    if ticklabels is not None:
        xm, xM, yM, ym = ticklabels;
        d1, d2 = (xM - xm), (yM - ym)
        # extent: floats (left, right, bottom, top)
        ax.imshow(pixels, extent=[xm, xM, ym, yM], aspect="auto", origin="lower")  # [80, 120, 32, 0]
    else:
        ax.imshow(pixels, origin="lower")
    plt.title(titel)
    plt.xlabel("koordx")
    plt.ylabel("koordy")
    plt.show()


def simpel_plot(X, Y, xtitel):
    plt.scatter(x=X, y=Y, c="blue", s=5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{xtitel}")
    plt.show()


def simpel_3dplot(X, Y):
    """ X (darf in R ^ m x n sein), Y (in R ^ m x 1) zwei numpy Arrays als Matrizen """
    xL = []
    for k in range(X.shape[0]):
        record = X[k].tolist()
        record.append(Y[k][0])
        xL.append(record)
    plot_xL_3d(xL, x3dlabels=("x0", "x1", "z"))


def simpel_xyystar_plot(X, Y, Ystar, titel_str=""):
    import matplotlib.pyplot as plt
    n = X.shape[1]
    fig, ax = plt.subplots(n, figsize=(12, 6))
    for k in range(n):
        _ax = ax if n == 1 else ax[k]
        _ax.scatter(x=X[:, k], y=Y, c="blue", s=4)
        _ax.scatter(x=X[:, k], y=Ystar, c="red", s=4)
        _ax.set_xlabel(f"X[:,{k}]")
        _ax.set_ylabel("Y (blau), Y* (rot)")
        _ax.grid(visible=True)
    plt.title(f"{titel_str}")
    plt.tight_layout()
    plt.show()
