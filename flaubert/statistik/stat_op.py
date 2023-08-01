
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import statsmodels as sm

from flaubert import einstellungen

# Funnel Beispiel
# anzahl_zeiteinheiten(3250, 520, 570, 0.025, 0.2)
anzahl_zeiteinheiten = lambda N1, N2, N2_star, alpha, beta: \
    experiment_size(p_null=N2 / N1, p_alt=N2_star / N1, alpha=(alpha / 2.0), beta=.20) / N1 * 2

features_in_df = lambda xdf_input, xcol_target, xcol_id: [x for x in xdf_input.columns.values if
                                                          x not in [xcol_target, xcol_id]]


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def two_proportions_test(success_a, size_a, success_b, size_b):
    """
    # Quelle: http://ethen8181.github.io/machine-learning/ab_tests/frequentist_ab_test.html
    A/B test for two proportions
    given a success a trial size of group A and B compute
    its zscore and pvalue

    Parameters
    ----------
    success_a, success_b : int
        Number of successes in each group

    size_a, size_b : int
        Size, or number of observations in each group

    Returns
    -------
    zscore : float
        test statistic for the two proportion z-test

    pvalue : float
        p-value for the two proportion z-test

    ACHTUNG: das hier ist gleich statsmodels.api.stats.proportions_ztest
    """
    prop_a = success_a / size_a
    prop_b = success_b / size_b
    prop_pooled = (success_a + success_b) / (size_a + size_b)
    var = prop_pooled * (1 - prop_pooled) * (1 / size_a + 1 / size_b)
    zscore = np.abs(prop_b - prop_a) / np.sqrt(var)
    one_side = 1 - scipy.stats.norm(loc=0, scale=1).cdf(zscore)
    pvalue = one_side * 2
    return zscore, pvalue


def two_proportions_confint(success_a, size_a, success_b, size_b, significance=0.05, rnormstd=0.05):
    """
    Quelle: http://ethen8181.github.io/machine-learning/ab_tests/frequentist_ab_test.html
    A/B test for two proportions
    given a success a trial size of group A and B compute its confidence interval.
    the resulting confidence interval matches R's prop.test function.

    Parameters
    ----------
    success_a, success_b : int
        Number of successes in each group

    size_a, size_b : int
        Size, or number of observations in each group

    significance : float, default 0.05
        Often denoted as alpha. Governs the chance of a false positive.
        A significance level of 0.05 means that there is a 5% chance of
        a false positive. In other words, our confidence level is
        1 - 0.05 = 0.95

    Returns
    -------
    prop_diff : float
        Difference between the two proportion

    confint : 1d ndarray
        Confidence interval of the two proportion test
    """
    prop_a = success_a / size_a
    prop_b = success_b / size_b
    var = prop_a * (1 - prop_a) / size_a + prop_b * (1 - prop_b) / size_b
    se = np.sqrt(var)
    # z critical value
    confidence = 1 - significance
    z = scipy.stats.norm(loc=0, scale=rnormstd).ppf(confidence + significance / 2)
    # standard formula for the confidence interval
    # point-estimtate +- z * standard-error
    prop_diff = prop_b - prop_a
    confint = prop_diff + np.array([-1, 1]) * z * se
    return prop_diff, confint


def sample_power_probtest(p1, p2, power=0.8, sig=0.05, alternative="one-sided"):
    """
        https://stackoverflow.com/questions/15204070/is-there-a-python-scipy-function-to-determine-parameters-needed-to-obtain-a-ta
    """
    from scipy.stats import norm
    if alternative == "one-sided":
        z = norm.isf([sig])
    else:
        z = norm.isf([sig / 2])  # two-sided t test
    zp = -1 * norm.isf([power])
    d = (p1 - p2)
    s = 2 * ((p1 + p2) / 2) * (1 - ((p1 + p2) / 2))
    n = s * ((zp + z) ** 2) / (d ** 2)
    return int(np.round(n[0]))


def sample_power_difftest(d, s, power=0.8, sig=0.05, alternative="one-sided"):
    """
    :param d: Differenz der Durchschnitte
    :param s: Standardabweichung
    :return:
    """
    from scipy.stats import norm
    if alternative == "one-sided":
        z = norm.isf([sig])
    else:
        z = norm.isf([sig / 2])
    zp = -1 * norm.isf([power])
    n = (2 * (s ** 2)) * ((zp + z) ** 2) / (d ** 2)
    return int(np.round(n[0]))


def plot_power(powerObj=None, dep_var='nobs', nobs=None, effect_size=None,
               alpha=0.05, ax=None, title=None, precision=2, plt_kwds=None, **kwds):
    """
        # aus dem Quellcode statsmodels
        # wegen benötigte Anpassungen übernommen
    """
    from statsmodels.graphics import utils
    from statsmodels.graphics.plottools import rainbow
    fig, ax = utils.create_mpl_ax(ax)
    import matplotlib.pyplot as plt
    colormap = plt.cm.Paired
    plt_alpha = 1
    lw = 2
    if dep_var == 'nobs':
        colors = rainbow(len(effect_size))
        colors = [colormap(i) for i in np.linspace(0, 1.0, len(effect_size))]
        for ii, es in enumerate(effect_size):
            power = powerObj.power(es, nobs, alpha, **kwds)
            ax.plot(nobs, power, lw=lw, alpha=plt_alpha,
                    color=colors[ii], label=('es=%.' + str(precision) + 'f') % es)
            xlabel = 'Number of Observations'
    elif dep_var in ['effect size', 'effect_size', 'es']:
        colors = rainbow(len(nobs))
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(nobs))]
        for ii, n in enumerate(nobs):
            power = powerObj.power(effect_size, n, alpha, **kwds)
            ax.plot(effect_size, power, lw=lw, alpha=plt_alpha,
                    color=colors[ii], label=('N=%.' + str(precision) + 'f') % n)
            xlabel = 'Effect Size'
    elif dep_var in ['alpha']:
        # experimental nobs as defining separate lines
        colors = rainbow(len(nobs))
        for ii, n in enumerate(nobs):
            power = powerObj.power(effect_size, n, alpha, **kwds)
            ax.plot(alpha, power, lw=lw, alpha=plt_alpha,
                    color=colors[ii], label=('N=%.' + str(precision) + 'f') % n)
            xlabel = 'alpha'
    else:
        raise ValueError('depvar not implemented')
    if title is None:
        title = 'Power of Test'
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc='lower right')
    return fig


def power(p_null, p_alt, n, alpha=.05, plot=True):
    """
    *** Quelle: Udacity ***
    Compute the power of detecting the difference in two populations with
    different proportion parameters, given a desired alpha rate.

    Input parameters:
        p_null: base success rate under null hypothesis
        p_alt : desired success rate to be detected, must be larger than
                p_null
        n     : number of observations made in each group
        alpha : Type-I error rate
        plot  : boolean for whether or not a plot of distributions will be
                created

    Output value:
        power : Power to detect the desired difference, under the null.
    """
    import matplotlib.pyplot as plt
    # Compute the power
    se_null = np.sqrt((p_null * (1 - p_null) + p_null * (1 - p_null)) / n)
    null_dist = scipy.stats.norm(loc=0, scale=se_null)
    p_crit = null_dist.ppf(1 - alpha)

    se_alt = np.sqrt((p_null * (1 - p_null) + p_alt * (1 - p_alt)) / n)
    alt_dist = scipy.stats.norm(loc=p_alt - p_null, scale=se_alt)
    beta = alt_dist.cdf(p_crit)

    if plot:
        # Compute distribution heights
        low_bound = null_dist.ppf(.01)
        high_bound = alt_dist.ppf(.99)
        x = np.linspace(low_bound, high_bound, 201)
        y_null = null_dist.pdf(x)
        y_alt = alt_dist.pdf(x)

        # Plot the distributions
        plt.plot(x, y_null)
        plt.plot(x, y_alt)
        plt.vlines(p_crit, 0, np.amax([null_dist.pdf(p_crit), alt_dist.pdf(p_crit)]),
                   linestyles='--')
        plt.fill_between(x, y_null, 0, where=(x >= p_crit), alpha=.5)
        plt.fill_between(x, y_alt, 0, where=(x <= p_crit), alpha=.5)

        plt.legend(['null', 'alt'])
        plt.xlabel('difference')
        plt.ylabel('density')
        plt.show()

    # return power
    return (1 - beta)


def experiment_size(p_null, p_alt, alpha=.05, beta=.20):
    """
    *** Quelle: Udacity ***
    Beispiel:
        Funnel: N1 -> N2 -> N3;
        Anzahl zeiteinheiten:
            experiment_size(p_null=N2/N1, p_alt=N2_Neu/N1, alpha=(0.025/2.0), beta=.20) / N1 * 2

    Compute the minimum number of samples needed to achieve a desired power
    level for a given effect size.

    Input parameters:
        p_null: base success rate under null hypothesis
        p_alt : desired success rate to be detected
        alpha : Type-I error rate
        beta  : Type-II error rate

    Output value:
        n : Number of samples required for each group to obtain desired power
    """

    # Get necessary z-scores and standard deviations (@ 1 obs per group)
    z_null = scipy.stats.norm.ppf(1 - alpha)
    z_alt = scipy.stats.norm.ppf(beta)
    sd_null = np.sqrt(p_null * (1 - p_null) + p_null * (1 - p_null))
    sd_alt = np.sqrt(p_null * (1 - p_null) + p_alt * (1 - p_alt))

    # Compute and return minimum sample size
    p_diff = p_alt - p_null
    n = ((z_null * sd_null - z_alt * sd_alt) / p_diff) ** 2
    return np.ceil(n)


def quantile_ci(data, q, c=.95, n_trials=1000):
    """
    *** Quelle: Udacity ***
    Compute a confidence interval for a quantile of a dataset using a bootstrap
    method.

    Input parameters:
        data: data in form of 1-D array-like (e.g. numpy array or Pandas series)
        q: quantile to be estimated, must be between 0 and 1
        c: confidence interval width
        n_trials: number of bootstrap samples to perform

    Output value:
        ci: Tuple indicating lower and upper bounds of bootstrapped
            confidence interval

    quantile_permtest(data['time'], data['condition'], 0.9, alternative = 'less')
    """

    # initialize storage of bootstrapped sample quantiles
    n_points = data.shape[0]
    sample_qs = []

    # For each trial...
    for _ in range(n_trials):
        # draw a random sample from the data with replacement...
        sample = np.random.choice(data, n_points, replace=True)

        # compute the desired quantile...
        sample_q = np.percentile(sample, 100 * q)

        # and add the value to the list of sampled quantiles
        sample_qs.append(sample_q)

    # Compute the confidence interval bounds
    lower_limit = np.percentile(sample_qs, (1 - c) / 2 * 100)
    upper_limit = np.percentile(sample_qs, (1 + c) / 2 * 100)

    return (lower_limit, upper_limit)


def quantile_permtest(x, y, q, alternative='less', n_trials=10000):
    """
    *** Quelle: Udacity ***
    Compute a confidence interval for a quantile of a dataset using a bootstrap
    method.

    Input parameters:
        x: 1-D array-like of data for independent / grouping feature as 0s and 1s
        y: 1-D array-like of data for dependent / output feature
        q: quantile to be estimated, must be between 0 and 1
        alternative: type of test to perform, {'less', 'greater'}
        n_trials: number of permutation trials to perform

    Output value:
        p: estimated p-value of test
    """

    # initialize storage of bootstrapped sample quantiles
    sample_diffs = []

    # For each trial...
    for _ in range(n_trials):
        # randomly permute the grouping labels
        labels = np.random.permutation(y)

        # compute the difference in quantiles
        cond_q = np.percentile(x[labels == 0], 100 * q)
        exp_q = np.percentile(x[labels == 1], 100 * q)

        # and add the value to the list of sampled differences
        sample_diffs.append(exp_q - cond_q)

    # compute observed statistic
    cond_q = np.percentile(x[y == 0], 100 * q)
    exp_q = np.percentile(x[y == 1], 100 * q)
    obs_diff = exp_q - cond_q

    # compute a p-value
    if alternative == 'less':
        hits = (sample_diffs <= obs_diff).sum()
    elif alternative == 'greater':
        hits = (sample_diffs >= obs_diff).sum()

    return (hits / n_trials)


def ranked_sum(x, y, alternative='two-sided'):
    """
    *** Quelle: Udacity ***
    Return a p-value for a ranked-sum test, assuming no ties.

    Input parameters:
        x: 1-D array-like of data for first group
        y: 1-D array-like of data for second group
        alternative: type of test to perform, {'two-sided', less', 'greater'}

    Output value:
        p: estimated p-value of test
    """

    # compute U
    u = 0
    for i in x:
        wins = (i > y).sum()
        ties = (i == y).sum()
        u += wins + 0.5 * ties

    # compute a z-score
    n_1 = x.shape[0]
    n_2 = y.shape[0]
    mean_u = n_1 * n_2 / 2
    sd_u = np.sqrt(n_1 * n_2 * (n_1 + n_2 + 1) / 12)
    z = (u - mean_u) / sd_u

    # compute a p-value
    if alternative == 'two-sided':
        p = 2 * scipy.stats.norm.cdf(-np.abs(z))
    if alternative == 'less':
        p = scipy.stats.norm.cdf(z)
    elif alternative == 'greater':
        p = scipy.stats.norm.cdf(-z)

    return p


def sign_test(x, y, alternative='two-sided'):
    """
    *** Quelle: Udacity ***
    Return a p-value for a ranked-sum test, assuming no ties.
    Input parameters:
        x: 1-D array-like of data for first group
        y: 1-D array-like of data for second group
        alternative: type of test to perform, {'two-sided', less', 'greater'}

    Output value:
        p: estimated p-value of test
    """

    # compute parameters
    n = x.shape[0] - (x == y).sum()
    k = (x > y).sum() - (x == y).sum()

    # compute a p-value
    if alternative == 'two-sided':
        p = min(1, 2 * scipy.stats.binom(n, 0.5).cdf(min(k, n - k)))
    if alternative == 'less':
        p = scipy.stats.binom(n, 0.5).cdf(k)
    elif alternative == 'greater':
        p = scipy.stats.binom(n, 0.5).cdf(n - k)

    return p


def get_confusion_matrix_stats(cm, i):
    """
        Given a Confusion Matrix cm, calculates precision, recall and F1 scores
        | true negatives  C_{0,0}  | false positivesC_{0,1} |
        | false negatives C_{1,0}  | true positives C_{1,1} |
    :param cm: confusion matrix
    :param i: position of the variable, for which the caculation be done
    :return: three statistics: precision, recall and the F1-Score
    """
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    tp = cm[1, 1]
    if i == 0:
        precision = tn / (tn + fn)
        recall = tn / (tn + fp)
    elif i == 1:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def apply_clf_model(input_x, clf, threshold=None, klass_k=None, labels_sorted=None):
    if labels_sorted is None: labels_sorted = clf.classes_
    if threshold is None:
        y_pred = np.array([labels_sorted[np.argmax(t)] for t in clf.predict_proba(input_x)])
    else:
        y_pred = (clf.predict_proba(input_x)[:, klass_k] >= threshold).astype(int)
    return y_pred


def stats_perf_clf(x, y, clf, threshold=None, klass_k=None, labels_sorted=None,
                   vis_confusion_mat=False):
    if labels_sorted is None: labels_sorted = clf.classes_
    input_x = x
    y_true = np.array(y)
    y_pred = apply_clf_model(input_x, clf, threshold, klass_k, labels_sorted)
    accuracy = accuracy_score(y_true, y_pred) # (y_pred == y_true).mean()
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    if vis_confusion_mat:
        print("\n[x] Visualisierung Konfusionsmatrix..")
        print("[x] Labels:", labels_sorted)
        print("| TN  | FP |")
        print("| FN  | TP |")
        print(cm)
        for i in range(0, len(labels_sorted)):
            precision, recall, f1_score = get_confusion_matrix_stats(cm, i)
            print("[x] Label {} - precision {}, recall {}, f1_score {}: ".format(
                i, np.round(precision, 2), np.round(recall, 2), np.round(f1_score, 2)
            ))
        print("[x] Accuracy:", accuracy)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
        disp.plot()
        plt.show()


def print_clf_statistics(xclf, features_train, labels_train, features_test, labels_test,
                         threshold=None, klass_k=None, vis_confusion_mat=False):
    if threshold is None:
        y_test_pred = xclf.predict(features_test)
        y_train_pred = xclf.predict(features_train)
    else:
        y_test_pred = apply_clf_model(features_test, xclf, threshold, klass_k)
        y_train_pred = apply_clf_model(features_train, xclf, threshold, klass_k)

    print("\n[x] Accuracy TRAIN und TEST:")
    print('[x] train:', accuracy_score(labels_train, y_train_pred))
    print('[x] test: ', accuracy_score(labels_test, y_test_pred))

    print('\n[x] Classification Report and Confusion Matrix (TRAIN):')
    print(classification_report(labels_train, y_train_pred))
    stats_perf_clf(features_train, labels_train, xclf, threshold, klass_k,
                   vis_confusion_mat=vis_confusion_mat)

    print('\n[x] Classification Report and Confusion Matrix (TEST):')
    print(classification_report(labels_test, y_test_pred))
    stats_perf_clf(features_test, labels_test, xclf, threshold, klass_k,
                   vis_confusion_mat=vis_confusion_mat)



def normalization(xSeries):
    xsIndx = xSeries.index
    xmin = xSeries.min()
    xmax = xSeries.max()
    xsRes = pd.Series([round(((x - xmin) / (xmax - xmin)), 3) for x in xSeries])
    xsRes.index = xsIndx
    return xsRes


def standardize(xSeries):
    xsIndx = xSeries.index
    xmean = xSeries.mean()
    xstd = xSeries.std(ddof=0)
    xsRes = pd.Series([x if np.isnan(x) else round(((x - xmean) / xstd), 3) for x in xSeries])
    xsRes.index = xsIndx
    return xsRes


def calculate_skewness(x):
    from scipy.stats import skew
    return skew(x)


def make_series_positiv(xs):
    xm = xs.min()
    if xm < 0.0:
        xs = xs + np.abs(xm)
    else:
        xm = 0.0
    return xs, xm


def make_df_neg2pos(xdfInput, xcolTarget, xcolID):
    """ verwendet make_series_positiv für alle Features außer ID und Target """
    xdfInput = xdfInput.copy()
    xdict_neg2pos = {}
    for xc in features_in_df(xdfInput, xcolTarget, xcolID):
        xdfInput[xc], xm = make_series_positiv(xdfInput[xc])
        xdict_neg2pos[xc] = xm
    return xdfInput, xdict_neg2pos


def list_cols_high_skew(xdfInput, xcolID=None, xcolTarget=None, xthreshold=0.8):
    xL = []
    for xcol in xdfInput:
        if xcol not in [xcolID, xcolTarget]:
            if np.abs(calculate_skewness(xdfInput[xcol].values)) > xthreshold:
                print(xcol)
                xL.append(xcol)
    return xL


def transform_series_by_sqrt(xSeries):
    return [1.0 / np.sqrt(x) for x in xSeries]


def transform_series_by_log(xSeries):
    return [np.log(x) for x in xSeries]


def transform_df_by_log(xdfInput, xepsilon=10 ** (-4)):
    xdfInput = xdfInput.copy()
    xlistSkew = list_cols_high_skew(xdfInput, "object_id", "target")
    xdict = {}
    for xc in xlistSkew:
        xm = xdfInput[xc].min()
        if xm <= 0:
            xdfInput[xc] = [x + np.abs(xm) + xepsilon for x in xdfInput[xc].values]
        xdfInput[xc] = transform_series_by_log(xdfInput[xc])
        xdict[xc] = [xm, xepsilon]
    return xdfInput, xdict


def simulation_ttest(mu1, sd1, mu2, sd2):
    from scipy import stats
    np.random.seed(12345678)
    rvs1 = scipy.stats.norm.rvs(loc=mu1, scale=sd1, size=500)
    rvs2 = scipy.stats.norm.rvs(loc=mu2, scale=sd2, size=500)
    # scipy.stats.ttest_ind(rvs1, rvs2)
    return scipy.stats.ttest_ind(rvs1, rvs2, equal_var=(sd1==sd2))


def ttest_mean(xseries):
    print(scipy.stats.normaltest(xseries))


def get_korrelationsk_pearson(xseries1, xseries2):
    print(pearsonr(xseries1, xseries2))


def get_corrlist(df, xcol):
    """ Ergibt Korrelationen für xcol zu allen anderen Features"""
    xr = df.corr()[xcol].sort_values()
    return xr


def median_diff(predicted, actual, sindKlassen=False):
    """ einfaches median-score """
    if not sindKlassen:
        return np.median(np.abs(predicted - actual))
    else:
        return np.mean(list([x[0] != x[1] for x in list(zip(predicted, actual))]))


def auswertung_ableitungen(y, doplot=False):
    import matplotlib.pyplot as plt
    from scipy.interpolate import splev, splrep, spalde
    import numpy as np
    import pandas as pd
    xres = [0.5, 0.5, 0.5]
    try:
        if len(y) <= 3 or len(set(y)) == 1:
            return xres
        x = np.linspace(0, len(y), len(y))
        spl = splrep(x, y, k=3)
        x2 = np.linspace(0, len(y), 1000)
        y2 = splev(x2, spl)
        xresdf = pd.DataFrame(spalde(x2, spl))
        xinklination1 = np.degrees([np.arctan(x) for x in xresdf[1].values])
        xinklination2 = np.degrees([np.arctan(x) for x in xresdf[2].values])
        xinklination3 = np.degrees([np.arctan(x) for x in xresdf[3].values])
        output = (xresdf > 0).sum() / xresdf.count()
        orientierung = xresdf > 0
        if doplot:
            print(xresdf)
            Positivitaet = "Positivität Ableitungen: {} (I), {} (II), {} (III)".format(*output.iloc[1:].tolist())
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].plot(x, y, 'o', x2, y2)
            axs[0, 0].set_title('Interpolation')
            axs[0, 1].scatter(x=x2, y=orientierung[1].astype(np.int).values, c='tab:orange', s=4)
            axs[0, 1].set_title('SIGN(Ableitung)')
            axs[1, 0].plot(x2, xinklination1, 'tab:green')
            axs[1, 0].set_title('Winkel (1)')
            axs[1, 1].plot(x2, xinklination2, 'tab:red')
            axs[1, 1].set_title('Winkel (2)')
            fig.suptitle(Positivitaet)
            plt.tight_layout()
            plt.show()
        xres = output.iloc[1:].tolist()
    except:
        import traceback
        traceback.print_exc()
    return xres


def kde_fit(xs, alpha=0.05, minWert=None, maxWert=None, doPlot=True):
    xresKDE = xs.astype(np.float64)
    kdeFit = sm.nonparametric.kde.KDEUnivariate(xresKDE)
    kdeFit.fit()
    if doPlot:
        fig = plt.figure(figsize=einstellungen.FIGSIZE_BREIT, dpi=einstellungen.FIGDPI_NIEDRIG)
        ax = fig.add_subplot(111)
        ax.hist(xresKDE, bins=45, density=True, label='Histogramm', zorder=5, edgecolor='k', alpha=0.5)
        ax.plot(kdeFit.support, kdeFit.density, lw=3, label='KDE', zorder=10)
        ax.legend(loc='best')
        ax.grid(True, zorder=-5)
        plt.show()
    # Ich brauche wissen die Werte Low/High mit Eigenschaft P(x < Low) <= alpha /2.0 und P(x>High) <= alpha / 2.0
    minWert = 0.8 * xs.min() if minWert is None else minWert
    maxWert = 1.2 * xs.max() if maxWert is None else maxWert
    xwerte_skala = np.linspace(minWert, maxWert, num=100)
    xp = np.array([kdeFit.evaluate(t)[0] for t in xwerte_skala])
    xp = xp / np.sum(xp)
    xL = xwerte_skala[np.where(np.cumsum(xp) >= alpha / 2.0)[0].min()]
    xH = xwerte_skala[np.where(np.cumsum(xp) >= (1.0 - alpha / 2.0))[0].min()]
    return kdeFit, xL, xH


def statistik_test(dfinput, xcol="Gender", werta="Male", wertb="Female",
                   target_col="Attrition"):
    """
        target muss binomial sein, d.h. 0 oder 1
    """
    n1 = dfinput[dfinput[xcol] == werta].shape[0]
    s1 = dfinput[dfinput[xcol] == werta][target_col].sum()
    p1 = s1 / n1
    n2 = dfinput[dfinput[xcol] == wertb].shape[0]
    s2 = dfinput[dfinput[xcol] == wertb][target_col].sum()
    p2 = s2 / n2
    print(f"[x] Statistische Maßen: s1={s1}, n1={n1}, p1={p1}, s2={s2}, n2={n2}, p2={p2}")
    zscore, pwert = two_proportions_test(s1, n1, s2, n2)
    prop_diff, confint = two_proportions_confint(p1, n1, p2, n2, rnormstd=0.05)
    print("\n[x] Resultat two_proportions_test(), two_proportions_confint():")
    print("[x] Differenz: ", np.round(prop_diff, 4))
    print("[x] Konfidenzintervall: ", confint)
    print("[x] Z-Score:", zscore)
    print("[x] p-Wert:", pwert)
    p = np.round((s1 + s2) / (n1 + n2), 4)
    z = (p1 - p2) / (p * (1 - p) * ((1 / n1) + (1 / n2))) ** 0.5
    alternative = 'smaller' if p1 < p2 else 'larger'
    zScore, pval = sm.api.stats.proportions_ztest([s1, s2], [n1, n2], alternative=alternative)
    print("[x] Resultat proportions_ztest():")
    print(('[x] Manueller Berechnung z-Score: {:.6f}'.format(z)))
    print(('[x] Z-score statsmodels: {:.6f}'.format(zScore)))
    print(('[x] Statsmodels p-Wert: {:.6f}'.format(pval)))
