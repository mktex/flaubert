
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api
from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.stats.power import NormalIndPower

from flaubert.model.outliers import sklearn_detect as sklof
from flaubert.statistik import stat_op

"""
from importlib import reload
from flaubert.statistik import abtest
reload(abtest)
"""

praezision = 5  # Dezimalzahlen
THRESHOLD_MIN_ANZAHL_DATENSAETZE = 10

"""
TODOS:
    -- Statistische Tests in einer Tabelle anzeigen: parametrische, non-parametrische: shapiro, wilcox, kruskal, kstest
    -- multivariate statistik
    -- varianz: anova, usw.

    Daten in Format: 
        |variante: {Variante A, Variante B} | werte {Anteile}|

    Beispiel Format: 
    >>> xdata.sample(10)
            variante    werte
    1307  Variante A  0.160000
    2936  Variante A  0.280000
    9650  Variante A  0.125000

    Zur Zeit beinhaltet das hier nur Auswertung für einen theoretischen ABTest zwischen zwei Variablen,
    die Anteile repräsentieren. So können hier Werte wie Spanne pro Warenkorb reingehen.

    Zusätzliche Dokumentation:
    Sehr gut ausgebauter AB/Test in Python: http://ethen8181.github.io/machine-learning/ab_tests/frequentist_ab_test.html

"""

# Ich checke ob Bayes oder einfache Abschätzung der Durchschnitt bessere Resultate liefern könnte
xfGetRVS = lambda sL, xsize: pd.Series([x for x in scipy.stats.norm.rvs(loc=np.round(np.array(sL).mean(), 2),
                                                                        scale=np.round(np.array(sL).std(), 2),
                                                                        size=xsize)])
xfJitter = lambda sL: pd.Series(
    [sL[k] + scipy.stats.norm.rvs(loc=0.0, scale=np.round(0.001 * np.array(sL).mean(), 2), size=1)[0] for k in
     range(len(sL))])


def info():
    print("[x] Beispiel Verwendung:")
    print("""    
    # definiere xdata mit 2 Features: "variante" und "werte"
    if True:
        reload(abtest)
        abt = abtest.ABTest(xdata)
        abt.ausreisser_entfernung(showPlots=True)
        xmeans_a, xmeans_b, xdictDurchschnitte = abt.get_durchschnitte_zwei_varianten()    
        abt.get_diff_stichprobe_methodik()
        abt.test_staerke_stichprobengroesse(alpha=0.05, beta=0.2)
        
        oder:
            abt = abtest.ABTest(xdata, 46851, 47346); abt.do_pipe_abtest()
    """)


class ABTest():
    konfidenz_intervall_dict = {}

    def __init__(self, xdata, n_a=None, n_b=None):
        """
        :param xdata: in Form |variante|werte|
        :param n_a: SUM(Variante A) oder MEAN(Variante A) als Größenordnung der Events
        :param n_b: SUM(Variante B) oder MEAN(Variante B) als Größenordnung der Events
        """
        self.setup_xdata(xdata)
        if n_a is None: n_a = xdata[xdata["variante"] == "Variante A"].shape[0]
        if n_b is None: n_b = xdata[xdata["variante"] == "Variante B"].shape[0]
        self.set_Ns(n_a, n_b)

    def setup_xdata(self, xdt):
        self.xdata = xdt

    def set_Ns(self, n_a, n_b):
        """
        :param n_a: Größenordnung für Gruppe A, SUM(A) oder MEAN(A)
        :param n_b: Größenordnung für Gruppe B, SUM(B) oder MEAN(B)
        :return:
        """
        self.n_a = n_a
        self.n_b = n_b

    def show_boxplot(self):
        # Diagram der Boxplots
        self.xdata[["variante", "werte"]].boxplot(by="variante", grid=False)
        plt.title("Boxplot der Werte nach Variante")
        plt.suptitle("")
        plt.show()

    def show_histogramme(self):
        # Diagramm der Histogramme
        self.xdata[self.xdata['variante'] == "Variante A"].werte.hist(grid=False, bins=50, color="cyan")
        self.xdata[self.xdata['variante'] == "Variante B"].werte.hist(grid=False, bins=50, color="red", alpha=0.5)
        plt.title("Variante A (Zyan); Variante B (Rot)")
        plt.show()

    def ausreisser_entfernung(self, disable=False, showPlots=True):
        """ stellt sicher, dass die Histogramme keine Ausreißern aufweisen """
        self.xdata_a = self.xdata[self.xdata['variante'] == "Variante A"]["werte"]
        self.xdata_b = self.xdata[self.xdata['variante'] == "Variante B"]["werte"]

        def ausreisser_ersatz(xs, ausreisser_lof):
            xdata_selekt = xs.copy()
            xdata_selekt = xdata_selekt.reset_index(drop=True)
            if ausreisser_lof.shape[0]:
                for ik in range(0, xs.shape[0]):
                    if ik not in [0, xs.shape[0] - 1]:
                        x1 = xdata_selekt.at[ik - 1]
                        x2 = xdata_selekt.at[ik + 1]
                        xdata_selekt.at[ik] = (x1 + x2) / 2.0
            else:
                xdata_selekt = xs.copy()
            xdata_selekt.index = xs.index
            return xdata_selekt

        if disable:
            self.xdata_b_non_outlier = self.xdata_b
            self.xdata_a_non_outlier = self.xdata_a
            return

        print("\n[x] Ausreißer Variante A:")
        xdata_a_outliers = sklof.ausreisser_lof(self.xdata_a)
        print(xdata_a_outliers)
        self.xdata_a_non_outlier = ausreisser_ersatz(self.xdata_a, xdata_a_outliers)

        xdata_b_outliers = sklof.ausreisser_lof(self.xdata_b)
        print("\n[x] Ausreißer Variante B:")
        print(xdata_b_outliers)
        self.xdata_b_non_outlier = ausreisser_ersatz(self.xdata_b, xdata_b_outliers)

        # Histogramm der Ausreißer
        if showPlots:
            self.xdata_a_non_outlier.hist(grid=False, bins=50, color="cyan")
            self.xdata_b_non_outlier.hist(grid=False, bins=50, color="red", alpha=0.5)
            plt.title("Variante A (Zyan); Variante B (Rot):"
                      "\nnach Ausreißer-Ersatz")
            plt.show()

        # Boxplot der Non-Ausreißer vs Ausreißer
        d1 = pd.DataFrame({
            "variante": ["Variante A"] * self.xdata_a_non_outlier.shape[0],
            "werte": self.xdata_a_non_outlier
        })
        d2 = pd.DataFrame({
            "variante": ["Variante B"] * self.xdata_b_non_outlier.shape[0],
            "werte": self.xdata_b_non_outlier
        })
        self.xdata_non_outlier = pd.concat([d1, d2])
        if showPlots:
            self.xdata_non_outlier[["variante", "werte"]].boxplot(by="variante", grid=False)
            plt.title("Boxplot der Werte nach Variante (OHNE Ausreißern)")
            plt.suptitle("")
            plt.show()

        print("\n[x] Beschreibung der Varianten (ohne Ausreißern)")
        print("[x] Variante A:")
        print(self.xdata_a_non_outlier.describe())
        print("[x] Variante B:")
        print(self.xdata_b_non_outlier.describe())

    def get_durchschnitte_zwei_varianten(self):
        """ Berechnet Durchschnitte der zwei Varianten und liefert die Konfidenzintervalle """
        global THRESHOLD_MIN_ANZAHL_DATENSAETZE
        print("==================================================================\n")
        if True:
            if self.xdata_a_non_outlier.shape[0] > THRESHOLD_MIN_ANZAHL_DATENSAETZE:
                xnA = int(0.2 * self.xdata_a_non_outlier.shape[0])
                xnB = int(0.2 * self.xdata_b_non_outlier.shape[0])
                nA = min(self.xdata_a_non_outlier.shape[0], 1000)
                nB = min(self.xdata_b_non_outlier.shape[0], 1000)
                self.xmeans_a = [self.xdata_a_non_outlier.sample(xnA, replace=True).mean() for x in range(nA)]
                self.xmeans_b = [self.xdata_b_non_outlier.sample(xnB, replace=True).mean() for x in range(nB)]
            else:
                print("[x] WARNUNG. Nicht implementierten Fall, zu wenige Daten.")
                xdictDurchschnitte = {"A": [self.xdata_a_non_outlier.mean()] * 2 + [1.0],
                                      "B": [self.xdata_b_non_outlier.mean()] * 2 + [1.0]}
                return [self.xdata_a_non_outlier.mean()], [self.xdata_b_non_outlier.mean()], xdictDurchschnitte
            # Wenn das korrekt war, sind die Durchschnitte normalverteilt
            xdictDurchschnitte = {}
            for x in [("A", self.xmeans_a), ("B", self.xmeans_b)]:
                xK, xmeans = x
                _, pwertA = scipy.stats.shapiro(xmeans)
                print("[x] Durchschnitt in Variante " + xK + ": ", np.round(np.mean(xmeans), praezision))
                print("[x] P-Wert des Shapiro-Tests für Durchschnitte Variante " + xK + ": ",
                      np.round(pwertA, praezision))
                xlowA, xhighA = statsmodels.stats.api.DescrStatsW(xmeans).tconfint_mean(alpha=0.05,
                                                                                        alternative="two-sided")
                print("[x] => 95% CI: [" + str(np.round(xlowA, praezision)) + ", " + str(
                    np.round(xhighA, praezision)) + "]")
                print()
                xdictDurchschnitte[x[0]] = [np.round(xlowA, praezision), np.round(xhighA, praezision),
                                            np.round(pwertA, praezision)]
        print("==================================================================")
        return self.xmeans_a, self.xmeans_b, xdictDurchschnitte

    def diff_methodik_bootstrap(self):
        print("\n\n==================================================================")

        # Methodik - Vermittlung der Differenz der Durchschnitte
        xsSampleA = min(1000, self.xdata_a_non_outlier.shape[0])
        xsSampleB = min(xsSampleA, self.xdata_b_non_outlier.shape[0])
        xL = []
        for i in range(10000):
            xL.append(
                [x[1] - x[0] for x in
                 zip(self.xdata_a_non_outlier.sample(xsSampleA), self.xdata_b_non_outlier.sample(xsSampleB))])

        xres = pd.Series([np.mean(x) for x in xL])
        # Wenn's alles gut gelaufen ist, ist xres normalverteilt (da er durchschnittliche Durchschnitt der Diff zeigt)
        _, pwert = scipy.stats.shapiro(xres)  # scipy.stats.normaltest(xres)
        xskew = scipy.stats.skew(xres.values)
        if pwert > 0.05:
            print("[x] OK. Unter Annahme p-Wert 5% sind die Differenzen der Durchschnitte normalverteilt.")
        else:
            print("[x] ACHTUNG: unter Annahme 5% sind die Differenzen der Durschnitte NICHT normalverteilt")
            print("[x] P-Wert liegt bei:", np.round(pwert, praezision))
            if np.abs(xskew) <= 0.5:
                print("[x] Heteroskädastizität akzeptabel (< 0.5):")
            else:
                print(
                    "[x] WARNUNG: Eine Vermittlung des korrekten Wertes "
                    "'Differenz der Durchschnitte' ist mit der aktuellen Methode nicht möglich!'")
                print("[x] Heteroskädastizität ist nicht in Intervall [-0.5, 0.5]:", xskew)
                print("[x] Histogramm: ")
                xres.hist(grid=False, bins=50)
                plt.show()
        print()
        print("[x] Differenz zwischen A und B:", np.round(xres.mean(), praezision))
        print("[x] 95% Konfidenzintervall für Durchschnitt der Differenz (vermittelt mit statsmodels):")
        xLowMean, xHighMean = statsmodels.stats.api.DescrStatsW(xres).tconfint_mean(alpha=0.05, alternative="two-sided")
        print("[x] [" + str(np.round(xLowMean, praezision)) + ", " + str(np.round(xHighMean, praezision)) + "]")
        print()
        self.bootstrap_means = xres
        self.konfidenz_intervall_dict["bootstrap_intervall"] = [xLowMean, xHighMean]

    def konfidenz_intervalle(self):
        # Methode der Berechnung der Konfidenzintervalle
        print(
            """
        CI 95% für Durchschnitt der Grundgesamtheit ist: mu +/- (zScore * SEM)
        SEM - standard error of the mean -> scipy.stats.sem oder auch xStd / sqrt(N) (standard deviation)
                Für die Auswertung der P1 vs P2 wird eine Binomial-Verteilung angenommen.
                So ist die Standard-Error aus der Formel: SE = (p1*(1-p1) / n1) + (p2 * (1 - p2) / n2)
        zScore - Z-Critical value (number of standard deviations to capture the 95% intervall) -> scipy.stats.norm.ppf
        mu - Mean of samples
        zScore * SEM - margin of error        
            """
        )
        xstd = self.bootstrap_means.std()
        xmean = self.bootstrap_means.mean()
        print("[x] Vermittlung mit scipy.stats.norm.intervall (Intervall 95%, "
              "wo die Durchschnitt der Grundgesamtheit fallen könnte):")
        xL, xH = scipy.stats.norm.interval(0.95, loc=np.mean(self.bootstrap_means),
                                           scale=scipy.stats.sem(self.bootstrap_means))
        print("[x] Intervall 95% [" + str(np.round(xL, praezision)) + ", " + str(np.round(xH, praezision)) + "]")
        print()

        print("[x] Vermittlung mit scipy.stats.norm.intervall: (95% Intervall, wo die Differenzen liegen):")
        xLowMeanStichprobe, xHighMeanStichprobe = scipy.stats.norm.interval(0.95, loc=xmean, scale=xstd)
        print("[x] [" + str(np.round(xLowMeanStichprobe, praezision)) + ", " + str(
            np.round(xHighMeanStichprobe, praezision)) + "]")
        xp = scipy.stats.norm.cdf(0.0, loc=xmean, scale=xstd)

        print("[x] Es gibt eine Wahrscheinlichkeit von %.2f" % np.round(xp * 100, 2)
              + "%, dass die Differenz kleiner als 0.0 ausfällt.")
        if xLowMeanStichprobe < 0 and xHighMeanStichprobe > 0:
            print("[x] ACHTUNG: hier fallen beiden Grenzwerte nicht ausschließlich "
                  "in positiven ODER negativen Bereich!")
            print("[x] Das bedeutet, dass der Resultat nicht eindeutig ist")
        m, m_low, m_high = stat_op.mean_confidence_interval(self.bootstrap_means)
        print()
        print("[x] Vermittlung Konfidenzintervall (mit scipy.stats.t.ppf)")
        print("[x] Durchschnitt: ", np.round(m, praezision))
        print("[x] Durchschnitt Low: ", np.round(m_low, praezision))
        print("[x] Durchschnitt High: ", np.round(m_high, praezision))
        print()

        self.konfidenz_intervall_dict["scipy.stats.norm.intervall"] = [np.round(xL, praezision),
                                                                       np.round(xH, praezision)]
        self.konfidenz_intervall_dict["scipy.stats.norm"] = [xLowMeanStichprobe, xHighMeanStichprobe]
        self.konfidenz_intervall_dict["scipy.stats.t.ppf"] = [np.round(m_low, praezision), np.round(m_high, praezision)]

    def plot_hist_diff(self):
        xLowMean, xHighMean = self.konfidenz_intervall_dict["bootstrap_intervall"]
        xLowMeanStichprobe, xHighMeanStichprobe = self.konfidenz_intervall_dict["scipy.stats.norm"]
        xp = scipy.stats.norm.cdf(0.0, loc=self.bootstrap_means.mean(), scale=self.bootstrap_means.std())
        plt.figure(figsize=(15, 5))
        ax = self.bootstrap_means.hist(grid=False, bins=100, color="cyan")
        ax.axvline(x=self.bootstrap_means.mean(), color='gray', linestyle='-')
        ax.axvline(x=xLowMean, color='gray', linestyle=':')
        ax.axvline(x=xHighMean, color='gray', linestyle=':')
        ax.axvline(x=xLowMeanStichprobe, color='gray', linestyle=':')
        ax.axvline(x=xHighMeanStichprobe, color='gray', linestyle=':')
        plt.text(1.1 * self.bootstrap_means.mean(), 20, "mu=" + str(np.round(self.bootstrap_means.mean(), praezision)))
        plt.text(xLowMeanStichprobe, 20, str(np.round(xLowMeanStichprobe, praezision)))
        plt.text(xHighMeanStichprobe, 20, str(np.round(xHighMeanStichprobe, praezision)))
        plt.text(0.0, 40, "P(mu<0):" + str(np.round(xp * 100, 2)) + "%")
        plt.title("Histogramm der Differenzen in Durchschnitte zwischen beiden Varianten")
        plt.show()

    def test_signifikanz(self):
        print("<pre>")
        print("[x] Vermittlung der Statistik Z-Score für die aggregierte Werte")
        n1 = self.n_a  # xres_counts.loc["Variante A"].werte + 0.0
        s1 = int(np.mean(self.xmeans_a) * n1)
        n2 = self.n_b  # xres_counts.loc["Variante B"].werte + 0.0
        s2 = int(np.mean(self.xmeans_b) * n2)
        p1 = np.round(np.mean(self.xmeans_a), praezision)
        p2 = np.round(np.mean(self.xmeans_b), praezision)
        p = np.round((s1 + s2) / (n1 + n2), praezision)
        z = (p1 - p2) / (p * (1 - p) * ((1 / n1) + (1 / n2))) ** 0.5
        self.p1 = p1
        self.p2 = p2
        alternative = 'smaller' if p1 < p2 else 'larger'
        print("Proportions Z-Test:", s1, s2, n1, n2, alternative)
        zScore, pval = statsmodels.api.stats.proportions_ztest([s1, s2], [n1, n2], alternative=alternative)
        print("\n[x] Resultat proportions_ztest():")
        print(('[x] Manueller Berechnung Z: {:.6f}'.format(z)))
        print(('[x] Z-score statsmodels: {:.6f}'.format(zScore)))
        print(('[x] Statsmodels pWert: {:.6f}'.format(pval)))
        if pval > 0.05:
            print("********************************************************************************************")
            print("[x] P-Wert: Unter Annahme p-Wert 5% kann hier die Nullhypothese NICHT wiederlegt werden.")
            print(
                "[x] D.h.: Aufgrund der Beweise aus aggregierten Daten besteht keine statistisch signifikante Differenz.")
            print("********************************************************************************************")
        zscore, pwert = stat_op.two_proportions_test(s1, n1, s2, n2)
        prop_diff, confint = stat_op.two_proportions_confint(p1, n1, p2, n2)
        print("\n[x] Resultat two_proportions_test(), two_proportions_confint():")
        print("[x] Differenz: ", np.round(prop_diff, praezision))
        print("[x] Konfidenzintervall: ", confint)
        print("[x] Z-Score:", zscore)
        print("[x] P-Wert:", pwert)
        print("==================================================================")
        print("</pre>")

    def get_diff_stichprobe_methodik(self):
        self.diff_methodik_bootstrap()
        self.konfidenz_intervalle()
        self.plot_hist_diff()
        self.test_signifikanz()

    def test_staerke_stichprobengroesse(self, alpha=0.05, beta=0.2):
        """
            Berechnung der Stichprobengröße, sodass statistisch signifikante Unterschiede messbar werden (wenn Sie gibt)
            Nutzung:
                A. Gibt es zwei P1 und P2 aus einem AB Test schon? Dann zeigt das hier, ob die Größe des Datenbestandes,
                   für die Auswertung ausreicht,
                   so dass eine Differenz P1 - P2 statistisch signifikant ist (wenn sie tatsächlich gibt)
                B. Stellt sich die Frage, wie hoch, bzw. wie lange ein Experiment laufen sollte, damit P1 und P2 = P1 + Delta(P)
                    Nachweisbar wäre. Dann kann man diese Methode nutzen, um die Dauer des Experimentes festzustellen.

            Teststärke und Falsch-Positiven Ergebnisse:
            Quelle: http://ethen8181.github.io/machine-learning/ab_tests/frequentist_ab_test.html
                "Significance level: Governs the chance of a false positive. A significance level of 0.05 means that
                there is a 5% chance of a false positive. Choosing level of significance is an arbitrary task,
                but for many applications, a level of 5% is chosen, for no better reason than that it is conventional
                Statistical power Power of 0.80 means that there is an 80% chance that if there was an effect,
                we would detect it (or a 20% chance that we'd miss the effect). In other words, power is equivalent to  1−β.
                There are no formal standards for power,
                most researchers assess the power of their tests using 0.80 for adequacy"
        :return:
        """
        p1 = self.p1
        p2 = self.p2
        if p1 > 1 and p2 > 1:
            p1 = p1 / (p1 + p2 + 0.0)
            p2 = 1.0 - p1

        print("\n\n==================================================================")
        print("[x] Benötigte Anzahl der Datensätze, um Signifikanz zu entdecken (pro Gruppe) feststellen zu können: \n")

        print("[x] KURZINFO Cohen's d")
        print("::   kleiner Effekt      | d | = 0,2")
        print("::   mittlerer Effekt    | d | = 0,5")
        print("::   großer Effekt       | d | = 0,8")

        print("[x] IST-Situation .. ")
        self.xdata.groupby("variante").count()

        print("[x] Vermittlung mit scipy.stats.norm.isf:")
        XN = stat_op.sample_power_probtest(p1, p2, power=1 - beta, sig=alpha)
        print("[x]", XN, "(pro Gruppe)\n")
        # sample_power_difftest(0.01, 0.4, power=0.8, sig=0.05)

        # Definition effect size: https://tedboy.github.io/statsmodels_doc/doc/generated/statsmodels.stats.power.zt_ind_solve_power.html?highlight=zt_ind_solve_power
        print("[x] Definition:")
        print("==== standardized effect size, difference between the two means divided by the standard deviation. ====")
        print()
        print("[x] Vermittlung der Effektstärke mit statsmodels .. ")
        es = statsmodels.stats.api.proportion_effectsize(p1, p2)
        alternative = 'smaller' if p1 < p2 else "larger"

        print("[x] Vermittlung mit statsmodels.stats.api.tt_ind_solve_power:")
        print("[x]",
              int(statsmodels.stats.api.tt_ind_solve_power(effect_size=es, nobs1=None, alpha=alpha, power=1 - beta,
                                                           ratio=1, alternative=alternative)),
              "(pro Gruppe)\n")

        print("[x] Vermittlung mit statsmodels.stats.api.zt_ind_solve_power:")
        print("[x]", int(statsmodels.stats.api.zt_ind_solve_power(es, None, alpha, 1 - beta, alternative=alternative)),
              "(pro Gruppe)")

        print("[x] Vermittlung mit statsmodels.stats.power.TTestIndPower():")
        xpower = statsmodels.stats.power.TTestIndPower()
        print("[x]", int(xpower.solve_power(es, None, alpha, 1 - beta, alternative=alternative)),
              "(pro Gruppe)\n")

        print("[x] Vermittlung mit statsmodels.stats.power.NormalIndPower():")
        print("[x]",
              int(NormalIndPower().solve_power(es, None, alpha=alpha, power=(1 - beta), alternative=alternative)),
              "(pro Gruppe)\n")

        print("[x] Vermittlung mit analytischer Lösung:")
        print("[x]", int(stat_op.experiment_size(p1, p2, alpha=alpha, beta=beta)),
              "(pro Gruppe)\n")

        # Cohen's d, Quelle https://matheguru.com/stochastik/effektstarke.html
        print("[x] Vermittlung Cohen's d:")
        c0 = self.xdata_a_non_outlier.values
        c1 = self.xdata_b_non_outlier.values
        cohens_d = (np.mean(c0) - np.mean(c1)) / (np.sqrt((np.std(c0) ** 2 + np.std(c1) ** 2) / 2))
        print("[x]", np.abs(cohens_d), "\n")

        # Plot Power
        print("[x] Diagramm der Effektstärken: {20% weniger, 100%, 20% stärker}")
        effect_sizes = np.array([es * (1 - beta), es, es * (1 + beta)])
        print("[x] Effektstärken: ", effect_sizes)
        sample_sizes = np.arange(int(0.1 * XN), int(1.5 * XN), int(0.1 * XN))
        stat_op.plot_power(xpower, dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes, precision=praezision)
        plt.title("Diagramm Test Signifikanz-Stärke vs Stichprobengrößen")
        plt.show()

    def do_pipe_abtest(self, disable_outlier_detect=False):
        self.ausreisser_entfernung(disable=disable_outlier_detect, showPlots=True)
        _ = self.get_durchschnitte_zwei_varianten()
        self.get_diff_stichprobe_methodik()
        self.test_staerke_stichprobengroesse(alpha=0.05, beta=0.2)