import time
import traceback
import operator
import random
import sys, os
import json

import pandas as pd
import numpy as np
from scipy import stats
import scipy.spatial
from sklearn import metrics

from deap import algorithms, base, creator, tools, gp

WORSE_SCORE = 100000
GLOBAL_SCORE = 100000
NMAX_SAMPLES = 1000

# mit FLM.SILENT in xFitLine gesteuert
SILENT = True

import warnings

warnings.filterwarnings("ignore")

xpath_gpfolder = "./data/"


def getScore():
    global GLOBAL_SCORE
    global xpath_gpfolder
    try:
        with open(f'{xpath_gpfolder}/global_score.pckl'.replace("//", "/"), 'r') as f:
            xstr = f.read()
            GLOBAL_SCORE = float(xstr)
    except:
        pass
    GLOBAL_SCORE = GLOBAL_SCORE
    return GLOBAL_SCORE


def setScore(GS):
    global GLOBAL_SCORE
    global xpath_gpfolder
    with open(f'{xpath_gpfolder}/global_score.pckl'.replace("//", "/"), 'w') as f:
        f.write(str(GS))
    GLOBAL_SCORE = GLOBAL_SCORE


def pAvg(a, b):
    try:
        xres = (a + b) / 2.0
        return xres
    except:
        pass
    return 1.0


def add(a, b):
    try:
        return np.add(a, b)
    except:
        pass
    return 1


def sub(a, b):
    try:
        return np.subtract(a, b)
    except:
        pass
    return 1


def neg(a):
    return -a


def zero(a):
    return 0


def one(a):
    return 1


def log2(a):
    if a > 1:
        return np.log2(a)
    else:
        return 0


def genR(a1, a2):
    xtemp = np.arange(a1, a2, 0.05)
    return round(xtemp[random.randint(0, len(xtemp) - 1)], 2)


print("***************************************************************************************************************")
try:
    XDATA = pd.read_csv("./data/gpformel.csv")
except:
    traceback.print_exc()
    print("[x] Aktueller Ordner:", os.getcwd())

print(XDATA.sample(10))

pset = gp.PrimitiveSet("MAIN", XDATA.shape[1] - 1)  # pset.renameArguments(ARG0='x')
pset.addPrimitive(add, 2)
pset.addPrimitive(sub, 2)
pset.addPrimitive(neg, 1)
pset.addPrimitive(pAvg, 2)
pset.addPrimitive(zero, 1)
pset.addPrimitive(one, 1)
pset.addPrimitive(log2, 1)

xfuncPI = np.pi
xfuncE = np.e
pset.addTerminal(xfuncPI, name="pi")
pset.addTerminal(xfuncE, name="e")

# COMMAND LINE
pset.addEphemeralConstant("rand" + str(random.randint(1, 1000000)), lambda: random.randint(-10, 10))
pset.addEphemeralConstant("rand" + str(np.random.randint(1000000)), lambda: genR(-1, 1))
pset.addEphemeralConstant("rand" + str(np.random.randint(1000000)), lambda: genR(-np.pi, np.pi))

# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=7)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def check_dt_perf(_xdata):
    from flaubert.model.decision_trees import dt
    from flaubert.model.decision_trees import dt_utils as dtu
    target_feature = _xdata.columns[-1]
    nklasse0 = _xdata[_xdata[target_feature] == 0].shape[0]
    nklasse1 = _xdata.shape[0] - nklasse0
    balanced_n = min(nklasse1, nklasse0)
    clf = dt.warum(_xdata, xcolumnsSet=filter(lambda x: x != target_feature, _xdata.columns),
                   xcolTarget=target_feature, xlambdaF=lambda x: 'g0' if x < 1.0 else 'g1',
                   useOrdered=["g0", "g1"], balancedN={"g1": balanced_n, "g0": balanced_n},
                   test_size=0.2, max_depth=6,
                   min_samples_split=10, min_samples_leaf=10, criterion="entropy",
                   showStuff=False)
    TP, FP, FN, P1, R1, F1Score1 = dtu.extract_metrics(clf, _xdata, target_feature)
    return 1 - F1Score1


def evalSymbReg(individual, points):
    global GLOBAL_SCORE
    func = toolbox.compile(expr=individual)
    if GLOBAL_SCORE < 0.075: return 100,
    try:
        nsample = min(len(points), NMAX_SAMPLES)
        indexes = pd.Series(range(len(points))).sample(nsample)
        train_data = points.copy()  # .loc[indexes].reset_index(drop=True)
        xTargetPos = train_data.shape[1] - 1
        xF = [func(*x) for x in train_data.values[:, 0:xTargetPos]]
        # xT = points.values[:, xTargetPos]
        train_data = train_data.fillna(0)
        train_features = train_data.columns.tolist()[:-1]
        target = train_data.columns.tolist()[-1]
        train_data['gpFeature'] = xF
        # xres = [scipy.spatial.distance.braycurtis(xF, xT) - stats.pearsonr(xF, xT)[0], 1]
        xres = check_dt_perf(train_data[train_features + ['gpFeature', target]])
    except:
        traceback.print_exc()
        input('...')
        xres = 100
    GLOBAL_SCORE = getScore()
    if not np.isnan(xres) and xres < GLOBAL_SCORE:
        print(f"\n[x] Individual: {str(individual).replace('(', '[').replace(')', ']')}"
              f"\n -> neuer Score {xres}, alter Score {GLOBAL_SCORE} \n")
        GLOBAL_SCORE = xres
        setScore(GLOBAL_SCORE)
        time.sleep(3)
    if GLOBAL_SCORE == 0:
        pass
    return 100 if np.isnan(xres) else xres,


toolbox.register("evaluate", evalSymbReg, points=XDATA)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=7)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))

# set GLOBAL_SCORE
setScore(WORSE_SCORE)


def main(nevals=15, npop=10, nmaxsamples=1000):
    global NMAX_SAMPLES

    if XDATA.shape[0] == 0:
        print("[x] FEHLER in formelFinder! XDATA darf nicht leer sein!")
        return None

    maxeval = nevals
    NMAX_SAMPLES = nmaxsamples

    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    try:
        pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.2, maxeval, stats=mstats,
                                       halloffame=hof, verbose=True)
    except:
        traceback.print_exc()

    individual = str(hof[0]).replace("(", "[").replace(")", "]")
    score = getScore()
    with open("./data/gp_expr.out", "w") as f:
        f.write(
            json.dumps({
                'expr': individual,
                'score': score
            })
        )

    return str(hof[0])


def evalExpr(individual, xd=None, inklTarget=True, return_func=False):
    """
        individual: Formel in STRING Format
        xd: Data Input kann in multidimensional R sein
    """
    try:
        if xd is None:
            xd = XDATA
        xfunc = toolbox.compile(individual)
        if xfunc is None:
            print("[x] Fehler in toolbox.compile! xfunc ist None")
            return
        if inklTarget:
            xTargetPos = xd.shape[1] - 1
            xF = []
            for elem in xd.values[:, 0:xTargetPos]:
                # print(xfunc(*elem))
                xF.append(xfunc(*elem))
                # xF = [xfunc(*x) for x in xd.values[:, 0:xTargetPos]]
        else:
            xF = [xfunc(*x) for x in xd.values]
        return xF if not return_func else (xF, xfunc)
    except:
        print("[x] xd:")
        print(xd)
        print("[x] XDATA:")
        print(XDATA)
        traceback.print_exc()
        input(" ... ")
    return None if not return_func else (None, None)


def run():
    setScore(WORSE_SCORE)
    main()


if __name__ == "__main__":
    run()
