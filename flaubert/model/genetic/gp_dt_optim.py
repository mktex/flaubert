from importlib import reload
import pandas as pd
import numpy as np
from pprint import pprint

from sklearn.metrics import confusion_matrix, classification_report

from flaubert.model.decision_trees import dt, dt_utils as dtu
from flaubert.model.genetic import formelDT as fdt
from flaubert.eda import dbeschreiben

reload(fdt)
reload(dt)

xfunk_wl = lambda individual: individual.replace('(', '[').replace(')', ']')


def do_sim(_xdata, nklasse_1=3000, nklasse_0=3000):
    target_feature = _xdata.columns[-1]
    clf = dt.warum(_xdata, xcolumnsSet=filter(lambda x: x != target_feature, _xdata.columns),
                   xcolTarget=target_feature, xlambdaF=lambda x: 'g0' if x < 1.0 else 'g1',
                   useOrdered=["g0", "g1"], balancedN={"g1": nklasse_1, "g0": nklasse_0},
                   test_size=0.2, max_depth=4,
                   min_samples_split=10, min_samples_leaf=10, criterion="entropy",
                   showStuff=True)
    predicted = clf.predict(_xdata[list(filter(lambda x: x != target_feature, _xdata.columns))].values)
    targets = _xdata[target_feature].values
    dtu.plot_dt_result(predicted, targets, class_labels=None)
    return clf


def zeige_formeln(gp_features, gp_formeln):
    for gpf, gpform in list(zip(gp_features, gp_formeln)):
        print("[x] Feature:", gpf)
        print("     Formel:", xfunk_wl(gpform))


# --------- Debug Zone ---------
def debug_classif_sim():
    x0 = np.random.randint(10)
    x1 = np.random.randint(10)
    x2 = np.random.randint(10)
    x3 = np.random.randint(10)
    x4 = np.random.randint(10)
    x5 = np.random.randint(10)
    output = 0
    if x1 + x2 == x3 + x4: output = 1
    if x1 + x3 == x2 + x5: output = 1
    return [x0, x1, x2, x3, x4, x5, output]


# Data 1
train_features = ["x0", "x1", "x2", "x3", "x4", "x5"]
target = "target"
xdata = pd.DataFrame(list(map(lambda x: debug_classif_sim(), range(30000))),
                     columns=(train_features + [target]))

# Data-Label corruption
if True:
    ikrandom = pd.Series(xdata[xdata.target == 1].sample(500).index.values.tolist())
    for ik in ikrandom: xdata.at[ik, 'target'] = 0

# --------- Debug Zone ---------

res = dbeschreiben.frequenz_werte(xdata, group_by_feature="target", prozente=False, sep=None)
NSamples = res['id'].min()
res

# Basic check
if True:
    clf = do_sim(xdata[train_features + ["target"]], nklasse_1=NSamples, nklasse_0=NSamples)
    TP, FP, FN, P1, R1, F1Score1_prev = dtu.extract_metrics(clf, xdata[train_features + ["target"]], target,
                                                            printout=True)

# GP Optimierung
gp_features = []
gp_formeln = []
for gp_count in range(0, 4):
    xdata = xdata[train_features + gp_features + [target]].copy()
    xdata.to_csv('./data/gpformel.csv', index=None)
    reload(fdt)
    zeige_formeln(gp_features, gp_formeln)
    input("...")
    xfunk = fdt.main(nevals=75, npop=100, nmaxsamples=NSamples)
    xdata[f"gpprog{gp_count}"] = fdt.evalExpr(xfunk, xd=xdata, inklTarget=True, return_func=False)
    gp_features.append(f"gpprog{gp_count}")
    gp_formeln.append(xfunk)
    clf = do_sim(xdata[train_features + gp_features + ["target"]], nklasse_1=NSamples, nklasse_0=NSamples)
    TP, FP, FN, P1, R1, F1Score1 = dtu.extract_metrics(clf, xdata[train_features + gp_features + ["target"]], target,
                                                       printout=True)
    if F1Score1 <= 0.975 * F1Score1_prev:
        print(f"[x] GP-Methode bringt nix Neues! (F1-Score vorher war {F1Score1_prev}, danach {F1Score1})")
        break
    F1Score1_prev = F1Score1

"""
Note that in binary classification, 
    recall of the positive class (R1) is also known as "sensitivity"; 
    recall of the negative (R0) class is "specificity".

Konfusionsmatrix - in binary classification, the count of 
    true negatives is :math:`C_{0,0}`, 
    false negatives is :math:`C_{1,0}`, 
    true positives is :math:`C_{1,1}` and 
    false positives is :math:`C_{0,1}`.
"""

"""
=> was hilft sind extra features!

"""
