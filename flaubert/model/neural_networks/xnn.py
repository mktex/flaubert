import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from flaubert.eda import dfgestalt, dbeschreiben
from flaubert.ersatz import dfersatz

debug = False
model_dir = "./data/modeldata/"


def print_wenn(*args):
    global debug
    if debug:
        print(*args)

# Entferne nasty warnings, z.B. division by zero, wenn's halt in confusion-matrix net geklappt hat
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

def entferne_xcolid(dftrain, xcolid):
    """ Etnfernt xcolid aus dem dftrain
    """
    if xcolid is not None and xcolid in dftrain.columns:
        dftrain = dftrain[list(filter(lambda feature: feature != xcolid, dftrain.columns))]
    return dftrain


def print_infobox(printOut=True):
    if not printOut:
        return
    print("""
    ****** ****** ****** ****** ****** ****** ****** ****** ****** ****** ****** ****** ****** ****** ****** 
    
    xNNT nur auf Transformierte Werte anwenden!
    
    Beispiel Code:
    xtemp_nn = xdf.copy()
    xtemp_nn_transformed = xdf.copy()
    for xc in xnn_features:
        xtemp_nn_transformed[xc] = xdictT[xc]['s0'].transform(xtemp_nn_transformed[xc].values.reshape(-1, 1)).reshape(-1)
        # Performe das Gleiche, falls 's1' vorhanden ist
    xtemp_nn["nnOutput"] = xNNT.predict(xtemp_nn_transformed[xnn_features])   

    ****** ****** ****** ****** ****** ****** ****** ****** ****** ****** ****** ****** ****** ****** ******     
    """)


def input_class_nn(xdfInput, xcolTarget="standorttyp", xcolID=None, showStuff=False,
                   hidden_layer_sizes=(8, 8), aktivF="logistic",
                   skalierung=(0, 1), standardisierung=True,
                   namedLearner=None):
    """ input None/NaN Werte mithilfe einer NN
        xdf: soll ein Pandas DF sein, mit allen Werte (Floats) vorhanden
    """
    from sklearn.neural_network import MLPClassifier

    print()
    xdfInput = entferne_xcolid(xdfInput.copy(), xcolID)
    xdfInput[xcolTarget] = [str(x) if not dbeschreiben.ist_ungueltig(x, ()) else x
                            for x in xdfInput[xcolTarget].values.tolist()]
    xdfT = xdfInput[[x for x in xdfInput.columns if x != xcolTarget]].copy()
    xdfT = dfersatz.ersatz_nullwerte_durch_mean(xdfT)

    nFeatures = xdfT.shape[1]
    xTarget = xdfInput[xcolTarget].copy()
    indexTrain = xTarget[[not dbeschreiben.ist_ungueltig(x, ()) for x in xTarget.values]].index
    indexPrognose = xTarget[[dbeschreiben.ist_ungueltig(x, ()) for x in xTarget.values]].index
    xKlassen = list(set(xTarget.loc[indexTrain]))
    xKlassen = list(zip(list(range(len(xKlassen))), xKlassen))
    xfGetKlassenWertByIndex = lambda xIndx: xKlassen[xIndx][1]
    zugeordneteFeatures = xdfT.columns.values.tolist()
    for k in indexTrain:
        xr = [y for y in xKlassen if y[1] == xTarget.at[k]][0]
        xTarget.at[k] = xr[0]

    xdt, xdictT = get_transformation(xdfT, zugeordneteFeatures, standardisierung, skalierung)

    xNNT = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=aktivF, solver="adam",
                         learning_rate="adaptive", learning_rate_init=0.001, momentum=0.001, max_iter=2500,
                         batch_size=max(5, int(0.01 * xdfT.shape[0])), shuffle=True,
                         verbose=False, early_stopping=True)
    X = xdt.loc[indexTrain].values.reshape(-1, nFeatures)
    datenpaket = dfgestalt.splitdata_train_test(X, xTarget.loc[indexTrain].values, ptrain=0.8)
    train_features, train_targets, test_features, test_targets = datenpaket
    xNNT.fit(train_features, train_targets.tolist())
    xNNOutputPred_train = xNNT.predict(train_features)
    xNNOutputPred_test = xNNT.predict(test_features)
    if showStuff:
        print(xKlassen)
        print("\n[x] Performanz in TRAIN:")
        print(classification_report(train_targets.tolist(), xNNOutputPred_train.tolist()))
        print("\n[x] Performanz in TEST:")
        print(classification_report(test_targets.tolist(), xNNOutputPred_test.tolist()))

    indexPrognoseWerte = [xfGetKlassenWertByIndex(x.index(max(x))) for x in
                          xNNT.predict_proba(xdt.loc[indexPrognose].values.reshape(-1, nFeatures)).tolist()]

    for xkey in list(xdictT.keys()):
        del xdictT[xkey]['xs']
    if namedLearner is not None:
        namedLearner = "xnn_" + namedLearner
        xDictModel = {
            "NN": xNNT, "transform": xdictT, "Klassen": xKlassen, "zugeordneteFeatures": zugeordneteFeatures
        }
        save_model(xDictModel, namedLearner)

    print_infobox(showStuff)
    return list(zip(indexPrognose, indexPrognoseWerte)), xNNT, xdictT, namedLearner


def get_transformation(xdfT, zugeordneteFeatures, standardisierung, skalierung):
    xdictT = get_dict_transform(xdfT, zugeordneteFeatures, standardisierung=standardisierung, skalierung=skalierung)
    xDFDict = {}
    for xcol in zugeordneteFeatures:
        xDFDict[xcol] = xdictT[xcol]['xs']
    xdt = pd.DataFrame(xDFDict)
    xdt = xdt[zugeordneteFeatures]
    xdt.index = xdfT.index
    if xdt.shape[0] > 25:
        print_wenn("[x] Stichprobe, 25 Datensätze, zum Training (nach Transformationen):")
        if debug: print(xdt.sample(25))
    else:
        print_wenn(xdt)
    return xdt, xdictT


def entferne_korr(xdfInput, xcolTarget, entferneKorr, thresholdKorr):
    from flaubert.statistik import stat_op as statistik
    if entferneKorr:
        xcols = [x for x in xdfInput.columns.values if x != xcolTarget]
        xcorr = statistik.get_corrlist(xdfInput[xcols + [xcolTarget]], xcolTarget)
        xcorr = xcorr[[np.abs(x) > thresholdKorr for x in xcorr.values]]
        if xcorr.shape[0] == 0:
            print_wenn(f"[x] WARNUNG keine der vorhandenen Features haben eine Korrelation > {thresholdKorr} "
                       f"zu der Zielvariable {xcolTarget}")
            return xdfInput
        else:
            print_wenn(xcorr)
            print_wenn("[x] Auswahl Features:", xcorr.index.tolist())
            xdfInput = xdfInput[xcorr.index.tolist()]
            print_wenn(xdfInput.describe())
    return xdfInput


def input_numeric_nn(xdfInput, xcolTarget="ant_pb_kunden_w", xcolID=None, hidden_layer_sizes=(8, 8),
                     verbose=False, skalierung=(-1, 1), standardisierung=True,
                     entferneKorr=True, thresholdKorr=0.1,
                     aktivF="logistic", namedLearner=None,
                     maxiter=2500,
                     lernrate=0.1,
                     nsample_nanwerte=50):
    """ input None/NaN Werte mithilfe einer NN
        xdf: das soll ein Pandas DF sein, mit allen Werte (Floats) vorhanden
        aktivF: tanh, logistic, relu
        skalierung: ist für MinMaxScaler als Parameter feature_range zu geben
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import mean_squared_error
    global debug

    debug = verbose
    print_wenn("")

    xdfInput = entferne_xcolid(xdfInput.copy(), xcolID)
    df_ohne_nans = xdfInput[xcolTarget].dropna()
    if df_ohne_nans.shape[0] == xdfInput.shape[0]:
        setze_indexen = pd.Series(range(xdfInput.shape[0])).sample(nsample_nanwerte)
        for id in setze_indexen:
            xdfInput.at[id, xcolTarget] = None

    xdfInput = entferne_korr(xdfInput, xcolTarget, entferneKorr, thresholdKorr)
    xdfT = xdfInput[[x for x in xdfInput.columns if x != xcolTarget]].copy()
    xdfT = dfersatz.ersatz_nullwerte_durch_mean(xdfT)
    nFeatures = xdfT.shape[1]
    zugeordneteFeatures = xdfT.columns.values.tolist()
    if len(zugeordneteFeatures) == 0:
        print_wenn("[x] Fehler: keine Feature nach Filter mit Korrelation verblieben..")
        return [None] * 4

    xTarget = xdfInput[xcolTarget].copy()
    indexTrain = xTarget[[x is not None and not np.isnan(x) for x in xTarget.values]].index
    indexPrognose = xTarget[[x is None or np.isnan(x) for x in xTarget.values]].index

    # Transformationen
    xdt, xdictT = get_transformation(xdfT, zugeordneteFeatures, standardisierung, skalierung)

    if True:
        xNNT = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=aktivF, solver="adam",
                            learning_rate="adaptive", learning_rate_init=lernrate, momentum=0.001, max_iter=maxiter,
                            batch_size=max(10, int(0.05 * xdfT.shape[0])), shuffle=True,
                            verbose=False, early_stopping=True, n_iter_no_change=int(0.5 * maxiter))
        # TRAIN, TEST, PROGNOSE
        X = xdt.loc[indexTrain].values.reshape(-1, nFeatures)
        datenpaket = dfgestalt.splitdata_train_test(X, xTarget.loc[indexTrain].values, ptrain=0.8)
        train_features, train_targets, test_features, test_targets = datenpaket
        xNNT.fit(train_features, train_targets)
        xNN_output_pred_train = xNNT.predict(train_features)
        xNN_output_pred_test = xNNT.predict(test_features)

        if verbose:
            print("\n[x] Performanz in NN-Training:")
            print("[x] R^2:", explained_variance_score(train_targets, xNN_output_pred_train))
            print("[x] MSE:", mean_squared_error(train_targets, xNN_output_pred_train))
            print("\n[x] Performanz in NN-Test:")
            print("[x] R^2:", explained_variance_score(test_targets, xNN_output_pred_test))
            print("[x] MSE:", mean_squared_error(test_targets, xNN_output_pred_test))
            print("[x] Prognose vs IST..")
            print("[x] Zielvariable:", xcolTarget)
            print("[x] Verwendete Variablen:", zugeordneteFeatures)
            plt.scatter(x=test_targets, y=xNN_output_pred_test, c="blue", s=3, alpha=0.5)
            plt.title(f"Prognose vs IST für: {xcolTarget}")
            plt.xlabel("IST-Werte")
            plt.ylabel("Prognose")
            plt.show()

    indexPrognoseWerte = xNNT.predict(xdt.loc[indexPrognose].values.reshape(-1, nFeatures))
    for xkey in list(xdictT.keys()):
        del xdictT[xkey]['xs']
    if namedLearner is not None:
        namedLearner = "xnn_" + namedLearner
        xDictModel = {
            "NN": xNNT, "transform": xdictT, "zugeordneteFeatures": zugeordneteFeatures
        }
        save_model(xDictModel, namedLearner)

    print_infobox(verbose)
    return list(zip(indexPrognose, indexPrognoseWerte)), xNNT, xdictT, namedLearner


def get_dict_transform(xdfT, zugeordneteFeatures, standardisierung=True, skalierung=None):
    print_wenn("[x] Zugeordnete Features:", zugeordneteFeatures)
    xdictT = {}
    for xcol in zugeordneteFeatures:
        print_wenn("****************** {} ******************".format(xcol))
        xs = xdfT[xcol].values
        if standardisierung:
            s0 = StandardScaler()
            x0 = s0.fit_transform(xs.reshape(-1, 1))  # z = (x - u) / s
            print_wenn(".. Standard Scaler (z = (x - u) / s)")
        else:
            s0 = None
            x0 = xs.reshape(-1, 1).copy()
        if skalierung is not None:
            print_wenn(".. Skalierung zum Intervall ", skalierung)
            s1 = MinMaxScaler(feature_range=skalierung)
            x1 = s1.fit_transform(x0)
            xs = pd.Series(x1.reshape(1, -1)[0])
        else:
            s1 = None
            xs = pd.Series(x0.reshape(1, -1)[0])
        xdictT[xcol] = {'xs': xs, 's0': s0, 's1': s1}
    return xdictT


def save_model(xObj, xname):
    global model_dir
    import pickle
    fname = (model_dir + "/" + xname + '.obj').replace("//", "/")
    with open(fname, 'wb') as fp:
        pickle.dump(xObj, fp)


def load_model(xname):
    global model_dir
    import pickle
    fname = (model_dir + "/" + xname + '.obj').replace("//", "/")
    with open(fname, 'rb') as fp:
        xObj = pickle.load(fp)
    return xObj


def apply_modell_nn_fuer_feature(df_input, target_col, modelnamen, ersatz_in_df=False):
    """
        Das Modell mit dem Modelnamen wird auf df_input angewendet, inkl. Transformation
    """
    xKlassen = []
    xfGetKlassenWertByIndex = lambda xIndx: xKlassen[xIndx][1]
    df_input = df_input.copy()
    modelDict = load_model(modelnamen)
    xTarget = df_input[target_col].copy()
    indexPrognose = xTarget[[x is None or np.isnan(x) for x in xTarget.values]].index
    xDFDict = {}
    for xcol in list(modelDict['transform'].keys()):
        xs = df_input[xcol].values
        s0 = modelDict['transform'][xcol]["s0"]
        x0 = s0.transform(xs.reshape(-1, 1))  # z = (x - u) / s
        s1 = modelDict['transform'][xcol]["s1"]
        if s1 is not None:
            x1 = s1.transform(x0)
            xs = pd.Series(x1.reshape(1, -1)[0])
        else:
            xs = pd.Series(x0.reshape(1, -1)[0])
        xDFDict[xcol] = xs
    xdt = pd.DataFrame(xDFDict)
    xdt = xdt[modelDict["zugeordneteFeatures"]]
    xdt.index = df_input.index
    xdt = dfersatz.ersatz_nullwerte_durch_mean(xdt)
    print("[x] Stichprobe, 25 Datensätze, zum Training (nach Transformationen):")
    print(xdt.sample(10))
    X = xdt.loc[indexPrognose].values.reshape(-1, xdt.shape[1])
    xNNT = modelDict["NN"]
    if "MLPClassifier" in str(type(xNNT)):
        xKlassen = modelDict["Klassen"]
        xNNOutputPred = xNNT.predict_proba(X)
        xretWerte = pd.DataFrame(
            [[round(y, 3) for y in x] for x in pd.Series(xNNOutputPred.tolist())]).drop_duplicates()
        print("[x] Output Pred hat eine Größe (eindeutige Vektoren) von:", xretWerte.shape[0])
        print(xretWerte.sample(min(25, xretWerte.shape[0])))
        xNNOutputPred = [xfGetKlassenWertByIndex(x.index(max(x))) for x in xNNOutputPred.tolist()]
    elif "MLPRegressor" in str(type(xNNT)):
        xNNOutputPred = xNNT.predict(X)
    else:
        print("[x] Unbekanntes Modell:", type(xNNT))
        return None
    if not ersatz_in_df:
        return indexPrognose, xNNOutputPred
    else:
        for ik, xwert in zip(indexPrognose, xNNOutputPred):
            df_input.at[ik, target_col] = xwert
        return df_input


def apply_nn_simpel(xdfInput, xnn_features, xNNT, xdictT):
    xtemp_nn_transformed = xdfInput.copy()
    for xc in xnn_features:
        xInputMatrix = xtemp_nn_transformed[xc].values.reshape(-1, 1)
        werte_transform = xInputMatrix.copy()
        if xdictT[xc]['s0'] is not None:
            werte_transform = xdictT[xc]['s0'].transform(werte_transform).reshape(-1)
        if xdictT[xc]['s1'] is not None:
            werte_transform = xdictT[xc]['s1'].transform(werte_transform).reshape(-1)
        xtemp_nn_transformed[xc] = werte_transform
    return xNNT.predict(xtemp_nn_transformed[xnn_features])
