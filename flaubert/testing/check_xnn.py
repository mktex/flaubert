
import pandas as pd
from flaubert.model.neural_networks import xnn

def check_input_numeric_nn():
    dfnum = pd.read_csv("../../data/test.csv")
    feature_target = "DistanceFromHome"
    dfnum["DistanceFromHome"] /= 50
    datenpaket = xnn.input_numeric_nn(dfnum,
                                      xcolTarget=feature_target, hidden_layer_sizes=(16, 16), verbose=True,
                                      aktivF="logistic",
                                      skalierung=(0, 1), standardisierung=True, entferneKorr=True, thresholdKorr=0.01,
                                      namedLearner=f"model_nn_{feature_target}")

def check_input_categ_nn():
    dfcat = pd.read_csv("../../data/test.csv")
    feature_target = dfcat.columns[-1]
    datenpaket = xnn.input_class_nn(dfcat, xcolTarget=feature_target, xcolID=None, showStuff=True,
                                    hidden_layer_sizes=(16, 16), aktivF="logistic",
                                    skalierung=(0, 1), standardisierung=True,
                                    namedLearner=f"model_nn_{feature_target}")

# check_input_numeric_nn()

check_input_categ_nn()