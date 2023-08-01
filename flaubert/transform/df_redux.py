
import numpy as np


def reduktion_multikolinearitaet(df_input, threshold_corr=0.7,
                                 target_col="empf", id_col="id"):
    """
        Reduziert DataFrame mit Dummy Variablen entsprechend Multikolinearität
        xdfcat_dummy kommt zB aus einem kateg2dummy Aufruf:
            xdfcat_dummy = kateg.kateg2dummy(dfcat, sep=None)
    """
    xcorrdf = df_input[list(filter(lambda feature: feature not in [id_col], df_input.columns))].corr()
    remove_cols = []
    if target_col is not None:
        features_check = list(filter(lambda feature: feature != target_col, xcorrdf.columns))
    else:
        features_check = xcorrdf.columns
    for xcol in features_check:
        if xcol in remove_cols:
            continue
        print("\nÜberprüfung Korrelationen für Feature:", xcol)
        list_high_corrs = list(map(lambda x: np.abs(x) > threshold_corr, xcorrdf[xcol].values))
        _cols = xcorrdf[list_high_corrs].index.values.tolist()
        if target_col is not None:
            korrelation_zur_zielvariable = list(map(lambda korreliertes_feature:
                                                    xcorrdf[target_col].to_dict()[
                                                        korreliertes_feature],
                                                    _cols))
            korrelation_zur_zielvariable_abs_werte = [np.abs(t) for t in korrelation_zur_zielvariable]
            max_corr_index = korrelation_zur_zielvariable_abs_werte.index(
                np.max(korrelation_zur_zielvariable_abs_werte))
            behalte_feature = _cols[max_corr_index]
            entferne_features = list(filter(lambda feature: feature not in [behalte_feature, target_col], _cols))
            print("\t Bleibt: {} (Korrelation zur Zielvariable {})".format(
                behalte_feature, korrelation_zur_zielvariable[max_corr_index]
            ))
            print("\t Werden entfernt:", entferne_features)
        else:
            _cols = list(filter(lambda x: x != xcol, _cols))
            entferne_features = _cols
        if len(entferne_features) != 0:
            print("\t {}:".format(xcol), entferne_features)
            remove_cols.extend(entferne_features)
    print("\n[x] Features mit hoher Korrelation zu anderen Features (non-target) werden entfernt:")
    print(remove_cols, "\n")
    remove_cols = list(set(remove_cols))
    n_vorher = df_input.shape[1]
    n_nachher = df_input.shape[1] - len(remove_cols)
    print("[x] Datenbestandsdimensionalität reduziert sich von {} auf {} Features ({}%)".format(
        n_vorher, n_nachher,
        np.round(100 * n_nachher / n_vorher, 2)
    ))
    df_input = df_input[list(filter(lambda x: x not in remove_cols, df_input.columns))]
    return df_input
