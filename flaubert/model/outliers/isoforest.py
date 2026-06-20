from sklearn.ensemble import IsolationForest


def do_outlier_detection(df, target, contamination=0.05, random_state=42):
    records = df[list(filter(lambda x: x not in [target], df.columns))].copy()
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    iso_forest.fit(records)
    df['outlier'] = iso_forest.predict(records)
    print(df['outlier'].describe())
    print(df[df['outlier'] == -1].shape)
    df = df[df['outlier'] >= 0]
    df = df[list(filter(lambda x: x != 'outlier', df.columns))].reset_index(drop=True)
    return df


def remove_outliers_by_isolation_forest(_df):
    iso_forest = IsolationForest(contamination='auto')
    iso_forest.fit(_df)
    _df['outlier'] = iso_forest.predict(_df)
    _df = _df[_df['outlier'] >= 0]
    _df = _df[list(filter(lambda x: x != 'outlier', _df.columns))]
    return _df, iso_forest

