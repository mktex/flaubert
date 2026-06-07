

def xdfmap(df, xfunk, features):
  makedict = lambda df, features: list(map(lambda x: dict(list(zip(features, x))), df[features].values))
  return list(map(lambda y: xfunk(y), makedict(df, features)))