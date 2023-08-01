
import numpy as np

def safe_zahl_check(xn):
    try:
        float(xn)
        return True
    except:
        pass
    return False


istWas = lambda xobj, xk: xk in str(type(xobj))
istList = lambda x: istWas(x, "list")
istDict = lambda x: istWas(x, "dict")
istNone = lambda x: x is None
istZahl = lambda x: safe_zahl_check(x)
istString = lambda x: istWas(x, "str") or istWas(x, "byte")

einWertOderListe = lambda xs: xs if xs.shape[0] != 1 else xs[0]

getNones = lambda xs: np.array([y is None for y in xs])
checkNone = lambda xs: True in getNones(xs)

getNaNs = lambda xs: np.array([np.isnan(float(y)) if (y is not None and isinstance(y, float)) else False for y in xs])
checkNaN = lambda xs: True in getNaNs(xs)

getInfs = lambda xs: np.array([np.isinf(float(y)) if (y is not None and isinstance(y, float)) else False for y in xs])
checkInf = lambda xs: True in getInfs(xs)

ist_ungueltig = lambda xs: einWertOderListe(
                    np.array(list(map(lambda a, b, c: True in [a, b, c], getNones(xs), getNaNs(xs), getInfs(xs))))
)
