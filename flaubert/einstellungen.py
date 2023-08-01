from contextlib import contextmanager
import pandas as pd


@contextmanager
def halt_pandas_warnings():
    with pd.option_context("mode.chained_assignment", None):
        yield


halt_pandas_warnings()

# Kosmetik fürs Terminal
TRENNLINIE_TYP_I = '---' * 30
TRENNLINIE_TYP_II = '***' * 30

# Diagrammen vereinheitlichen
FIGSIZE_BREIT = (12, 8)
FIGSZE_QUADRAT = (12, 12)
FIGDPI_HOCH = 120
FIGDPI_NORM = 100
FIGDPI_NIEDRIG = 80
FIG_FONTZISE = 8

# Delays
TIMEOUT_SECONDS = 300
