import re

import lmfit as lm
import numpy as np
import pandas as pd
from seaborn import color_palette

from driven_nickelate.config import paths

__all__ = ["field_from_hwp"]

read_kwargs_long = dict(
    names=["Delay", "X1", "X1 SEM", "Y1", "Y1 SEM", "X2", "X2 SEM", "Y2", "Y2 SEM"],
    usecols=list(range(9)),
    sep="\t",
    header=None,
    comment="#",
)

files = [
    "raw_data/20.02.23/10h05m35s_PNPA xHWP_3.13-4.80_Tem=3.10_Del=160.66_Sam=9.54.txt",
    "raw_data/20.02.23/09h49m48s_PNPA xHWP_3.21-4.96_Tem=3.10_Del=163.72_Sam=9.54.txt",
    "raw_data/20.02.23/09h53m41s_PNPA xHWP_3.18-4.87_Tem=3.10_Del=163.60_Sam=9.54.txt",
    "raw_data/20.02.23/09h57m37s_PNPA xHWP_3.14-4.84_Tem=3.10_Del=163.50_Sam=9.54.txt",
    "raw_data/20.02.23/10h01m35s_PNPA xHWP_3.14-4.82_Tem=3.10_Del=163.42_Sam=9.54.txt",
]
files = [paths.root / f for f in files]
lockin_phase = -87  # deg

data = pd.read_csv(
    paths.root
    / "raw_data/20.02.23/09h12m44s_PNPA_120.00-120.69_HWP=50.00_Sam=9.54.txt",
    sep="\t",
    comment="#",
    header=None,
)
eos = np.real(
    (data.iloc[:, 3] + 1j * data.iloc[:, 4]) * np.exp(1j * np.deg2rad(lockin_phase))
)

colors = color_palette("Set1", n_colors=len(files))
delay = [float(re.search(r"Del=(\d+\.\d+)", f.name).group(1)) for f in files]

for i in range(len(files)):
    data = pd.read_csv(files[i], **read_kwargs_long)
    eos = np.real(
        (data["X2"] + 1j * data["Y2"]) * np.exp(1j * np.deg2rad(lockin_phase))
    )


data = pd.read_csv(files[2], **read_kwargs_long)
hwp = data["Delay"].values
eos = np.real((data["X2"] + 1j * data["Y2"]) * np.exp(1j * np.deg2rad(lockin_phase)))
eos /= np.max(eos)


def fit_func(x, x0, a):
    return a * np.cos(2 * np.deg2rad(x - x0)) ** 2


model = lm.Model(fit_func)
params = model.make_params(x0=0, a=1)
result = model.fit(eos, params, x=hwp)


def field_from_hwp(hwp):
    return result.eval(x=hwp)
