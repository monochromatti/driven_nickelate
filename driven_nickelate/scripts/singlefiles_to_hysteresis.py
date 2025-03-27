import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

read_kwargs = dict(
    names=["Delay", "X", "SEM X", "Y", "Y SEM"],
    usecols=list(range(5)),
    sep="\t",
    header=None,
    comment="#",
)

filenames = os.listdir(
    "/Users/mmatthiesen/surfdrive/Measurements/2023/eSRR NNO/Raw data/01.02.23/peak scan"
)
amplitudes = []
temperatures = []
times = []
for i, fn in enumerate(filenames):
    if fn.endswith(".txt"):
        df = pd.read_csv(
            f"/Users/mmatthiesen/surfdrive/Measurements/2023/eSRR NNO/Raw data/01.02.23/peak scan/{fn}",
            **read_kwargs,
        )
        temperatures.append(float(re.split("Tem=|.txt", fn)[-2]))
        amplitudes.append(df["X"])

        time = datetime.strptime(re.split("_", fn)[0], "%Hh%Mm%Ss").time()
        times.append(time)

times, amplitudes, temperatures = list(
    zip(*sorted(zip(times, amplitudes, temperatures)))
)

data = pd.DataFrame(
    np.c_[times, temperatures, amplitudes],
    columns=["Measurement completed", "Temperature", "Amplitude"],
)
data.to_csv("Processed data/pump_probe/030223_peak_hyst.csv", index=False)
