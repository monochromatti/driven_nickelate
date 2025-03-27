import os
import re

import numpy as np
import pandas as pd

parent_dir = os.path.dirname(os.getcwd())
folders = [
    "Raw data/03.02.23",
    "Raw data/04.02.23",
    "Raw data/05.02.23",
    "Raw data/06.02.23",
]
files = [
    os.path.join(parent_dir, folder, filename)
    for folder in folders
    for filename in os.listdir(os.path.join(parent_dir, folder))
]
files = list(filter(lambda x: "xPP xHWP" in x, files))
hwp, delay, time = [], [], []
for fn in files:
    x = float(re.findall(r"Del=([0-9]*\.[0-9]+)", fn)[-1])
    hwp.append(float(re.findall(r"HWP=([0-9]*\.[0-9]+)", fn)[0]))
    delay.append(x)
    time.append(round(6.67 * (163.52 - x), 4))

hwp, delay, time, files = map(list, zip(*sorted(zip(hwp, delay, time, files))))
pd.DataFrame(
    np.c_[hwp, delay, time, files], columns=["HWP", "Delay", "t", "Path"]
).to_csv(
    os.path.join(parent_dir, "File lists/pump_probe/050223_xPP_xHWP.csv"), index=False
)
