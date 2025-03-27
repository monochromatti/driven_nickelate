import os
import re

import pandas as pd

c_const = 0.299792458

paths = []
file_id = "ZnTe-PNPA delay dependence (fine, chopped probe)"
for folder in ["Raw data/21.01.23", "Raw data/22.01.23", "Raw data/23.01.23"]:
    paths += [
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if re.split("_", fn)[1] == file_id
    ]

delpos, ppdel = [], []
for i in range(len(paths)):
    delpos.append(float(re.search(r"Del=(.*?)_Dum", paths[i]).group(1)))
    ppdel.append((2 / c_const) * (160.93 - delpos[-1]))

ppdel, paths = list(zip(*sorted(zip(ppdel, paths))))

pd.DataFrame(data={"Pump-probe delay (ps)": ppdel, "Path": paths}).to_csv(
    "File lists/xDelay_ChoppedProbe.dat", index=False, sep=";"
)
