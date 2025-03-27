import os
import sys

import pandas as pd

project_dir = "/Users/mmatthiesen/surfdrive/Measurements/2023/eSRR NNO"
sys.path.insert(0, project_dir)
from support_functions import autophase, enclosing_indices, read_kwargs

os.chdir(project_dir)

files = pd.read_csv("File lists/pump_probe/220223_xT_LSAT.dat", comment="#")
data = {}

for i in range(len(files)):
    df = pd.read_csv(files["Path"][i], **read_kwargs)
    df["X"], df["Y"] = autophase(df["X"], df["Y"])
    df[["X", "Y"]] -= df[["X", "Y"]].mean()

    data[files["Temperature"][i]] = df

data = pd.concat(data.values(), keys=data.keys(), names=["Temperature"])
data = data.reset_index(level=["Temperature"])
T_meas = sorted(data["Temperature"].unique())


def get_data_at_temp(T_target):
    index_lo, index_hi = enclosing_indices(T_meas, T_target)
    T_hi, T_lo = T_meas[index_hi], T_meas[index_lo]

    if T_hi == T_lo:
        if abs(T_hi - T_target) > 1:
            print("Closest LSAT measurement temperature: {} K".format(T_lo))
        return data[data["Temperature"] == T_lo].drop("Temperature", axis=1)
    else:
        data_hi = data[data["Temperature"] == T_hi]
        data_lo = data[data["Temperature"] == T_lo]
        x = (T_target - T_hi) / (T_lo - T_hi)

        data_target = x * data_lo + (1 - x) * data_hi

        return data_target.drop("Temperature", axis=1)
