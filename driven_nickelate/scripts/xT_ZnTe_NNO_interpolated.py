import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

c_const = 0.299792458

parent_dir = os.path.dirname(os.getcwd())
ref_files = pd.read_csv(
    os.path.join(parent_dir, "File lists/xT_ZnTe_NNO.dat"), comment="#"
)

read_kwargs = dict(
    names=["Delay", "X", "SEM X", "Y", "Y SEM"],
    usecols=list(range(5)),
    sep="\t",
    header=None,
    comment="#",
)
df = pd.read_csv(os.path.join(parent_dir, ref_files["Path"].iloc[0]), **read_kwargs)
N_samples = df["Delay"].size

idx_warming = ref_files["Direction"] > 0
data_warming = np.zeros((N_samples, len(ref_files["Path"].loc[idx_warming])))
for i, fn in enumerate(ref_files["Path"].loc[idx_warming]):
    df = pd.read_csv(os.path.join(parent_dir, fn), **read_kwargs)
    data_warming[:, i] = df["X"]

idx_cooling = ref_files["Direction"] < 0
data_cooling = np.zeros((N_samples, len(ref_files["Path"].loc[idx_cooling])))
for i, fn in enumerate(ref_files["Path"].loc[idx_cooling]):
    df = pd.read_csv(os.path.join(parent_dir, fn), **read_kwargs)
    data_cooling[:, i] = df["X"]

data_cooling = np.flip(data_cooling, axis=1)

tempr_list = np.arange(10, 290, 1)
interp_kwargs = {"kind": "linear", "fill_value": "extrapolate", "bounds_error": False}
interp_warming = np.zeros((data_warming.shape[0], len(tempr_list)))
for i in range(data_warming.shape[0]):
    interp_warming[i, :] = interp1d(
        ref_files["Get temperature"].loc[idx_warming],
        data_warming[i, :],
        **interp_kwargs,
    )(tempr_list)

interp_cooling = np.zeros((data_cooling.shape[0], len(tempr_list)))
for i in range(data_cooling.shape[0]):
    interp_cooling[i, :] = interp1d(
        ref_files["Get temperature"].loc[idx_cooling][::-1],
        data_cooling[i, :],
        **interp_kwargs,
    )(tempr_list)

pd.DataFrame(
    np.c_[df["Delay"], interp_warming], columns=["Delay", *list(tempr_list)]
).to_csv(
    os.path.join(parent_dir, "Processed data/xT_ZnTe_NNO_interp_warming.dat"),
    index=False,
)
pd.DataFrame(
    np.c_[df["Delay"], interp_cooling], columns=["Delay", *list(tempr_list)]
).to_csv(
    os.path.join(parent_dir, "Processed data/xT_ZnTe_NNO_interp_cooling.dat"),
    index=False,
)
