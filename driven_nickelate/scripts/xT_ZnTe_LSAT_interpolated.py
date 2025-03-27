import os

import numpy as np
import pandas as pd

c_const = 0.299792458

parent_dir = os.path.dirname(os.getcwd())
ref_files = pd.read_csv(
    os.path.join(parent_dir, "File lists/xT_ZnTe_LSAT.dat"), comment="#"
)

read_kwargs = dict(
    names=["Delay", "X", "SEM X", "Y", "Y SEM"],
    usecols=list(range(5)),
    sep="\t",
    header=None,
    comment="#",
)
df = pd.read_csv(os.path.join(parent_dir, ref_files["Path"].iloc[0]), **read_kwargs)
N = df["Delay"].size

ref_data = np.zeros((N, 1 + len(ref_files["Path"])))
ref_data[:, 0] = df["Delay"]
for i, fn in enumerate(ref_files["Path"]):
    df = pd.read_csv(os.path.join(parent_dir, fn), **read_kwargs)
    ref_data[:, i + 1] = df["X"]

df_ref = pd.DataFrame(ref_data, columns=["Delay", *list(ref_files["Get temperature"])])

tempr_list = np.arange(10, 290, 1)
ref_data_interp = np.zeros((df_ref.shape[0], 1 + len(tempr_list)))
ref_data_interp[:, 0] = ref_data[:, 0]

for i in range(1, df_ref.shape[0]):
    ref_data_interp[i, 1:] = np.interp(
        tempr_list, ref_files["Get temperature"], df_ref.drop("Delay", axis=1).loc[i]
    )

pd.DataFrame(ref_data_interp, columns=["Delay", *list(tempr_list)]).to_csv(
    os.path.join(parent_dir, "Processed data/xT_ZnTe_LSAT_interp.dat"), index=False
)
