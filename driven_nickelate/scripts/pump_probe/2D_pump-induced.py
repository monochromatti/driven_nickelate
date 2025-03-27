import os
import re
import sys

import numpy as np
import pandas as pd

project_dir = "/Users/mmatthiesen/surfdrive/Measurements/2023/eSRR NNO"
sys.path.insert(0, project_dir)

os.chdir(project_dir)

read_kwargs = dict(
    names=[
        "Delay",
        "X 166 Hz",
        "SEM X 166 Hz",
        "Y 166 Hz",
        "SEM Y 166 Hz",
        "X 333 Hz",
        "SEM X 333 Hz",
        "Y 333 Hz",
        "SEM Y 333 Hz",
    ],
    usecols=list(range(9)),
    sep="\t",
    header=None,
    comment="#",
)

paths = []
dates = ["14.02.23", "13.02.23", "12.02.23", "11.02.23"]
for date in dates:
    datepath = os.path.join("Raw data/", date)
    paths += [os.path.join(datepath, filename) for filename in os.listdir(datepath)]
paths = list(filter(lambda x: x.endswith(".txt"), paths))
paths = list(filter(lambda x: re.split("_", x)[1] == "2D THz pump-induced", paths))

delays = []
for path in paths:
    delays.append(float(re.search("Del=(\d+\.\d+)", path).group(1)))

delays, paths = list(zip(*sorted(zip(delays, paths))))

unique_delays = sorted(list(set(delays)))
files_agg = [
    [y[1] for y in list(zip(delays, paths)) if (y[0] == x)] for x in unique_delays
]
data_agg = []

delay_c = np.arange(8, 10, 3e-3)
for i in range(len(files_agg)):
    dataframes = []
    mean_data = {}
    for j in range(len(files_agg[i])):
        data = pd.read_csv(files_agg[i][j], **read_kwargs)
        data.set_index("Delay", inplace=True)
        dataframes.append(data)
    mean_data = {}
    df_concat = pd.concat(dict(enumerate([df for df in dataframes])))

    mean_data["Delay"] = (
        pd.concat(dict(enumerate([df.index.to_series() for df in dataframes])))
        .groupby(level=1)
        .mean()
    )
    mean_data["X 166 Hz"] = df_concat["X 166 Hz"].groupby(level=1).mean()
    mean_data["Y 166 Hz"] = df_concat["Y 166 Hz"].groupby(level=1).mean()
    mean_data["X 333 Hz"] = df_concat["X 333 Hz"].groupby(level=1).mean()
    mean_data["Y 333 Hz"] = df_concat["Y 333 Hz"].groupby(level=1).mean()

    mean_data["SEM X 166 Hz"] = (
        df_concat["SEM X 166 Hz"].pow(2).groupby(level=1).mean().pow(1 / 2)
    )
    mean_data["SEM Y 166 Hz"] = (
        df_concat["SEM Y 166 Hz"].pow(2).groupby(level=1).mean().pow(1 / 2)
    )
    mean_data["SEM X 333 Hz"] = (
        df_concat["SEM X 333 Hz"].pow(2).groupby(level=1).mean().pow(1 / 2)
    )
    mean_data["SEM Y 333 Hz"] = (
        df_concat["SEM Y 333 Hz"].pow(2).groupby(level=1).mean().pow(1 / 2)
    )

    data_agg.append(pd.DataFrame(mean_data))

# Save data
for i in range(len(data_agg)):
    fn = "Processed data/pump_probe/2D Pump-induced/2D_pump-induced_Del={:.3f}.csv".format(
        unique_delays[i]
    )
    with open(fn, "w") as file:
        file.write("# Measured 11-13 February, generated with 2D_pump-induced.py\n")

        data_agg[i].to_csv(file, index=False)
