import os
from pathlib import Path

os.chdir(Path(__file__).parents[1])

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dataset_tools import (
    Dataset,
    load_simulation,
)
from matplotlib import style
from polars import col as c

style.use("publication.mplstylize")

datasets = []


path_pairs = [
    (
        Path("simulations/data/31.10.23/2023-10-31_10h06m50s_spectral_data.csv"),
        Path("simulations/data/31.10.23/spectral_data_reference.csv"),
    ),
    (
        Path("simulations/data/30.10.23/2023-10-30_12h17m02s_spectral_data.csv"),
        Path("simulations/data/30.10.23/spectral_data_reference.csv"),
    ),
    (
        Path("simulations/data/26.10.23/2023-10-26_17h14m05s_spectral_data.csv"),
        Path("simulations/data/26.10.23/spectral_data_reference.csv"),
    ),
    (
        Path("simulations/data/24.10.23/2023-10-24_14h14m38s_spectral_data.csv"),
        Path("simulations/data/24.10.23/spectral_data_reference.csv"),
    ),
]

fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.set(xscale="log", yscale="log")

common_frequencies = pl.Series("freq", np.arange(0.2 + 1e-2, 2.2, 1e-2))
for path, path_ref in path_pairs:
    ds = (
        Dataset(
            load_simulation(path, path_ref).collect(),
            index="freq",
            id_vars=["cond_gap", "cond_film"],
        )
        .select(c("freq"), c("cond_gap"), c("cond_film"), c("^t..*$"))
        .with_columns(
            c("freq").count().over("cond_gap", "cond_film").alias("num_freqs")
        )
        .filter(c("num_freqs").eq(c("num_freqs").max()))
        .drop("num_freqs")
        .unique()
        .sort("cond_gap", "cond_film", "freq")
        .regrid(common_frequencies)
    )
    sns.scatterplot(
        ds.select("cond_gap", "cond_film").unique(),
        x="cond_gap",
        y="cond_film",
        s=2,
        ax=ax,
    )
    datasets.append(ds)
