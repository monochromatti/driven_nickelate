# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#     path: /nix/store/ndw01kv30vabg62apaa655lyl0s4lys2-python3-3.12.9-env/share/jupyter/kernels/python3
# ---

# %% [markdown]
# ---
# title: "Cavity polarization"
# ---

# %%
# | output: false


import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from mplstylize import mpl_tools

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.tools import create_dataset, pl_fft

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(10)

# %%
fnames = [
    "10h30m08s_F21059_eSRR THz probe chopped xHWP xPP vertical pump blocked_2.99-4.04_Del=163.13_HWP=30.00.txt",
    "10h40m43s_F21059_eSRR THz probe chopped xHWP xPP vertical pump_2.99-4.05_Del=163.13_HWP=30.00.txt",
    "09h48m44s_F21059_eSRR THz probe chopped xHWP xPP_2.99-4.06_Del=163.13_HWP=55.00.txt",
]
paths = [str(PROJECT_PATHS.raw_data / "09.03.23" / fname) for fname in fnames]
labels = ["no_pump", "aligned_pump", "crossed_pump"]

paths = pl.DataFrame(zip(labels, paths), schema={"label": pl.Utf8, "path": pl.Utf8})

data = (
    create_dataset(
        paths,
        column_names=["delay", "X", "Y"],
        index="delay",
        lockin_schema={"X": ("X", "Y")},
    )
    .with_columns((6.67 * (9.67 - pl.col("delay"))).alias("time"))
    .set(index="time")
    .drop("delay")
    .sort_columns()
)

time = pl.Series("time", np.arange(-10, 100, 6e-3))

data = data.regrid(time, method="linear", fill_value=0)

data_ref = data.filter(pl.col("label") == "no_pump").drop("label")

data = (
    data.join(data_ref, on="time", how="inner", suffix="_ref")
    .with_columns((pl.col("X") - pl.col("X_ref")).alias("dX"))
    .filter(pl.col("label") != "no_pump")
)


data_fft = pl_fft(data, xname="time", id_vars=["label"])
data_fft = (
    data_fft.complex.struct()
    .with_columns(
        pl.col("dX[c]").complex.divide(pl.col("X_ref[c]")).alias("t.reldiff[c]"),
    )
    .with_columns(pl.col("t.reldiff[c]").complex.modulus().alias("t.reldiff.mag"))
)

import lmfit as lm

model = lm.models.SineModel(prefix="osc_")
model *= lm.models.ExponentialModel(prefix="exp_")
model *= lm.models.StepModel(prefix="step_")
params = model.make_params(
    osc_amplitude=1e-3, osc_frequency=6, step_sigma=2, exp_amplitude=1, exp_decay=5
)
params["exp_amplitude"].set(vary=0)

offset = -1e-3
for (label,), group in data.filter(pl.col("time").is_between(-5, 11)).group_by(
    ["label"], maintain_order=True
):
    x, y = group.select("time", "dX").get_columns()
    result = model.fit(y, params, x=x)
    print(result.fit_report())
    plt.plot(x, offset + y, label=label)
    plt.plot(x, offset + result.best_fit, label=f"{label} fit", c="black")
    offset += 2e-3


# %%
sns.set_palette("Set2")
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4))
g = sns.lineplot(
    data.filter(
        pl.col("time").is_between(-2, 11)
        & pl.col("label").is_in(["crossed_pump", "aligned_pump"])
    )
    .with_columns(pl.col("label").cum_count().over("time").alias("id"))
    .with_columns(
        pl.col("label").replace(
            {
                "aligned_pump": "$y$, aligned",
                "crossed_pump": "$x$, crossed",
            }
        )
    ),
    x="time",
    y="dX",
    hue="label",
    ax=ax[0],
)
g.set(ylim=(-3e-3, 3e-3), xlabel="$t$ (ps)", ylabel=r"$\Delta V_\text{EO}$ (V)")
g.legend(title="Pump polarization")

g = sns.lineplot(
    data_fft.select("freq", "t.reldiff.mag", "label").filter(
        pl.col("freq").is_between(0.2, 2.0)
    ),
    x="freq",
    y="t.reldiff.mag",
    hue="label",
    ax=ax[1],
    legend=False,
)
g.set(ylim=(0, 1), xlabel="$f$ (THz)", ylabel=r"$|\Delta t/t|$")

mpl_tools.enumerate_axes(ax)

fig.savefig(PROJECT_PATHS.figures / "pump/polarization.pdf")
