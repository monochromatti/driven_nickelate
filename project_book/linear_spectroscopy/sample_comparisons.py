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
# ---

# %% [markdown]
# ---
# title: Sample comparisons
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .qmd
#       format_name: quarto
#       format_version: '1.0'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
#| output: false

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from polars import col
from polars_complex import ccol
from driven_nickelate.conductivity_mapping.calculation import (
    DATAFILES as CALCULATION_DATAFILES,
)
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.linear_spectroscopy.three_samples import (
    DATAFILES as SAMPLE_DATAFILES,
)
from IPython.display import display
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from polars_dataset import Dataset
from mplstylize import mpl_tools, colors, color_maps

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(10)

# %%
#| label: fig-sample-comparisons
#| fig-cap: Comparison of the transmission spectra of **a**. Resonators on thin film. **b**. Resonators on a bare substrate. Transmission through a blanket substrate is used as reference.

data = SAMPLE_DATAFILES["spectra"].load().filter(col("freq").is_between(0.2, 2.3))

fig = plt.figure(figsize=(6.8, 2.6))
ax = fig.subplot_mosaic(
    [["film_esrr", "esrr", "calc"]],
)

norm = plt.Normalize(0, 200)

sample_plot_kwargs = {
    "x": "freq",
    "y": "t.mag",
    "hue": "temperature",
    "hue_norm": norm,
    "palette": sns.blend_palette(colors("citron", "dark_cyan"), as_cmap=True),
}
film_esrr = (
    data.filter(col("sample") == "esrr")
    .with_columns(ccol("t.real", "t.imag").alias("t[c]"))
    .with_columns(ccol("t[c]").modulus().alias("t.mag"))
    .group_by("freq", "temperature")
    .agg(col("t.mag").mean())
    .filter(col("temperature").is_in(data.extrema("temperature")))
)
g = sns.lineplot(
    film_esrr,
    legend=True,
    ax=ax["film_esrr"],
    **sample_plot_kwargs,
)
g.set(ylim=(0, 1), xlabel="$f$ (THz)", ylabel="$|t|$")
g.legend(title=r"$T$ (K)")


lsat_esrr = (
    data.filter(col("sample") == "lsat_esrr")
    .with_columns(ccol("t.real", "t.imag").alias("t[c]"))
    .with_columns(ccol("t[c]").modulus().alias("t.mag"))
    .group_by("freq")
    .agg(col("temperature").gather([0, -1]), col("t.mag").gather([0, -1]))
    .explode("temperature", "t.mag")
)
g = sns.lineplot(lsat_esrr, legend=False, ax=ax["esrr"], **sample_plot_kwargs)
g.set(ylim=(0, 1), xlabel="$f$ (THz)", yticklabels=[], ylabel=None)

spectra_calc = (
    CALCULATION_DATAFILES["spectra"]
    .load()
    .select_data(ccol("t.real", "t.imag").alias("t[c]"))
)

spectra_calc = spectra_calc.filter((col("cond_gap") - col("cond_film")).abs() < 1e3)
cond_norm = LogNorm(1e2, 1e5)
cond_palette = sns.cubehelix_palette(
    start=-1, rot=-3, dark=0.8, light=0.3, as_cmap=True
)
sns.lineplot(
    spectra_calc.select_data(ccol("t[c]").modulus().alias("t.mag")),
    x="freq",
    y="t.mag",
    hue="cond_gap",
    hue_norm=cond_norm,
    palette=cond_palette,
    legend=False,
    errorbar=None,
    ax=ax["calc"],
).set(ylim=(0, 1), xlabel="$f$ (THz)", ylabel=None, yticklabels=[])

cbar_ax = ax["calc"].inset_axes([0.52, 0.2, 0.4, 0.03])
fig.colorbar(
    ScalarMappable(
        cmap=cond_palette,
        norm=cond_norm,
    ),
    cax=cbar_ax,
    orientation="horizontal",
    pad=0,
)
cbar_ax.set_xlabel(
    r"$\sigma$ (S/m)",
    rotation=0,
    loc="center",
    va="top",
)
cbar_ax.xaxis.set_ticks_position("top")
cbar_ax.set(xticks=np.geomspace(1e2, 1e5, 4))

for _ax in ax.values():
    _ax.set_xticks(np.arange(0.5, 2.3, 0.5))

mpl_tools.enumerate_axes(ax.values())
mpl_tools.breathe_axes(ax.values(), axis="y")
mpl_tools.square_axes(ax.values())

fig.savefig(PROJECT_PATHS.figures / "linear_spectroscopy/sample_comparisons.pdf")

plt.show()

# %%
data = SAMPLE_DATAFILES["spectra"].load().filter(col("freq").is_between(0.2, 2.3))
lsat_esrr = (
    data.filter(col("sample") == "lsat_esrr")
    .with_columns(pl.col("t.real", "t.imag").complex.struct("t[c]"))
    .with_columns(ccol("t[c]").modulus().alias("t.mag"))
    .group_by("freq")
    .agg(col("temperature").gather([0, -1]), col("t.mag").gather([0, -1]))
    .explode("temperature", "t.mag")
)

lsat_esrr.group_by("freq").mean().drop("temperature").sort("freq").write_csv(
    PROJECT_PATHS.root / "simulations/comsol/lsat_esrr.csv"
)


film_esrr = (
    (
        data.filter(col("sample") == "esrr")
        .with_columns(pl.col("t.real", "t.imag").complex.struct("t[c]"))
        .with_columns(ccol("t[c]").modulus().alias("t.mag"))
        .group_by("freq", "temperature")
        .agg(col("t.mag").mean())
    )
    .filter(col("temperature").eq(col("temperature").min()))
    .sort("freq")
)

film_esrr.write_csv(PROJECT_PATHS.root / "simulations/comsol/film_esrr.csv")
