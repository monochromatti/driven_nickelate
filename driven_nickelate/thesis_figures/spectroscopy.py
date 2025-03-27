from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import svgutils.compose as sc
from cairosvg import svg2pdf
from mplstylize import colors, mpl_tools
from polars import col
from polars_complex import ccol
from polars_dataset import Dataset

from driven_nickelate.conductivity_mapping.temperature_lossy import (
    DATAFILES as TEMPERATURE_LOSSY_DATAFILES,
)
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.linear_spectroscopy.film import (
    DATAFILES as FILM_DATAFILES,
)

data = (
    FILM_DATAFILES["spectra"]
    .load()
    .with_columns(
        col("direction").cast(pl.Int16),
    )
)

lsat_model = pl.read_csv(PROJECT_PATHS.root / "literature_data/lsat_model.csv")
lsat_model = Dataset(lsat_model.rename({"f (THz)": "freq"}), index="freq")
lsat_model = lsat_model.regrid(data["freq"].unique().sort())
lsat_model = lsat_model.with_columns(
    ccol("eps.real", "eps.imag").alias("eps[c]", fields="eps")
).select_data(
    (0.5 * (ccol("eps[c]").modulus() + ccol("eps[c]").real())).sqrt().alias("n")
)

data = data.join(lsat_model, on="freq")

sigma = (1 / col("t.mag") - 1) * (1 + col("n")) / (377 * 10.8e-9)
data_avg = (
    data.filter(col("freq") < 2.0)
    .group_by("temperature", "direction")
    .agg(col("t.mag").mean(), sigma.mean().alias("sigma"))
    .with_columns(
        col("direction").replace_strict({1: "Warming", -1: "Cooling"}),
    )
)

fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4), sharex=True)
palette = sns.color_palette(colors("lapiz_lazuli", "jasper"))

g = sns.lineplot(
    data_avg,
    x="temperature",
    y="t.mag",
    hue="direction",
    palette=palette,
    marker="o",
    ax=ax[0],
)
g.legend(title="")
g.set(xlabel="$T$ (K)", ylabel=r"$|t|$")


solution = TEMPERATURE_LOSSY_DATAFILES["solution"].load()

g = sns.lineplot(
    solution.select("temperature", "direction", "freq", "cond")
    .unique()
    .sort("direction")
    .with_columns(col("direction").replace_strict({1: "Warming", -1: "Cooling"})),
    x="temperature",
    y="cond",
    hue="direction",
    palette=palette,
    marker="o",
    legend=False,
    ax=ax[1],
)
g.set(xlabel=r"$T$ (K)", ylabel=r"$\sigma$ (S/m)")

mpl_tools.enumerate_axes(ax)

x, y = fig.get_size_inches()
x, y = x * 72, y * 72
scale = 72 / 600 / 4
sc.Figure(
    *[f"{x}pt", f"{y}pt"],
    sc.MplFigure(fig),
    sc.SVG(PROJECT_PATHS.figures / "illustrations/no_resonator.svg")
    .scale(scale)
    .move(0.1 * x, 0.1 * y),
    sc.SVG(PROJECT_PATHS.figures / "illustrations/one_resonator.svg")
    .scale(scale)
    .move(x / 2 + 0.1 * x, 0.1 * y),
).save("_tmp.svg")

svg2pdf(
    url="_tmp.svg",
    write_to=str(
        PROJECT_PATHS.figures / "linear_spectroscopy" / "film_vs_metasurface.pdf"
    ),
)

Path("./_tmp.svg").unlink()
