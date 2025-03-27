from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import svgutils.compose as sc
from cairosvg import svg2pdf
from lmfit.models import LinearModel, StepModel
from mplstylize import colors, mpl_tools
from polars import col
from scipy.integrate import cumulative_simpson

from driven_nickelate.conductivity_mapping.overlap import (
    DATAFILES as OVERLAP_DATAFILES,
)
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.simulations.gapfield import (
    DATAFILES as GAPFIELD_DATAFILES,
)

if __name__ == "__main__":
    data = pl.read_csv(
        PROJECT_PATHS.raw_data
        / "09.02.23/00h14m30s_Pump-induced (166 Hz) peak-track xPP_3.00-4.84_HWP=30.00_Sam=9.33.txt",
        comment_prefix="#",
        separator="\t",
        has_header=False,
        new_columns=["delay", "X"],
    ).select(
        ((2 / 0.299792) * (163.75 - col("delay"))).alias("time"),
        col("X") / col("X").max(),
    )

    model = StepModel(form="erf") * LinearModel()
    params = model.make_params(amplitude=data["X"].max(), center=-0.5, sigma=1)
    result = model.fit(data["X"], params, x=data["time"])
    tau_rise = 2 * 0.906194 * result.params["sigma"].value

    fig: plt.Figure = plt.figure(figsize=(6.8, 5.1))
    ax: dict[plt.Axes] = fig.subplot_mosaic([["1D", "1Dexpl"], ["2D", "2Dexpl"]])
    xlim = (-1, 4)
    sns.lineplot(data, x="time", y="X", ax=ax["1D"], c=colors("auburn"))
    sns.lineplot(data, x="time", y=result.best_fit, ls="--", ax=ax["1D"], c="black")

    ax["1D"].text(
        result.params["center"].value - 0.4,
        0.5,
        r"$\tau_\text{{rise}}$ = {:.1f} ps".format(tau_rise),
        ha="center",
        va="center",
        rotation=70,
    )
    ax["1D"].set(
        xlabel=r"$\tau$ (ps)",
        ylabel=r"$-\Delta E(t_0)$ (norm.)",
        xlim=xlim,
        ylim=(0, 1),
    )

    mapping = OVERLAP_DATAFILES["mapping"].load().filter(col("pp_delay") > -5)
    sns.lineplot(mapping, x="pp_delay", y="cond_gap", ax=ax["2D"], c=colors("auburn"))
    ax["2D"].set(xlabel=r"$\tau$ (ps)", ylabel=r"$\sigma$ (S/m)", xlim=xlim)

    time_data = (
        GAPFIELD_DATAFILES["temporal"]
        .load()
        .with_columns((col("time") - 1).alias("time"), col("Ey"))
    )
    idx = (time_data["cond"] - 1e4).abs().arg_min()
    time_data = time_data.filter(col("cond") == col("cond").slice(idx, 1).first())

    # - Cumulative field
    energy = cumulative_simpson(time_data["Ey"] ** 2, x=time_data["time"])
    energy /= time_data.select(col("time").last() - col("time").first()).item()

    twinx = ax["1D"].twinx()
    twinx.plot(time_data["time"][1:], energy, color=sns.colors.xkcd_rgb["denim"])
    twinx.set(ylabel=r"$U_\text{gap}$ norm.", ylim=(0, 1))
    twinx.annotate(
        "",
        xy=(1.5, 0.6),
        xytext=(2.0, 0.6),
        arrowprops=dict(
            arrowstyle="<|-",
            fc="k",
            lw=0.5,
        ),
    )
    ax["1D"].annotate(
        "",
        xy=(1.1, 0.85),
        xytext=(0.6, 0.85),
        arrowprops=dict(
            arrowstyle="<|-",
            fc="k",
            lw=0.5,
        ),
    )

    ax["1Dexpl"].axis("off")
    ax["2Dexpl"].axis("off")

    mpl_tools.breathe_axes([twinx, ax["1D"]], axis="y")
    mpl_tools.enumerate_axes(ax.values())

    x, y = 72 * fig.get_size_inches()
    sc.Figure(
        *[f"{x}pt", f"{y}pt"],
        sc.MplFigure(fig),
        sc.SVG(PROJECT_PATHS.figures / "illustrations/1DScan.svg")
        .scale(72 / 600)
        .move(20 + x / 2, -30),
        sc.SVG(PROJECT_PATHS.figures / "illustrations/2DScan.svg")
        .scale(72 / 600)
        .move(20 + x / 2, -30 + y / 2),
    ).save("_tmp.svg")

    svg2pdf(
        url="_tmp.svg",
        write_to=str(PROJECT_PATHS.figures / "fast_dynamics/1D-scan.pdf"),
    )
    Path("_tmp.svg").unlink()
