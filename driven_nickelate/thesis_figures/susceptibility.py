import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from lmfit import Model
from mplstylize import colors, mpl_tools
from polars import col

from driven_nickelate.conductivity_mapping.susceptibility_evolution import (
    DATAFILES as SUSCEPTIBILITY_DATAFILES,
)
from driven_nickelate.config import paths as PROJECT_PATHS


def fitting_func(x, a, b, c):
    return a * np.exp(-np.pi * b / x) + c


if __name__ == "__main__":
    solution = SUSCEPTIBILITY_DATAFILES["solution"].load()

    model = Model(fitting_func)
    params = model.make_params(a=1e4, b=50, c=1e4)
    params["b"].set(min=1, max=150)

    field_strength = pl.Series("field_strength", np.linspace(0, 200, 1_000))

    def fit_group(group):
        result = model.fit(group["cond_gap"], x=group["field_strength"], params=params)
        result_df = pl.DataFrame(
            {
                "pp_delay": group.select(col("pp_delay").first()).item(),
                "temperature": group.select(col("temperature").first()).item(),
                "field_strength": field_strength,
                "cond_gap": result.eval(x=field_strength),
                "field_thresh": result.params["b"].value,
                "cond0": result.params["c"].value,
                "amplitude": result.params["a"].value,
            }
        )
        results.append(result_df)
        return group.with_columns(
            best_fit=result.best_fit,
        )

    results = []
    solution.filter(col("pp_delay") == 20).group_by("temperature").map_groups(fit_group)
    fit_results = pl.concat(results)

    fig: plt.Figure = plt.figure(figsize=(6.8, 3.4))
    axes: dict[plt.Axes] = fig.subplot_mosaic([["temperature", "pp_delay"]])

    solution = (
        SUSCEPTIBILITY_DATAFILES["solution"]
        .load()
        .with_columns((col("cond_gap") - col("cond_film").min()).alias("delta_cond"))
        .filter(col("delta_cond") / col("cond_film") > 0.05)
    )
    ylabel = r"$\sigma$ (S/m)"
    xlabel = r"$E_0$ (kV/cm)"

    palette = sns.color_palette(colors("lapiz_lazuli", "caramel", "cambridge_blue"))
    g = sns.lineplot(
        solution.filter(col("temperature").eq(col("temperature").min())),
        x="field_strength",
        y="cond_gap",
        hue="pp_delay",
        legend=True,
        marker="o",
        palette=palette,
        ax=axes["pp_delay"],
    )
    g.set(xlabel=xlabel, ylabel=ylabel, yscale="linear")
    g.set_title(r"$T = 5\,\mathrm{K}$", loc="right", pad=5)
    g.legend(title=r"$\tau$ (ps)")

    palette = sns.blend_palette(["#feeace", "#b62e2b"], as_cmap=True)
    g = sns.lineplot(
        fit_results,
        x="field_strength",
        y="cond_gap",
        hue="temperature",
        palette=palette,
        legend=False,
        ax=axes["temperature"],
    )
    sns.scatterplot(
        solution.filter(col("pp_delay") == 20),
        x="field_strength",
        y="cond_gap",
        hue="temperature",
        palette=palette,
        ax=axes["temperature"],
        zorder=np.inf,
        ec="white",
        s=20,
    )
    g.set(xlabel=xlabel, ylabel=ylabel)
    g.set_title(r"$\tau = 20\,\mathrm{ps}$", loc="right", pad=5)
    g.legend(title=r"$T$ (K)")

    mpl_tools.enumerate_axes(axes.values())

    fig.savefig(PROJECT_PATHS.figures / "susceptibility/xPP_xT.pdf")
