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
# title: Temperature
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
# | output: false

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import svgutils.compose as sc
from cairosvg import svg2pdf
from IPython.display import display
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from mplstylize import colors, mpl_tools
from polars import col
from polars_complex import ccol
from polars_dataset import Dataset

from driven_nickelate.conductivity_mapping.calculation import (
    DATAFILES as CALCULATION_DATAFILES,
)
from driven_nickelate.conductivity_mapping.temperature import (
    DATAFILES as TEMPERATURE_DATAFILES,
)
from driven_nickelate.conductivity_mapping.temperature_lossy import (
    DATAFILES as TEMPERATURE_LOSSY_DATAFILES,
)
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.linear_spectroscopy.esrr import (
    DATAFILES as MEASUREMENT_DATAFILES,
)
from driven_nickelate.linear_spectroscopy.film import (
    DATAFILES as FILM_DATAFILES,
)

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(10)

# %% [markdown]
# ## Film
# Let us start by considering the spectroscopy of a thin film, without the deposition of a gold metasurface. The average transmission amplitude is shown in @fig-film-temperature-dependence. The corresponding real conductivity is estimated from the transmission magnitude $|t|$,
#
# $$
#     \sigma = \frac{1 + n_s}{Z_0 d}\frac{1-|t|}{|t|}
# $$
#
# with film thickness $d = 10.8\,\mathrm{nm}$, and refractive index $n_s = 4.8$ of the substrate. The conductivity is shown on the secondary axis in @fig-film-temperature-dependence.

# %%
# | label: fig-film-temperature-dependence
# | fig-cap: The temperature dependence of the average transmission amplitude of the film, for both heating directions.

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
    col("eps.real", "eps.imag").complex.struct("eps[c]")
).select_data(
    (0.5 * (ccol("eps[c]").modulus() + ccol("eps[c]").real())).sqrt().alias("n")
)

data = data.join(lsat_model, on="freq")

sigma = (1 / col("t.mag") - 1) * (1 + col("n")) / (377 * 10.8e-9)
data_avg = (
    data.filter(col("freq") < 2.0)
    .group_by("temperature", "direction")
    .agg(col("t.mag").mean(), sigma.mean().alias("sigma"))
)

fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4))
palette = sns.color_palette(colors("lapiz_lazuli", "jasper"))
g = sns.lineplot(
    data_avg.with_columns(
        col("direction").replace_strict({1: "Warming", -1: "Cooling"})
    ),
    x="temperature",
    y="t.mag",
    hue="direction",
    palette=palette,
    marker="o",
    ax=ax[0],
)
g.set(xlabel="$T$ (K)", ylabel=r"$|t|$")
g.legend(loc="center left")

g = sns.lineplot(
    data_avg,
    x="temperature",
    y="sigma",
    hue="direction",
    palette=palette,
    ax=ax[1],
    legend=False,
)
g.set(xlabel="$T$ (K)", ylabel=r"$\sigma$ (S/m)")

data = pl.read_csv(PROJECT_PATHS.root / "characterization/transport/data.csv")

g = sns.lineplot(
    data=data.sample(fraction=0.1).filter(col("data_type") == "raw"),
    x="temperature",
    y="sigma",
    hue="direction",
    ax=ax[1],
    legend=False,
    ls="--",
    palette=palette,
)

ax[1].plot([], [], "k-.", label="Transport")
ax[1].plot([], [], "k", label="Spectroscopy")
ax[1].legend()

mpl_tools.enumerate_axes(ax)

fig.savefig(
    PROJECT_PATHS.figures
    / "linear_spectroscopy/blanket_film_temperature_dependence.pdf"
)

plt.show()

# %% [markdown]
# ## Metasurface
#
# After depositing a gold metasurface on the film, the transmission amplitude is shown in @fig-metasurface-spectra. This amplitude is defined as the ratio of the spectrum of a waveform transmitted through the sample $\hat{S}_{\mathrm{sam}}(T)$ to the spectrum of a waveform transmitted through a reference substrate $\hat{S}_{\mathrm{sub}}(T)$,
#
# $$
#     \hat{t} = \frac{\hat{S}_{\mathrm{sam}}(T)}{\hat{S}_{\mathrm{sub}}(T)}
# $$ {#eq-transmission-amplitude-meas}
#
# In panel **b** of Figure @fig-metasurface-spectra is shown the amplitude at 1 THz as a function of temperature for the two temperature sweep directions. Notice that no hysteresis is present.

# %%
# | label: fig-metasurface-spectra
# | fig-cap: The transmission amplitude spectra for a set of temperatures, for both heating directions (cooling, -1, and heating, +1).

spectra_meas = (
    MEASUREMENT_DATAFILES["spectra"]
    .load()
    .select_data(col("t.real", "t.imag").complex.struct())
)

spectra_diravg = (
    spectra_meas.sort("temperature")
    .group_by("freq")
    .agg((pl.all().gather_every(2) + pl.all().gather_every(2, 1)) / 2)
    .explode(pl.all().exclude("freq"))
    .drop("direction")
    .sort("temperature", "freq")
)
spectra_diravg = Dataset(spectra_diravg, index="freq", id_vars=["temperature"])

fig: plt.Figure = plt.figure(figsize=(6.8, 3.4))
ax: dict[plt.Axes] = fig.subplot_mosaic(
    [["cbar", "."], ["spectrum", "lines"]],
    height_ratios=(0.05, 1),
    width_ratios=(1, 0.333),
)
g = sns.lineplot(
    spectra_diravg.select_data(ccol("t[c]").modulus().alias("t.mag")).filter(
        col("freq").is_between(0.2, 2.3)
    ),
    x="freq",
    y="t.mag",
    hue="temperature",
    palette="Reds",
    legend=False,
    ax=ax["spectrum"],
).set(
    xlabel=r"$f$ (THz)",
    ylabel=r"$|t|$",
    ylim=(0, 1),
)
# ax["spectrum"].set_box_aspect(1)
cbar = fig.colorbar(
    ScalarMappable(
        cmap="Reds",
        norm=Normalize(*spectra_meas["temperature"].unique().sort()[[0, -1]]),
    ),
    cax=ax["cbar"],
    pad=0.01,
    label=r"$T$ (K)",
    orientation="horizontal",
    location="top",
)

sns.scatterplot(
    spectra_meas.select_data(ccol("t[c]").modulus().alias("t.mag"))
    .filter(col("freq").is_between(0.97, 1.03))
    .group_by("temperature", "direction")
    .agg(col("t.mag").mean())
    .filter(col("direction") < 0),
    x="temperature",
    y="t.mag",
    color="black",
    marker="o",
    s=20,
    hue="temperature",
    palette="Reds",
    legend=False,
    ax=ax["lines"],
).set(ylim=(0, 1), xlim=(0, 220), xticks=[50 * i for i in range(5)])
# ax["lines"].set_box_aspect(1)
ax["lines"].ticklabel_format(scilimits=(-3, 3))
ax["lines"].set(xlabel=r"$T$ (K)", ylabel=None, yticks=[])

ax.pop("cbar")
mpl_tools.enumerate_axes(ax.values())
mpl_tools.breathe_axes(ax.values(), axis="y")

fig.savefig(
    PROJECT_PATHS.figures
    / "linear_spectroscopy/metasurface_spectra_temperature_dependence.pdf"
)

plt.show()

# %% [markdown]
# We know from transport measurements that the sample exhibits a clear metal-to-insulator transition around 170 K. This transition temperature is not easily identifiable from @fig-metasurface-spectra **b**, but will become clear when we convert the transmission amplitude to conductivity.
#
# ## Calculations
#
# The transmission amplitude of the sample was calculated theoretically, assuming a homogeneous conductivity across the sample. A subset of the calculated transmission amplitude spectra is shown in @fig-spectra-calc.

# %%
spectra_calc = (
    CALCULATION_DATAFILES["spectra"]
    .load()
    .select_data(col("t.real", "t.imag").complex.struct("t[c]"))
)
display(spectra_calc.complex.unnest())

# %%
# | label: fig-spectra-calc
# | fig-cap: A subset of the calculated transmission amplitude spectra, for a set of conductivities.

spectra_calc = spectra_calc.filter((col("cond_gap") - col("cond_film")).abs() < 1e3)
cond_norm = LogNorm(*spectra_calc["cond_gap"].unique().sort()[[0, -1]])
g = sns.lineplot(
    spectra_calc.select_data(ccol("t[c]").modulus().alias("t.mag")),
    x="freq",
    y="t.mag",
    hue="cond_gap",
    hue_norm=cond_norm,
    palette="Spectral",
    legend=False,
    errorbar=None,
)
g.set(xlabel=r"$f$ (THz)", ylabel=r"$|t|$")
cbar = g.figure.colorbar(
    ScalarMappable(
        cmap="Spectral",
        norm=cond_norm,
    ),
    ax=g.axes,
    label=r"$\sigma$ (S/m)",
    orientation="horizontal",
    location="top",
)
g.axes.set_box_aspect(1)
g.figure.set_size_inches(3.4, 4.0)
plt.show()

# %% [markdown]
# We want to extract the conductivity $\sigma$ from the calculations, at each experimental temperature $T$. To enhance the match between calculation an experiment, it is helpful to consider the *relative transmission amplitude* rather than the absolute:
#
# $$
#     \hat{t}_r(T) \equiv \frac{\Delta\hat{t}}{\hat{t}_0} = \frac{\hat{S}(T) - \hat{S}_0}{\hat{S}_0}
# $$ {#eq-relative-transmission-meas}
#
# This removes any hidden response functions from the experiment that were not successfully deconvolved from the signal. We choose $\hat{S}_0 = \hat{S}(T_{\mathrm{min}})$ with $T_{\mathrm{min}}$ being the base temperature of the optical cryostat of about 10 kelvin.
#
# Similarly, for the calculation, the relative transmission amplitude is
#
# $$
#     \hat{t}_r(\sigma, \sigma_0) \equiv \frac{\Delta\hat{t}}{\hat{t}_0} = \frac{\hat{S}(\sigma) - \hat{S}(\sigma_0)}{\hat{S}(\sigma_0)}
# $$ {#eq-relative-transmission-calc}
#
# where $\sigma_0$ is the conductivity of the sample at the base temperature $T_{\mathrm{min}}$, which is not known *a priori*. By moving to this relative representation, we add one extra unknown to our problem, namely the value of $\sigma_0 = \sigma(|t|)$, which is a shared parameter for all temperatures $T$.

# %%
# | label: fig-data
# | fig-cap: To the left, the experimental relative spectra; to the right, the calculated relative spectra, for a fixed choice of $\sigma_0$.

trel_meas = TEMPERATURE_DATAFILES["trel_meas"].load()
trel_calc = TEMPERATURE_DATAFILES["trel_calc"].load()

fig = plt.figure(figsize=(6.8, 6.8))
ax = fig.subplot_mosaic(
    [["meas.real", "meas.imag"], ["calc.real", "calc.imag"]], sharey=True
)

cond0 = trel_calc.item(row=(trel_calc["cond0"] - 5e3).abs().arg_min(), column="cond0")
trel_calc = trel_calc.select("cond", "cond0", "freq", "^t.*$").filter(
    col("cond0").eq(cond0) & col("cond").is_in(col("cond").unique().gather_every(5))
)

cond_norm = LogNorm(*trel_calc.extrema("cond"))
temp_norm = Normalize(*trel_meas.extrema("temperature"))

for comp in ("real", "imag"):
    sns.lineplot(
        data=trel_meas,
        x="freq",
        y=f"t.reldiff.{comp}",
        hue="temperature",
        hue_norm=temp_norm,
        errorbar=None,
        legend=False,
        palette="Reds",
        ax=ax[f"meas.{comp}"],
    ).set(xlabel=r"$f$ (THz)", ylabel=f"{comp}" r"$(\hat{t}_r)$")
    sns.lineplot(
        data=trel_calc,
        x="freq",
        y=f"t.reldiff.{comp}",
        hue="cond",
        hue_norm=cond_norm,
        errorbar=None,
        legend=False,
        palette="Spectral",
        ax=ax[f"calc.{comp}"],
    ).set(xlabel=r"$f$ (THz)", ylabel=f"{comp}" r"$(\hat{t}_r)$")
    ax[f"meas.{comp}"].set_title("Measurement", loc="left", fontsize="small")
    ax[f"calc.{comp}"].set_title("Calculation", loc="left", fontsize="small")
fig.colorbar(
    ScalarMappable(norm=temp_norm, cmap="Reds"),
    ax=[ax["meas.real"], ax["meas.imag"]],
    label=r"$T$ (K)",
    location="right",
    pad=0.01,
).ax.ticklabel_format(scilimits=(0, 3))
fig.colorbar(
    ScalarMappable(norm=cond_norm, cmap="Spectral"),
    ax=[ax["calc.real"], ax["calc.imag"]],
    label=r"$\sigma$ (S/m)",
    location="right",
    pad=0.01,
)

plt.show()

# %% [markdown]
# We seek to find $\sigma_0$ and $\{\sigma\}_T$ that solves
#
# $$
#     \min_{\sigma_0} \frac{1}{N}\sum_{i=1}^N  \min_\sigma \,\lVert \hat{t}_{r, \mathrm{meas}}(T_i) - \hat{t}_{r, \mathrm{calc}}(\sigma,\sigma_0)
# $$ {#eq-minimize}
#
# where $\lVert\cdot\rVert$ is the norm over the frequency domain. The calculations of the relative transmission amplitude are performed on a dense grid of $\sigma$, and cubic interpolation is used to estimate the relative transmission amplitude at arbitrary conductivities.
#
# To find the optimal reference conductivity ($\sigma_0$), we perform the inner minimization for each candidate $\sigma_0$ to obtain the optimal set $\{\sigma\}_T$, in the sense of minimizing the residual sum of squares. This gives us an outer error measure $\varepsilon_{\sigma_0}$ for each candidate $\sigma_0$,
#
# $$
#     \varepsilon(\sigma_0) = \frac{1}{N}\sum_{i=1}^N  \min_\sigma \,\lVert \hat{t}_{r, \mathrm{meas}}(T_i) - \hat{t}_{r, \mathrm{calc}}(\sigma,\sigma_0)\rVert
# $$ {#eq-outer-error}
#
# which we plot in @fig-error-cond0.

# %%
# | label: fig-error-cond0
# | fig-cap: The aggregate error as defined in @eq-outer-error, as a function of the reference conductivity $\sigma_0$.

error_cond0 = TEMPERATURE_DATAFILES["error_cond0"].load()
root = error_cond0["cond0_best"][0]

fig, ax = plt.subplots(figsize=(3.4, 3.4))

sns.lineplot(
    error_cond0,
    x="cond0",
    y="error",
    ax=ax,
).set(xscale="log", xlabel=r"$\sigma_0$ (S/m)", ylabel=r"$\varepsilon(\sigma_0)$")

ax.axvline(x=root, color="k", linestyle="--", lw=0.5)
ax.annotate(
    f"{root:.2e}",
    xy=(root, error_cond0["error"].mean()),
    annotation_clip=False,
    rotation=90,
    ha="right",
    fontsize="small",
)
plt.show()

# %% [markdown]
# Fortunately this has a global minimum, and we have a good estimate for the reference conductivity $\sigma_0$. Fixing this $\sigma_0$, we can go back and explore the error landscape in the space of $\sigma$,
#
# $$
#     \varepsilon(\sigma, T) = \,\lVert \hat{t}_{r, \mathrm{meas}}(T) - \hat{t}_{r, \mathrm{calc}}(\sigma)\rVert_2
# $$ {#eq-error-element}

# %%
# | label: fig-error-cond
# | fig-cap: The aggregate error as defined in @eq-outer-error, as a function of the reference conductivity $\sigma$.

error = TEMPERATURE_DATAFILES["error"].load()

norm = Normalize(*error.get_column("temperature")[[0, -1]])
g = sns.lineplot(
    error.select("cond", "temperature", "error").unique(),
    x="cond",
    y="error",
    hue="temperature",
    hue_norm=norm,
    palette="Reds",
    legend=False,
    errorbar=None,
)
g.axes.set_box_aspect(1)
g.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\sigma$ (S/m)",
    ylabel=r"$\varepsilon(\sigma, T)$",
)
cbar = g.figure.colorbar(
    ScalarMappable(cmap="Reds", norm=norm), ax=g.axes, label=r"$T$ (K)"
)
g.figure.set_size_inches(3.4, 2.6)
plt.show()

# %% [markdown]
# The error is larger at higher temperatures, since that is where the relative transmission amplitude is also the largest. But there is at least one minimum for each temperature, and we get good estimates for the conductivity at each temperature. There is a local minimum around $\sigma = 3\times10^3$ which is temperature independent, and therefore not physical. We therefore restrict the minima to $\sigma > 3\times10^3$, and obtain unique solutions.
#
# We inspect the $\hat{t}_r$ experimental and calculated distributions, and the corresponding residuals, in @fig-data.

# %%
# | label: fig-solution
# | fig-cap: Solution to @eq-minimize. To the left, the experimental relative spectra; in the middle the calculated relative spectra, and to the right the residuals.


def plot_lineplot(data, y, ax, **kwargs):
    sns.lineplot(
        data,
        x="freq",
        y=y,
        hue="temperature",
        ax=ax,
        errorbar=None,
        legend=False,
        palette="Reds",
        **kwargs,
    ).set(ylim=(-1, 1))


solution = TEMPERATURE_DATAFILES["solution"].load()

fig = plt.figure(figsize=(6.8, 3.4))
ax = fig.subplot_mosaic(
    [
        ["meas_real", "calc_real", "resid_real", "cbar"],
        ["meas_imag", "calc_imag", "resid_imag", "cbar"],
    ],
    sharey=False,
    width_ratios=(1, 1, 1, 0.05),
)
norm = Normalize(*solution.get_column("temperature").sort()[[0, -1]])
for comp in ("real", "imag"):
    plot_lineplot(
        solution,
        f"t_meas.reldiff.{comp}",
        ax[f"meas_{comp}"],
    )
    plot_lineplot(solution, f"t_calc.reldiff.{comp}", ax[f"calc_{comp}"], hue_norm=norm)

    residual_data = solution.with_columns(
        (col("t_meas.reldiff.real") - col("t_calc.reldiff.real")).alias(
            "residual.real"
        ),
        (col("t_meas.reldiff.imag") - col("t_calc.reldiff.imag")).alias(
            "residual.imag"
        ),
    )
    plot_lineplot(residual_data, f"residual.{comp}", ax[f"resid_{comp}"])

mpl_tools.breathe_axes(ax.values())

cbar = fig.colorbar(
    ScalarMappable(cmap="Reds", norm=norm), label=r"$T$ (K)", cax=ax["cbar"]
)

plt.show()

# %% [markdown]
# The residuals are small, and the match between calculation and experiment is acceptable, aside from a systematic error related to an overall slope. The final mapping from temperature to conductivity is shown in @fig-mapping.

# %%
# | label: fig-mapping
# | fig-cap: The mapping that solves @eq-minimize.

fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4))

for i, scale in enumerate(("linear", "log")):
    sns.lineplot(
        solution.select("temperature", "direction", "freq", "cond").unique(),
        x="temperature",
        y="cond",
        hue="direction",
        palette="Set2",
        marker="o",
        ax=ax[i],
    )
    ax[i].ticklabel_format(scilimits=(0, 3))
    ax[i].set(xlabel=r"$T$ (K)", ylabel=r"$\sigma$ (S/m)", yscale=scale)

plt.show()

# %% [markdown]
# ## Metasurface with finite scattering time
#
# The above analysis uses a constant, and real, conductivity $\sigma$ for the film. In reality, the conductivity is frequency dependent,
#
# $$
#     \sigma(\omega) = \frac{\sigma_\text{dc}}{1 - i\omega\tau}
# $$
#
# where $\sigma_\text{dc}$ is the DC conductivity, and $\tau$ is an electronic scattering time. Again, using the relative transmission amplitude requires us to find the reference values $\sigma_0$ and $\tau_0$ that minimizes the outer error as defined in @eq-outer-error. A colormap of this error is shown in @fig-error-cond0-lossy.

# %%
# | label: fig-error-cond0-lossy
# | fig-cap: Color map of the logarithm of the aggregate error as defined in @eq-outer-error, as a function of the reference conductivity $\sigma_0$ and scattering time $\tau_0$. A clear minimum is present at $\tau_0 \approx 60\,\text{fs}$ and $\sigma_0 = 7e4\,\text{S/m}$. This is used as the reference parameters for the subsequent analysis, and the scattering time $\tau_0$ is assumed equal at all temperatures.

error0 = TEMPERATURE_LOSSY_DATAFILES["error_cond0"].load()

fig = plt.figure(figsize=(3.4, 3.4))
g = sns.scatterplot(
    error0.with_columns((col("error") - col("error").min() + 1e-4).log()),
    x="cond0",
    y="tau0",
    hue="error",
    ec=None,
    palette="Reds_r",
    legend=False,
)
g.set(xscale="log", xlabel=r"$\sigma_0$ (S/m)", ylabel=r"$\tau_0$ (s)")
plt.scatter(
    *error0.select(col("cond0", "tau0").sort_by("error").first()),
    c="white",
    ec="black",
    lw=0.6,
    marker="D",
    s=50,
)
plt.show()

# %% [markdown]
# Having identified a minimum for the reference parameters, we proceed to fix $\tau = \tau_0$ across all temperatures, and select the minimizing values for $\sigma$ at each temperature. The resulting mapping is shown in @fig-mapping-lossy.

# %%
# | label: fig-mapping-lossy
# | fig-cap: The mapping that solves @eq-minimize.

solution = TEMPERATURE_LOSSY_DATAFILES["solution"].load()

fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4))

for i, scale in enumerate(("linear", "log")):
    sns.lineplot(
        solution.select("temperature", "direction", "freq", "cond").unique(),
        x="temperature",
        y="cond",
        hue="direction",
        palette="Set2",
        marker="o",
        ax=ax[i],
    )
    ax[i].ticklabel_format(scilimits=(0, 3))
    ax[i].set(xlabel=r"$T$ (K)", ylabel=r"$\sigma$ (S/m)", yscale=scale)

plt.show()

# %%
# | label: fig-mapping-lossy-linear
# | fig-cap: The mapping that solves @eq-minimize.

solution = TEMPERATURE_LOSSY_DATAFILES["solution"].load()

fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))

g = sns.lineplot(
    solution.select("temperature", "direction", "freq", "cond").unique(),
    x="temperature",
    y="cond",
    hue="direction",
    palette="Set2",
    marker="o",
    ax=ax,
)
g.legend(title="Sweep direction")
ax.set(xlabel=r"$T$ (K)", ylabel=r"$\sigma$ (S/m)")

fig.savefig(
    PROJECT_PATHS.figures / "linear_spectroscopy/metasurface_temperature_dependence.pdf"
)
plt.show()

# %%
# | label: fig-film-metasurface
# | fig-cap: The temperature dependence of the average transmission amplitude of the film, for both heating directions.

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
    col("eps.real", "eps.imag").complex.struct("eps[c]")
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

plt.show()

# %% [markdown]
# Overall, we find that including this finite scattering time slightly improves the match between calculation and experiment (i.e. a slightly smaller residual), particularly in the imaginary part of the transmission amplitude.

# %%
# | label: fig-metasurface-film
# | fig-cap: Comparison of film (left) and metasurface (right).


def plot_lineplot(data, y, ax, **kwargs):
    sns.lineplot(
        data,
        x="freq",
        y=y,
        hue="temperature",
        ax=ax,
        errorbar=None,
        legend=False,
        palette="Reds",
        **kwargs,
    ).set(ylim=(-1, 1))


solution = TEMPERATURE_LOSSY_DATAFILES["solution"].load()

fig = plt.figure(figsize=(6.8, 3.4))
ax = fig.subplot_mosaic(
    [
        ["meas_real", "calc_real", "resid_real", "cbar"],
        ["meas_imag", "calc_imag", "resid_imag", "cbar"],
    ],
    sharey=False,
    width_ratios=(1, 1, 1, 0.05),
)
norm = Normalize(*solution.get_column("temperature").sort()[[0, -1]])
for comp in ("real", "imag"):
    plot_lineplot(
        solution,
        f"t_meas.reldiff.{comp}",
        ax[f"meas_{comp}"],
    )
    plot_lineplot(solution, f"t_calc.reldiff.{comp}", ax[f"calc_{comp}"], hue_norm=norm)

    residual_data = solution.with_columns(
        (col("t_meas.reldiff.real") - col("t_calc.reldiff.real")).alias(
            "residual.real"
        ),
        (col("t_meas.reldiff.imag") - col("t_calc.reldiff.imag")).alias(
            "residual.imag"
        ),
    )
    plot_lineplot(residual_data, f"residual.{comp}", ax[f"resid_{comp}"])

mpl_tools.breathe_axes(ax.values())

cbar = fig.colorbar(
    ScalarMappable(cmap="Reds", norm=norm), label=r"$T$ (K)", cax=ax["cbar"]
)

plt.show()
