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
# title: Detailed look at three timescales
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


import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from mplstylize import colors, mpl_tools
from polars import col

from driven_nickelate.conductivity_mapping.susceptibility_evolution import (
    DATAFILES as SUSCEPTIBILITY_DATAFILES,
)
from driven_nickelate.config import paths as PROJECT_PATHS

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(10)

# %% [markdown]
# ## Experiment
#
# Waveforms transmitted through the sample is recorded over the parameter space constituted by the pump-probe delay (the common delay line of the optical gate and THz probe, with respect to the THz pump), and the pump field strength (controlled by a half-wave plate before the electro-optic crystal, which is followed by a polarizer), and the sample temperature. This analysis involves multiple experiments, taking place in the period 23-26 February 2023.
#
# We load the raw data from experiment, and pad with zeros to interpolate in the frequency domain. The dataset is structured as follows:

# %%
waveforms = SUSCEPTIBILITY_DATAFILES["waveforms_meas"].load()
display(waveforms)

# %% [markdown]
# Figure @fig-time-domain shows the raw transmitted waveforms.

# %%
# | label: fig-time-domain
# | fig-cap: Summary of raw waveforms, recorded after transmission through the sample for various values of the THz pump field strength, and the pump-probe delay.
g = sns.relplot(
    waveforms.filter(col("time").is_between(-2, 10)).with_columns(col("X") * 1e3),
    x="time",
    y="X",
    hue="field_strength",
    col="pp_delay",
    row="temperature",
    kind="line",
    height=3.4 / 3,
    aspect=2,
    facet_kws={
        "margin_titles": True,
        "despine": False,
        "gridspec_kws": {"wspace": 0.05, "hspace": 0.1},
    },
)
g.set_titles(row_template="{row_name} K", col_template="{col_name} ps")
g.set_xlabels(r"$t$ (ps)")
g.set_ylabels(r"$\mathrm{EOS}$ (mV)")

plt.show()

# %% [markdown]
# Next, we transform the time-traces to the frequency domain, and we consider the *relative* transmission amplitude to deconvolve the signal from the response function of the experimental apparatus (electro-optic crystal, mirrors, etc.). The relative transmission amplitude is defined as
#
# $$
#     \hat{t}_r(\tau,\mathscr{E}) \equiv \frac{\Delta\hat{t}}{\hat{t}_0} = \frac{\hat{S}(\tau,\mathscr{E})-\hat{S}_0}{\hat{S}_0}
# $$ {#eq-relative-transmission}
#
# Here, $\hat{S}$ is the discrete Fourier transform of a waveform, $\tau$ is the pump-probe delay, and $\mathscr{E}$ the pump field strength (unity corresponds to a peak electric field of about 200 kV/cm in the time domain). The dependence on frequency is omitted in the notation. $\hat{S}_0$ represents waveform with $\mathscr{E} = 0$.

# %%
trel_meas = SUSCEPTIBILITY_DATAFILES["trel_meas"].load()
display(trel_meas)

# %%
# | label: fig-trel-meas
# | fig-cap: Relative transmission amplitude.

sns.relplot(
    trel_meas.filter(col("freq").is_between(0.2, 2.2)),
    x="freq",
    y="t.reldiff.real",
    hue="field_strength",
    col="pp_delay",
    row="temperature",
    kind="line",
    height=3.4 / 3,
    aspect=2,
    facet_kws={
        "margin_titles": True,
        "despine": False,
        "gridspec_kws": {"wspace": 0.05, "hspace": 0.1},
    },
).set_titles(row_template="{row_name} K", col_template="{col_name} ps")

# %% [markdown]
# ## Calculations
#
# The calculated spectra are relativized with respect to the equilibrium spectrum, $\hat{S}_0$, which is the spectrum of the sample at the conductivity of the sample at the experimental temperature, determined in a separate experiment. This coincides with the $\mathscr{E} = 0$ spectra of the experiment, so that the reference spectra are the same for both measured and calculated relative transmission amplitudes.

# %%
trel_calc = SUSCEPTIBILITY_DATAFILES["trel_calc"].load()

# %% [markdown]
# ## Mapping
#
# To extract the resonator-gap conductivity $\sigma$ as a function of experimental parameters (the pump-probe delay $\tau$, the pump field strength $\mathscr{E}$, and the temperature $T$), we minimize the following error (see @eq-error):
#
# $$
#     \epsilon(T,\mathscr{E},\tau;\sigma) = \sum_{i}\left\lvert w(f_i)\left(\hat{t}_{r, \mathrm{meas}}(f_i; T, \mathscr{E},\tau) - \hat{t}_{r, \mathrm{calc}}(f_i; {\sigma})\right)\right\rvert^2
# $$ {#eq-error}
#
# Here $w$ is a weight that peaks around the resonance frequency. We verify in @fig-error that each experimental configuration has a global minimum in the error.

# %%
# | label: fig-error
# | fig-cap: The residual sum of squares (RSS) for all pump-probe time-delays ($\tau$), relative field strengths ($\mathscr{E}$), and temperatures ($T$).

error = SUSCEPTIBILITY_DATAFILES["error"].load()
g = sns.relplot(
    error,
    x="cond_gap",
    y="error",
    hue="field_strength",
    col="pp_delay",
    row="temperature",
    kind="line",
    height=3.4 / 3,
    aspect=2,
    facet_kws={
        "margin_titles": True,
        "despine": False,
        "subplot_kws": {"yscale": "log", "xscale": "log", "ylim": (1e-2, 1e2)},
        "gridspec_kws": {"wspace": 0.05, "hspace": 0.1},
    },
).set_titles(row_template="{row_name} K", col_template="{col_name} ps")

mpl_tools.breathe_axes(g.axes.flatten(), axis="y")

# %% [markdown]
# Selecting the least-error values for $\sigma$, we can plot the conductivity as a function of the experimental parameters, shown in @fig-sigma-summary.

# %%
# | label: fig-sigma-summary
# | fig-cap: The time evolution of the resonator gap conductivity $\sigma$, for a set of field strengths $\mathscr{E}$.

solution = SUSCEPTIBILITY_DATAFILES["solution"].load()
g = sns.relplot(
    solution,
    x="field_strength",
    y="cond_gap",
    col="pp_delay",
    row="temperature",
    hue="dataset",
    height=3.4 / 3,
    aspect=2,
    kind="line",
    marker="o",
    facet_kws={
        "margin_titles": True,
        "despine": False,
        "subplot_kws": {"yscale": "log"},
        "gridspec_kws": {"wspace": 0.1, "hspace": 0.1},
    },
)
sns.move_legend(g, "lower center", ncol=3, title=None)
g.set_xlabels(r"$\mathscr{E}$")
g.set_ylabels(r"$\sigma$ (S/m)")
g.set_titles(row_template="{row_name} K", col_template="{col_name} ps")
plt.show()


# %%
def fitting_func(x, a, b, c):
    return a * np.exp(-np.pi * b / x) + c


model = lm.Model(fitting_func)
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

g = sns.lineplot(
    fit_results,
    x="field_strength",
    y="cond_gap",
    hue="temperature",
    palette="Reds",
    legend=False,
)
sns.scatterplot(
    solution.filter(col("pp_delay") == 20),
    x="field_strength",
    y="cond_gap",
    hue="temperature",
    s=20,
    ec="white",
    palette="Reds",
)

g.set(xlabel=r"$E_0$ (kV/cm)", ylabel=r"$\sigma$ (S/m)")
g.set_title(r"$\tau = 20\,\mathrm{ps}$", loc="right", pad=5)
g.legend(title=r"$T$ (K)")

plt.show()

# %% [markdown]
# We take an extra look at the lowest temperature, considering only the field-strength and pump-probe delay dependence, shown in @fig-susceptibilities.

# %%
# | label: fig-susceptibilities
# | fig-cap: The time evolution of the resonator gap conductivity $\sigma$, for a set of field strengths $\mathscr{E}$.

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
plt.show()
