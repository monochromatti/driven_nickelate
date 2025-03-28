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
# title: "Nanosecond dynamics"
# ---

# %%
# | output: false

import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, SymLogNorm
from polars import col

from driven_nickelate.conductivity_mapping.susceptibility_long import (
    DATAFILES as SUSCEPTIBILITY_DATAFILES,
)

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(10)

# %% [markdown]
# ## Experiment
#
# Waveforms transmitted through the sample is recorded over the parameter space constituted by the pump-probe delay (the common delay line of the optical gate and THz probe, with respect to the THz pump), and the pump field strength (controlled by a half-wave plate before the electro-optic crystal, which is followed by a polarizer).
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


def label(x, color, label):
    ax = plt.gca()
    ax.text(
        0.05,
        0,
        rf"$\mathscr{{E}}_r$ = {x.iloc[0]:.2f}",
        color="gray",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
        fontsize=5,
    )


delay_norm = SymLogNorm(1, 1, *waveforms.extrema("pp_delay"))
g = sns.FacetGrid(
    waveforms.filter(col("time").is_between(-1, 8)),
    row="field_strength",
    hue="pp_delay",
    despine=False,
    sharex=True,
    sharey=True,
    palette="Spectral",
    gridspec_kws={"hspace": 0},
)
g.map_dataframe(
    sns.lineplot,
    x="time",
    y="X",
    errorbar=None,
    hue_norm=delay_norm,
    linewidth=0.5,
)
g.figure.set_size_inches(6.8, 6.8)

g.map(label, "field_strength")
colorbar = g.fig.colorbar(
    plt.cm.ScalarMappable(norm=delay_norm, cmap="Spectral"),
    ax=g.axes,
    label=r"$\tau$ (ps)",
    aspect=20,
    shrink=0.5,
    location="right",
)
g.despine(bottom=True, left=True)
g.set_titles("")
g.set(xticks=[], yticks=[], ylabel="", xlabel="")
plt.show()

# %% [markdown]
# Next, we transform the time-traces to the frequency domain, and we consider the *relative* transmission amplitude to deconvolve the signal from the response function of the experimental apparatus (electro-optic crystal, mirrors, etc.). The relative transmission amplitude is defined as
#
# $$
#     \hat{t}_r(\tau,\mathscr{E}) \equiv \frac{\Delta\hat{t}}{\hat{t}_0} = \frac{\hat{S}(\tau,\mathscr{E})-\hat{S}_0}{\hat{S}_0}
# $$ {#eq-relative-transmission}
#
# Here, $\hat{S}$ is the discrete Fourier transform of a waveform, $\tau$ is the pump-probe delay, and $\mathscr{E}$ the relative pump field strength (unity corresponds to a peak electric field of about 200 kV/cm in the time domain). The dependence on frequency is omitted in the notation. $\hat{S}_0$ represents the negative-delay ($\tau < 0$) waveform, before the pump arrives, corresponding to the equilibrium signal.

# %%
trel_meas = SUSCEPTIBILITY_DATAFILES["trel_meas"].load()
display(trel_meas)

# %%
# | label: fig-trel-meas
# | fig-cap: Relative transmission amplitude for measured spectra.

delay_norm = SymLogNorm(1, 1, *trel_meas.extrema("pp_delay"))
g = sns.FacetGrid(
    trel_meas.filter(col("freq").is_between(0.3, 2.2))
    .rename({"t.reldiff.real": "real", "t.reldiff.imag": "imag"})
    .unpivot(
        index=["freq", "field_strength", "pp_delay"],
        on=["real", "imag"],
        variable_name="measurement",
    ),
    col="measurement",
    row="field_strength",
    hue="pp_delay",
    height=0.85,
    aspect=4,
    despine=False,
    sharex=True,
    sharey=True,
    palette="Spectral",
    gridspec_kws={"hspace": 0, "wspace": 0},
    margin_titles=True,
)
g.map_dataframe(
    sns.lineplot,
    x="freq",
    y="value",
    errorbar=None,
    hue_norm=delay_norm,
)


def label(col_name):
    comp = col_name.split(".")[-1]
    return comp


colorbar = g.fig.colorbar(
    plt.cm.ScalarMappable(norm=delay_norm, cmap="Spectral"),
    ax=g.axes,
    label=r"$\tau$ (ps)",
    aspect=20,
    shrink=0.5,
    location="right",
    pad=0.1,
)
g.set_titles(
    row_template=r"$\mathscr{{E}} = {row_name}$",
    col_template="{col_name}" r"$(\hat{{t}}_r)$",
)
g.set(ylabel=r"$t_r$", xlabel=r"$f$ (THz)")
plt.show()

# %% [markdown]
# ## Calculations
#
# The calculated spectra are relativized with respect to the equilibrium spectrum, $\hat{S}_0$, which is the spectrum of the sample at the conductivity of the sample at the experimental temperature, determined in a separate experiment. This coincides with the negative-delay spectrum of the experiment, so that the reference spectra are the same for both measured and calculated relative transmission amplitudes.

# %%
trel_calc = SUSCEPTIBILITY_DATAFILES["trel_calc"].load()
display(trel_calc)

# %% [markdown]
# ## Extracting the resonator-gap conductivity $\sigma$
#
# We map the experimental parameter space ($\tau$, $\mathscr{E}$) to the compuational one ($\sigma$) by minimizing the error function in @eq-error.
#
# $$
#     \epsilon(\tau,\mathscr{E};\sigma,\sigma_0) = \sum_{i}\left\lvert w(f_i)\left(\hat{t}_{r, \mathrm{meas}}(f_i; \tau, \mathscr{E}) - \hat{t}_{r, \mathrm{calc}}(f_i; {\sigma},\sigma_0)\right)\right\rvert^2
# $$ {#eq-error}
#
# Here $w$ is a weight that peaks around the resonance frequency, and tapers off at higher frequencies. In @fig-error we show the logarithm of $\epsilon$ for each combination of parameters, and we can identify a clear minimum for all combinations.
#

# %%
# | label: fig-error
# | fig-cap: The residual sum of squares (RSS) for all pump-probe time-delays ($\tau$) and relative field strengths ($\mathscr{E}$).

error = (
    SUSCEPTIBILITY_DATAFILES["error"]
    .load()
    .with_columns(col("error").log().alias("log-error"))
)
g = sns.relplot(
    error,
    x="pp_delay",
    y="cond_gap",
    hue="log-error",
    palette="Spectral",
    col="field_strength",
    facet_kws={"despine": 0},
    height=1.7,
    col_wrap=2,
    ec=None,
)
g.set(yscale="log")
g.figure.set_size_inches(6.8, 6.8)
plt.show()

# %% [markdown]
# Selecting the least-error values for $\sigma$, we can plot the conductivity as a function of the pump-probe delay, shown in @fig-sigma-time.

# %%
solution = SUSCEPTIBILITY_DATAFILES["solution"].load()
mapping = solution.fetch(
    col("cond_gap", "cond_film", "pp_delay", "field_strength")
).unique()
cond_base = mapping["cond_film"].unique().item()
display(mapping)

# %%
# | label: fig-sigma-time
# | fig-cap: The time evolution of the resonator gap conductivity $\sigma$, for a set of relative field strengths $\mathscr{E}$.

fig, ax = plt.subplots(figsize=(3.4, 2.6))
field_norm = Normalize(*mapping["field_strength"].sort()[[0, -1]])

g = sns.lineplot(
    mapping.with_columns(col("pp_delay") / 1e3),
    x="pp_delay",
    y="cond_gap",
    hue="field_strength",
    hue_norm=field_norm,
    palette="flare",
    marker="o",
    legend=False,
)
g.set(yscale="log", xlabel=r"$\tau$ (ns)", ylabel=r"$\sigma$ (S/m)", ylim=(3e3, 1e5))
g.axhline(cond_base, color="k", linestyle="--", lw=0.5)
g.annotate(
    r"Equilibrium", xy=(0, cond_base), xytext=(0, cond_base * 0.95), ha="left", va="top"
)
g.figure.colorbar(
    plt.cm.ScalarMappable(norm=field_norm, cmap="flare"),
    ax=g.axes,
    label=r"$\mathscr{E}$",
    location="right",
    pad=0.05,
)
plt.show()

# %% [markdown]
# It is instructive to consider the dependence on the relative pump field strength $\mathscr{E}$ at various pump-probe delays $\tau$. This is shown in @fig-sigma-field.

# %%
# | label: fig-sigma-field
# | fig-cap: Field-strength dependence, at various pump-probe time delays $\tau$. Solid lines are fits to the exponential function, in @eq-exponential.

line = lm.models.LinearModel()
params = line.make_params(intercept=np.log(cond_base), slope=10)
params["intercept"].set(vary=False)

fits = {}
frames = []
for (pp_delay,), df in mapping.with_columns(
    col("cond_gap").log().alias("logcond")
).group_by(["pp_delay"]):
    fits[pp_delay] = line.fit(df["logcond"], params, x=df["field_strength"])
    df = df.with_columns(
        pl.lit(fits[pp_delay].eval(x=df["field_strength"])).exp().alias("cond_fit")
    )
    frames.append(df)

delay_norm = SymLogNorm(1, 1, *mapping["pp_delay"].sort()[[0, -1]])
g = sns.relplot(
    pl.concat(frames),
    x="field_strength",
    y="cond_fit",
    hue="pp_delay",
    hue_norm=delay_norm,
    height=3.4,
    legend=False,
    palette="Spectral",
    kind="line",
)
g.map(
    sns.scatterplot,
    data=pl.concat(frames),
    x="field_strength",
    y="cond_gap",
    hue="pp_delay",
    hue_norm=delay_norm,
    palette="Spectral",
)
g.set(xlabel=r"$\mathscr{E}$", ylabel=r"$\sigma$ (S/m)", xlim=(0, None), yscale="log")
g.figure.colorbar(
    ScalarMappable(norm=delay_norm, cmap="Spectral"), ax=g.axes, label=r"$\tau$ (ps)"
)
g.axes[0, 0].axhline(cond_base, color="k", linestyle="--")
g.axes[0, 0].set_box_aspect(1)
g.despine(top=False, right=False)
g.figure.set_size_inches(3.4, 2.9)
plt.show()

# %% [markdown]
# The relationship looks exponential, so in @fig-sigma-field we fit the data to the form in @eq-exponential, shown in solid lines.
# $$
#     \frac{\sigma(t)}{\sigma_0} = {\mathrm{e}}^{\alpha\mathscr{E}(t = 0)}
# $$ {#eq-exponential}
#
# After fitting, we can then have a look at the time dependence, $\alpha(\tau)$, shown in @fig-slopes.

# %%
# | label: fig-slopes
# | fig-cap: The exponential factor $\alpha$ from @eq-exponential, as a function of the pump-probe delay $\tau$.

slopes = pl.DataFrame(
    {
        "pp_delay": list(fits.keys()),
        "slope": [fit.params["slope"].value for fit in fits.values()],
    }
)
g = sns.lineplot(
    slopes.with_columns(col("pp_delay") * 1e-3), x="pp_delay", y="slope", marker="o"
)
g.set(
    xlabel=r"$\tau$ (ns)",
    ylabel=r"$\alpha = \partial\log(\sigma) / \partial{\mathscr{E}}$",
)
g.figure.set_size_inches(3.4, 3.4)
plt.show()


# %% [markdown]
# ## Timescales
#
# Let us consider the decay dynamics of the conductivity, and a generic decay model of the form in @eq-decaymodel. The decay model is a sum of exponentials, with a step function $\Theta(t-t_0)$ that accounts for the fact that the conductivity is zero before the pump pulse arrives. The parameters are the initial conductivity $\sigma_0$, the time $t_0$ at which the pump pulse arrives, and the decay times $\tau_i$ and amplitudes $a_i$. We include also a constant term $a_0$, which models an “infinite” lifetime component.
#
# $$
#     \sigma = \sigma_0 + \Theta(t-t_0)\left(a_0 + \sum_{i=1}^N a_i \mathrm{e}^{-t/\tau_i}\right)
# $$ {#eq-decaymodel}


# %%
def decay_model(t, t0, a0, a1, a2, tau1, tau2, cond0):
    t = t - t0
    step = np.heaviside(t, 0)
    return step * (a0 + a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)) + cond0


def decay_model_zero(t, t0, a0, a1, a2, tau1, tau2, cond0):
    step = np.heaviside(t, 0)
    t = t - t0
    return step * (a0 + a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)) + cond0


model = lm.Model(decay_model)
cond0 = mapping["cond_film"].unique().item()
params = model.make_params(
    t0=0,
    a0=1e3,
    a1=1e4,
    a2=1e3,
    tau1=1e2,
    tau2=1e3,
    cond0=cond0,
)
params["tau1"].set(min=0)
params["tau2"].set(min=0)


def fit_model(mapping):
    results = []
    for (fs,), group in mapping.group_by(["field_strength"]):
        result = model.fit(group["cond_gap"], params, t=group["pp_delay"])
        results.append(result.best_values | {"field_strength": fs})
    return pl.DataFrame(results)


def plot_fits(results, xlim=None):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    field_norm = Normalize(*mapping["field_strength"].sort()[[0, -1]])

    g = sns.lineplot(
        mapping.with_columns(col("pp_delay") / 1e3),
        x="pp_delay",
        y="cond_gap",
        hue="field_strength",
        hue_norm=field_norm,
        palette="flare",
        marker="o",
        linewidth=0.5,
        legend=False,
    )
    xlim = xlim or (-0.05, 0.5)
    g.set(yscale="log", xlabel=r"$\tau$ (ns)", ylabel=r"$\sigma$ (S/m)", xlim=xlim)
    g.figure.colorbar(
        plt.cm.ScalarMappable(norm=field_norm, cmap="flare"),
        ax=g.axes,
        label=r"$\mathscr{E}$",
        location="right",
        pad=0.02,
    )

    time = np.linspace(-50, 5_000, 10_000)
    for x in results.iter_rows():
        g.axes.plot(
            time / 1e3, decay_model_zero(time, *x[:-1]), color="k", lw=0.5, zorder=-1
        )
    g.axes.set_box_aspect(1)
    return g


variable_mapping = {
    "t0": r"$t_0$ (ps)",
    "a0": r"$a_0$ (S/m)",
    "a1": r"$a_1$ (S/m)",
    "a2": r"$a_2$ (S/m)",
    "tau1": r"$\tau_1$ (ps)",
    "tau2": r"$\tau_2$ (ps)",
    "cond0": r"$\sigma_0$ (S/m)",
}


def plot_params(results):
    g = sns.relplot(
        results.unpivot(
            index=["field_strength"],
            on=results.columns[:-1],
            variable_name="parameter",
        ).with_columns(col("parameter").replace(variable_mapping)),
        x="field_strength",
        y="value",
        col="parameter",
        col_wrap=2,
        legend=False,
        height=1.7,
        facet_kws={"sharey": False, "despine": False, "xlim": (0, None)},
        kind="scatter",
        s=20,
        color="black",
    )
    g.set_titles("{col_name}")
    # for ax, p in zip(g.axes.flat, results.columns[:-1]):
    #     ax.ticklabel_format(axis="y", useOffset=False, scilimits=(-1, 1))
    #     ax.set(xlabel=r"$\mathscr{E}$", ylabel=variable_mapping[p])
    return g


# %% [markdown]
# We first consider a single decay time ($N = 1$), no infinite-lifetime component ($a_0 = 0$), and $t_0 = 0$. The result is shown in @fig-sigma-uniexponential-0.

# %%
# | label: fig-sigma-uniexponential-0
# | fig-cap: Fit to @eq-decaymodel with single decay time ($N = 1$), no infinite-lifetime component ($a_0 = 0$), and $t_0 = 0$.
# | fig-subcap:
# |   - "Time dynamics."
# |   - "Best-fit parameters."
# | layout-ncol: 2
# | layout-valign: center

params["t0"].set(value=0, vary=False)
params["a0"].set(value=0, vary=False)
params["tau1"].set(min=0)
params["tau2"].set(min=0, vary=False)
params["a2"].set(value=0, vary=False)
params["cond0"].set(vary=False)

results = fit_model(mapping.filter(col("pp_delay") >= params["t0"].value))

g_fits = plot_fits(results)
g_params = plot_params(
    results.select([p for p in params if params[p].vary] + ["field_strength"])
)
g_params.figure.set_size_inches(6.8, 3.4)
plt.show()

# %% [markdown]
# This is not a good fit, not least because there is a pump pulse echo at 20 ps, where the conductivity is raised further. We consider now $t_0 = 20$, measuring the decay time from the last pump pulse echo, where the conductivity is at its highest. We also include the infinite-lifetime component as a fit parameter. The result is shown in @fig-sigma-uniexponential-20.

# %%
# | label: fig-sigma-uniexponential-20
# | fig-cap: Fit to @eq-decaymodel with single decay time ($N = 1$), a free infinite-lifetime component ($a_0 = 0$), and $t_0 = 20$.
# | fig-subcap:
# |   - "Time dynamics."
# |   - "Best-fit parameters."
# | layout-ncol: 2
# | layout-valign: center

params["t0"].set(value=20, vary=False)
params["tau2"].set(vary=False)
params["a0"].set(value=0, vary=True)
params["a2"].set(value=0, vary=False)
params["cond0"].set(vary=False)

results = fit_model(mapping.filter(col("pp_delay") >= params["t0"].value))

g_fits = plot_fits(results)
g_params = plot_params(
    results.select([p for p in params if params[p].vary] + ["field_strength"])
)
g_params.figure.set_size_inches(6.8, 5.1)
plt.show()

# %% [markdown]
# Finally, we include a second decay time, and remove the infinite-lifetime component. This is shown in @fig-sigma-biexponential-finite.

# %%
# | label: fig-sigma-biexponential-finite
# | fig-cap: Fit to @eq-decaymodel with two decay times ($N = 2$), no infinite-lifetime component ($a_0 = 0$), and $t_0 = 20$.
# | fig-subcap:
# |   - "Time dynamics."
# |   - "Best-fit parameters."
# | layout-ncol: 2
# | layout-valign: center

params["t0"].set(value=19, vary=False)
params["tau1"].set(value=20)
params["tau2"].set(value=1e5, vary=True)
params["a0"].set(value=0, vary=False)
params["a1"].set(value=1e4, vary=True)
params["a2"].set(value=1e4, vary=True)
params["cond0"].set(vary=False)

results = fit_model(mapping.filter(col("pp_delay") >= params["t0"].value))

g_fits = plot_fits(results, xlim=(-0.05, 1.2))
g_plots = plot_params(
    results.select([p for p in params if params[p].vary] + ["field_strength"])
)
g_plots.figure.set_size_inches(6.8, 5.1)
plt.show()

# %% [markdown]
# This seems to match the data quite well. It looks like a single exponential decay is not sufficient to describe the dynamics, and a biexponential model is required. We take a closer look at the long-time behavior in @fig-sigma-biexponential-finite-long.

# %%
# | label: fig-sigma-biexponential-finite-long
# | fig-cap: A look at the fit to @eq-decaymodel with two decay times ($N = 2$), no infinite-lifetime component ($a_0 = 0$), and $t_0 = 20$, over long timescales.

plot_fits(results, xlim=(-0.2, 2))

# %% [markdown]
# There is seeming inconsistency in the long-term behavior as a function of field-strength. We can try to include an infinite-lifetime component, as shown in @fig-sigma-biexponential-infinite.

# %%
# | label: fig-sigma-biexponential-infinite
# | fig-cap: Fit to @eq-decaymodel with two decay times ($N = 2$), a free infinite-lifetime component ($a_0 = 0$), and $t_0 = 20$.
# | fig-subcap:
# |   - "Time dynamics."
# |   - "Best-fit parameters."
# | layout-ncol: 2
# | layout-valign: center

params["t0"].set(value=19, vary=False)
params["tau1"].set(value=20)
params["tau2"].set(value=200, vary=True)
params["a0"].set(value=0, vary=True)
params["a1"].set(value=1e4, vary=True)
params["a2"].set(value=1e3, vary=True)
params["cond0"].set(vary=False)

results = fit_model(mapping.filter(col("pp_delay") >= params["t0"].value))

g_fits = plot_fits(results)
g_params = plot_params(
    results.select([p for p in params if params[p].vary] + ["field_strength"])
)
g_params.figure.set_size_inches(6.8, 6.8)
plt.show()

# %% [markdown]
# This seems to match the data very well, and the fit parameters show some trends, with the exception of the lowest-field trace, which does not fit well to any of the above-mentioned models.
#
# ::: {.callout-note icon=false}
# ## Takeaway
#
# The conductivity dynamics following the second pump pulse is best described by a *biexponential decay* model, with the inclusion of an *infinite-lifetime component*.
#
# - There is a fast decay process of 10--35 ps, which slows down with increasing field strength. Its amplitude is (sub-)linear in field strength.
# - There is a slow decay process of 200--300 ps, with a quadratic field strength dependence.
# - There is a remnant conductivity that persists for at least tens of nanoseconds, which increases linearly with field strength.
#
# :::
