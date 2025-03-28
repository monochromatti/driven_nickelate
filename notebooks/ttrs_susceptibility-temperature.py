# %% [markdown]
# # Temperature

# %%
import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from IPython.display import display
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from mplstylize import mpl_tools
from polars import col as c

from driven_nickelate.conductivity_mapping.susceptibility_temperature import (
    DATAFILES as SUSCEPTIBILITY_DATAFILES,
)
from driven_nickelate.conductivity_mapping.temperature import cond_from_temperatures

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(10)

# %% [markdown]
# ## Experiment
#
# Waveforms transmitted through the sample is recorded for a set of temperatures and pump field strengths, with the aim of assessing the “optical susceptibility”, in the sense of how the material response to the pump field strength. In this experiment, the probe THz waveforms arrives *20 ps* after the THz pump pulse. The dataset is structured as follows:

# %%
waveforms = SUSCEPTIBILITY_DATAFILES["waveforms_meas"].load()
display(waveforms)

# %% [markdown]
# The raw signal `X` is the absolute waveforms, and `dX` is calculated by subtracting `X_eq`, which is defined as the waveform corresponding to zero pump field strength.

# %%
# | label: fig-waveforms
# | fig-cap: Summary of raw differential waveforms, recorded after transmission through the sample for various values of the THz pump field strength and sample temperature. In black, the equilibrium absolute waveform.

colors = sns.color_palette("Spectral", n_colors=waveforms["field_strength"].n_unique())
g = sns.FacetGrid(
    waveforms.with_columns(
        c("dX") * 1e3 + c("dX").cum_count().over("time", "temperature"), c("X_eq") * 1e3
    ),
    col="temperature",
    hue="field_strength",
    despine=False,
    palette=colors,
    height=1.7,
    aspect=0.5,
    gridspec_kws=dict(wspace=0),
)
vmin, vmax = waveforms.extrema("field_strength")
field_norm = Normalize(vmin, vmax)
g.map(sns.lineplot, "time", "dX", lw=0.8, hue_norm=field_norm)
g.set(xlim=(-2, 12), xlabel=r"$t$ (ps)", ylabel=r"dX (mV)")
g.set_titles(r"{col_name:.0f} K")
g.figure.colorbar(
    ScalarMappable(norm=field_norm, cmap=ListedColormap(colors.as_hex())),
    ax=g.axes,
    label=r"$\mathscr{E}_r$",
    pad=0.01,
)
plt.show()

# %% [markdown]
# The time-domain data is transformed to the frequency domain. The measured signal is the differential signal (pump on minus pump off), and one trace of the absolute waveform absent pump. We therefore look at the relative transmission amplitude in the frequency domain, defined as
#
# $$
# \hat{t}_r(\tau) \equiv \frac{\Delta\hat{t}}{\hat{t}_0} = \frac{\Delta\hat{S}(\tau)}{\hat{S}_0}
# $$ {#eq-relative-transmission}
#
# Here, $\Delta\hat{S}$ is the discrete Fourier transform of “dX”, and $\hat{S}_0$ is the Fourier transform of the `X_eq`, the equilibrium waveform. $T$ is the sample temperature, and $\mathscr{E}_r$ the pump field strength (normalized, with unity corresponding to about 200 kV/cm).
#
# ## Calculations
#
# The transmission amplitude of the sample can be calculated theoretically, with the simplifying assumption that any field-induced variation in conductivity occurs exclusively in the capacitor-like gap of the split-ring resonator.
#
# Aiming to map the theoretical resonator-gap conductivity $\sigma$ to the experimental parameters $T$ and $\mathscr{E}_r$, we need to choose the appropriate reference spectrum when calculating the *relative* transmission amplitude,
#
# $$
# \hat{t}_r(\sigma_0 = \sigma_0(T)) = \frac{\Delta\hat{t}}{\hat{t}_0} = \frac{\hat{t}(\sigma, \sigma_0) - \hat{t}(\sigma = \sigma_0)}{\hat{t}(\sigma = \sigma_0)}
# $$
#
#
# To do this, we need to make a choice for $\sigma_0$, which is different for each temperature, and possibly for each field-strength. If we assume it is independent of field strength (*viz*. the field only affects the material in the resonator gap), we can use the analysis of the linear spectroscopy data, where a correspondence between temperature and film conductivity was obtained, as shown in @fig-temperature-mapping.

# %%
temperatures = waveforms.coord("temperature")
temperatures_cont = np.linspace(0, 200, 100)

fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.scatter(temperatures, cond_from_temperatures(temperatures))
ax.plot(temperatures_cont, cond_from_temperatures(temperatures_cont))
ax.ticklabel_format(axis="x", scilimits=(0, 3))
ax.set(xlabel=r"$T$ (K)", ylabel=r"$\sigma$ (S/m)", yscale="log")
plt.show()

# %% [markdown]
# Then we obtain $\sigma(T,\mathscr{E}_r)$ by solving @eq-minimize-fixed, which we call the *fixed-$\sigma_0$ minimization*.
#
# $$
#     \min_{\sigma} \epsilon(T,\mathscr{E}_r ; \sigma, \sigma_0)\Big\vert_{\sigma_0 = \sigma_0(T)}
# $$ {#eq-minimize-fixed}
#
# where
#
# $$
#     \epsilon(T,\mathscr{E}_r;\sigma,\sigma_0) = \sum_{i}\left\lvert w(f_i)\left(\hat{t}_{r, \mathrm{meas}}(f_i; T, \mathscr{E}_r) - \hat{t}_{r, \mathrm{calc}}(f_i; {\sigma},\sigma_0)\right)\right\rvert^2
# $$ {#eq-error}
#
# the weighted residual sum of squares, with $w(f)$ a weighting function that peaks around the resonance frequency.
#
# On the other hand, we can also solve for both the local resonator gap conductivity $\sigma$ and the film conductivity $\sigma_0$ in a bilevel minimization scheme, determining both $\sigma$ and $\sigma_0$ from the same experiment. This leaves two choices: either we assume, as we did in the *fixed-$\sigma_0$ minimization*, that $\sigma_0 = \sigma_0(T)$, and does not depend on field strength $\mathscr{E}_r$, or we allow $\sigma_0$ to vary with $\mathscr{E}_r$ as well. We call these the *rigid-$\sigma_0$ minimization* and the *free-$\sigma_0$ minimization*, respectively.
#
# The *rigid-$\sigma_0$ minimization* is given by @eq-minimize-rigid,
# $$
#     \min_{\sigma_0} \Big\langle \epsilon(T,\mathscr{E}_r ; \sigma,\sigma_0)\Big\vert_{\sigma = \sigma^\ast} \Big\rangle_{\mathscr{E}_r}
#     \quad\mathrm{where}\quad
#     \sigma^\ast(T,\mathscr{E}_r;\sigma_0) = \arg\min_{\sigma} \epsilon(T,\mathscr{E}_r ; \sigma, \sigma_0)
# $$ {#eq-minimize-rigid}
#
# where $\langle\cdot\rangle_\alpha$ denotes the mean over variable $\alpha$. The *free-$\sigma_0$ minimization* is given by @eq-minimize-free,
#
# $$
#     \min_{\sigma_0} \epsilon(T,\mathscr{E}_r ; \sigma,\sigma_0)\Big\vert_{\sigma = \sigma^\ast}
#     \quad\mathrm{where}\quad
#     \sigma^\ast(T,\mathscr{E}_r;\sigma_0) = \arg\min_{\sigma} \epsilon(T,\mathscr{E}_r ; \sigma, \sigma_0)
# $$ {#eq-minimize-free}
#
# where the inner error is not averaged over $\mathscr{E}_r$. In @fig-solution-comparison, we compare the results of these three minimzation schemes.

# %%
# | label: fig-solution-comparison
# | fig-cap: The resonator-gap conductivity as a function of field strength $\mathscr{E}_r$ and temperature $T$, shown for each of the three minimization schemes.

suffixes = ["fixed", "rigid", "free"]
fig = plt.figure(figsize=(6.8, 6.8 / len(suffixes)))
ax = fig.subplot_mosaic([[suffix for suffix in suffixes]], sharey=True)
temperature_norm = Normalize(5, 160)
for suffix in suffixes:
    result = SUSCEPTIBILITY_DATAFILES[f"solution_{suffix}"].load()
    g = sns.lineplot(
        result,
        x="field_strength",
        y="cond_gap",
        hue_norm=temperature_norm,
        hue="temperature",
        palette="Reds",
        marker="o",
        ax=ax[suffix],
        legend=False,
    )
    g.set(yscale="log", xlabel=r"$\mathscr{E}_r$ (norm.)", ylabel=r"$\sigma$ (S/m)")
    g.set_title(rf"{suffix}-$\sigma_0$")
    ax[suffix].set_box_aspect(1)
cbar = fig.colorbar(
    ScalarMappable(norm=temperature_norm, cmap="Reds"),
    ax=ax[suffixes[-1]],
    label=r"$T$ (K)",
    pad=0.01,
    shrink=0.9,
)
plt.show()

# %% [markdown]
# The solutions are quite consistent with each other, giving the same trends while displaying some differences in absolute values. Henceforth, we will use the *fixed-$\sigma_0$ minimization*, as this ensures consistency with the linear spectroscopy data and with other analyses, and takes advantage of the high-quality solution of the linear spectroscopy experiment. We can look at this data as a function of temperature, for fixed field strengths, or as a function of field-strength, for fixed temperatures. These views are shown in @fig-dual-view.

# %%
# | label: fig-dual-view
# | fig-cap: The temperature dependence of the resonator-gap conductivity $\sigma$, for a set of field strengths $\mathscr{E}_r$.
# | fig-subcap:
# |   - "As a function of temperature."
# |   - "As a function of field-strength."
# | layout-ncol: 2
# | layout-valign: bottom

result = SUSCEPTIBILITY_DATAFILES["solution_fixed"].load()
tempr_norm = Normalize(*result["temperature"].sort()[[0, -1]])
g = sns.lineplot(
    result,
    x="field_strength",
    y="cond_gap",
    hue="temperature",
    hue_norm=tempr_norm,
    palette="Reds",
    legend=False,
    marker="o",
)
g.set(yscale="log", xlabel=r"Field strength, $\mathscr{E}_r$", ylabel=r"$\sigma$ (S/m)")
g.figure.set_size_inches(3.4, 2.6)
cbar = g.figure.colorbar(
    ScalarMappable(norm=tempr_norm, cmap="Reds"),
    ax=g.axes,
    label=r"Temperature, $T$ (K)",
)
plt.show()

field_norm = Normalize(*result["field_strength"].sort()[[0, -1]])
g = sns.lineplot(
    result,
    x="temperature",
    y="cond_gap",
    hue="field_strength",
    hue_norm=field_norm,
    palette="Blues",
    legend=False,
    marker="o",
)
g.set(yscale="log", xlabel=r"Temperature, $T$ (K)", ylabel=r"$\sigma$ (S/m)")
g.figure.set_size_inches(3.4, 2.6)
cbar = g.figure.colorbar(
    ScalarMappable(norm=field_norm, cmap="Blues"),
    ax=g.axes,
    label=r"Field strength, $\mathscr{E}_r$ (norm.)",
)
plt.show()


# %% [markdown]
# ## Field-assisted tunneling
#
# The Landau-Dykhne expression for the transition rate is
# $$
#     \Gamma = \frac{1}{2\pi}\exp\left(-\pi\frac{\mathscr{E}_0}{\mathscr{E}_r}\right)
# $$
#
# If we take the conductivity to be proportional to this transition rate (which is a reasonable assumption, since the conductivity is proportional to the number of carriers), and we allow for a constant background conductivity $\sigma_{\mathrm{b}}$, we can write the conductivity as
#
# $$
#     \sigma = \sigma_{\mathrm{b}} + \sigma_{\mathrm{LD}}\exp\left(-\frac{\pi\mathscr{E}_0}{\mathscr{E}_r}\right)
# $$ {#eq-landau-dykhne}
#
# We can fit this model to the data, and evaluate the temperature dependence of the relative threshold field $\mathscr{E}_0$. The value of $\sigma_{\mathrm{LD}}$ can be determined from the linear spectroscopy data, so that it is not a free parameter.


# %%
def landau_dykhne(E, Eth, sig_ld, sig_b):
    return sig_ld * np.exp(-np.pi * Eth / (E + 1e-15)) + sig_b


ld_model = lm.Model(landau_dykhne)
ld_params = ld_model.make_params(Eth=0.2, sig_ld=1e3, sig_b=1e3)
ld_params["Eth"].set(min=0, max=1)

fit_results = {}

for i, ((temperature,), group) in enumerate(
    SUSCEPTIBILITY_DATAFILES["solution_fixed"]
    .load()
    .group_by(["temperature"], maintain_order=True)
):
    group = group.sort("field_strength")
    cond0 = cond_from_temperatures([temperature])[0]
    ld_params["sig_b"].set(value=cond0, vary=False)
    ld_fit = ld_model.fit(
        group["cond_gap"],
        E=group["field_strength"],
        params=ld_params,
    )
    fit_results[temperature] = ld_fit


fit_df = pl.DataFrame(
    {
        "temperature": list(fit_results.keys()),
        **{
            f"{param}.{stat}": [
                getattr(fit.params[param], stat) for fit in fit_results.values()
            ]
            for param in ["Eth", "sig_ld", "sig_b"]
            for stat in ["value", "stderr"]
        },
    }
)

# %%
# | label: fig-ld-fit
# | fig-cap: Landau-Dykhne fit to the resonator-gap conductivity $\sigma$, for a set of temperatures $T$.

fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))
cmap = plt.get_cmap("Reds")
norm = Normalize(*result["temperature"].sort()[[0, -1]])
sns.scatterplot(
    result,
    x="field_strength",
    y="cond_gap",
    hue="temperature",
    hue_norm=norm,
    palette=cmap,
    ax=ax,
    legend=False,
)
field = np.linspace(0, 1, 100)
for temperature, fit in fit_results.items():
    ax.plot(field, fit.eval(E=field), color=cmap(norm(temperature)), lw=0.8, zorder=-1)

ax.set(
    ylabel=r"$\sigma$ (S/m)",
    yscale="linear",
    xscale="linear",
    xlabel=r"$\mathscr{E}_r$ (norm.)",
)
cbar = fig.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    label=r"$T$ (K)",
)
plt.show()

# %% [markdown]
# The Landau-Dykhne model fits the data rather well, and yields the parameters $\mathscr{E}_0$, which is the (relative) field-strength threshold, and $\sigma_0$ This is shown in @fig-ld-params.

# %%
# | label: fig-ld-params
# | fig-cap: The field-strength threshold $\mathscr{E}_0$ as a function of temperature $T$.

fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.4), sharex=True)

ax[0].errorbar(
    *fit_df.select("temperature", c("sig_ld.value"), c("sig_ld.stderr")).get_columns(),
    marker="o",
    linestyle="none",
)
ax[0].set(ylabel=r"$\sigma_\mathrm{LD}$ (S/m)")
ax[1].errorbar(
    *fit_df.select("temperature", "Eth.value", "Eth.stderr").get_columns(),
    marker="o",
    linestyle="none",
)
ax[1].set(xlabel=r"$T$ (K)", ylabel=r"$\mathscr{E}_0$")

mpl_tools.breathe_axes(ax)

plt.show()

# %% [markdown]
# A positive correlation between temperature and $\mathscr{E}_0$ is surprising, and it should be considered whether this is physically realistic.
