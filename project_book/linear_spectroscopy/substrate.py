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
# title: Substrate
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

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from polars import col 
from polars_complex import ccol
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.linear_spectroscopy.lsat import (
    DATAFILES as LSAT_DATAFILES,
)
from driven_nickelate.linear_spectroscopy.nosample import (
    DATAFILES as NOSAMPLE_DATAFILES,
)
from driven_nickelate.tools import pl_fft
from polars_dataset import Dataset
from mplstylize import mpl_tools

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(10)

# %%
#| label: fig-substrate-signals
#| fig-cap: Time-domain signals of the LSAT substrate and the no-sample reference (gray).

waveforms_lsat = (
    LSAT_DATAFILES["waveforms"]
    .load()
    .filter(
        (col("detector") == "GaP")
        & col("date").is_in(["2023-01-30", "2023-01-31"])
    )
    .with_columns((6.67 * (11.4 - col("delay"))).alias("time"))
    .set(index="time")
    .drop("delay")
    .sort_columns()
    .sort("time")
)
waveforms_nosample = (
    NOSAMPLE_DATAFILES["waveforms"]
    .load()
    .filter(
        (col("detector") == "GaP")
        & (col("source") == "ZnTe")
        & (col("date") == "2023-01-28")
    )
    .with_columns((6.67 * (11.4 - col("delay"))).alias("time"))
    .set(index="time")
    .drop("delay")
    .sort_columns()
    .pivot(
        index=["time", "temperature", "filename", "date"],
        values="value",
        columns="variable",
    )
    .sort("time")
)
waveforms_nosample = Dataset(
    waveforms_nosample, index="time", id_vars=["temperature", "filename", "date"]
)

fig = plt.figure(figsize=(3.4, 3.4))

sns.lineplot(
    waveforms_lsat,
    x="time",
    y="X",
    hue="temperature",
    palette="flare",
)
g = sns.lineplot(
    waveforms_nosample, x="time", y="X.avg", legend=False, color="gray", estimator=None
)
g.legend(title="$T$ (K)")
plt.show()

# %%
#| label: fig-substrate-spectra-interpolated
#| fig-cap: Interpolated waveforms of the LSAT substrate and the no-sample reference (gray), with windowing.


def time_window(t, t1, t2):
    left = np.min([np.ones(len(t)), np.exp((t - t1) / 0.5)], axis=0)
    right = np.min([np.ones(len(t)), np.exp(-(t - t2) / 2)], axis=0)
    return left + right - 1


time = pl.Series("time", np.arange(-10, 30, 6e-3))

waveforms_nosample = waveforms_nosample.regrid(
    time, fill_value=0
)  # Resample on longer, denser time grid
waveforms_nosample = (
    waveforms_nosample.group_by("time")
    .agg(
        col("temperature").mean(),
        pl.lit(
            "*_LSAT_eSRR REF (no sample, thermalized at room temp)_292.92-293.10_.txt"
        ).alias("filename"),
        col("date").first(),
        col("X.avg").mean().alias("X"),
        col("X.sem").mean(),
    )  # Average the two measurements
    .with_columns(
        col("X") * col("time").map_batches(lambda t: time_window(t, -5, 15)),
    )  # Apply time window
)
waveforms_lsat = waveforms_lsat.regrid(
    time, fill_value=0
)  # Resample on longer, denser time grid

waveforms_lsat = waveforms_lsat.with_columns(
    col("X") * col("time").map_batches(lambda t: time_window(t, 0, 20))
)  # Apply time window

fig = plt.figure(figsize=(3.4, 3.4))
sns.lineplot(
    waveforms_nosample, x="time", y="X", legend=False, color="gray", estimator=None
)
g = sns.lineplot(
    waveforms_lsat, x="time", y="X", hue="temperature", palette="flare", estimator=None
)
g.legend(title="$T$ (K)")

plt.show()

# %%
#| label: fig-substrate-spectra
#| fig-cap: Substrate spectra, defined as the ratio of the Fourier transforms of the waveforms transmitted through the substrate, to that of the no-sample reference. **a**. Magnitude. **b**. Phase.

spectra = pl_fft(
    waveforms_lsat.join(
        waveforms_nosample.rename({"X": "X_ref", "X.sem": "X_ref.sem"}), on="time"
    ).select("temperature", "time", "X", "X.std", "X_ref", "X_ref.sem"),
    xname="time",
    id_vars=["temperature"],
).filter(col("freq").is_between(0.1, 3))

spectra = (
    spectra.select(
        col("temperature"),
        col("freq"),
        col("X.real", "X.imag").complex.struct("X[c]"),
        col("X_ref.real", "X_ref.imag").complex.struct("X_ref[c]"),
    )
    .with_columns(
        (ccol("X[c]") / ccol("X_ref[c]")).alias("t[c]"),
    )
    .with_columns(
        ccol("t[c]").modulus().alias("t.mag"),
        (ccol("t[c]").phase_unwrapped().over("temperature") - 2 * np.pi).alias("t.pha"),
    )
)

fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4))
g = sns.lineplot(
    spectra.select("temperature", "freq", "t.mag"),
    x="freq",
    y="t.mag",
    hue="temperature",
    palette="flare",
    ax=ax[0],
)
g.legend(title="$T$ (K)")

g = sns.lineplot(
    spectra.select("temperature", "freq", "t.pha"),
    x="freq",
    y="t.pha",
    hue="temperature",
    palette="flare",
    ax=ax[1],
)
g.legend(title="$T$ (K)")

mpl_tools.enumerate_axes(ax)

plt.show()

# %%
#| label: fig-substrate-fit
#| fig-cap: Fit of the substrate spectra to a Drude-Lorentz model, using a single resonance. **a**. Real part of the refractive index. **b**. Imaginary part of the refractive index.

sub_thickness = 0.5  # [mm]


def subtract_offset(group):
    group_limited = group.filter(col("freq").is_between(0.2, 1.0))
    c = np.polyfit(group_limited["freq"], group_limited["t.pha"], 2)
    return group.with_columns((col("t.pha") - c[-1]).alias("t.pha"))


refractive_index = (
    1 - 0.29979 / (2 * np.pi * col("freq") * sub_thickness) * col("t.pha")
).alias("n")
extinction_coeff = (
    -0.29979
    / (2 * np.pi * col("freq") * sub_thickness)
    * np.log((col("n") + 1) ** 2 / (4 * col("n")) * col("t.mag"))
).alias("κ")

spectra = (
    spectra.group_by("temperature")
    .map_groups(subtract_offset)
    .select(
        col("freq"),
        col("temperature"),
        refractive_index,
        col("t.pha"),
        col("t.mag"),
    )
    .with_columns(extinction_coeff)
)

fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4))
sns.lineplot(spectra, x="freq", y="n", hue="temperature", ax=ax[0]).set(yscale="log")
sns.lineplot(spectra, x="freq", y="κ", hue="temperature", ax=ax[1]).set(yscale="log")

# ---
freq0 = 4.0


def drude_lorentz(freq, eps_inf, strength, gamma, freq0):
    return eps_inf + strength * freq0**2 / (freq0**2 - freq**2 - 1j * freq * gamma)


def objective(x, group, n_weights, κ_weights):
    eps_inf, strength, gamma = x
    eps = drude_lorentz(group["freq"].to_numpy(), eps_inf, strength, gamma, freq0)
    κ = np.imag(np.sqrt(eps))
    n = np.real(np.sqrt(eps))
    n_error = (n - group["n"].to_numpy()) * n_weights
    κ_error = (κ - group["κ"].to_numpy()) * κ_weights
    return np.sum(n_error**2 + κ_error**2)


from scipy.optimize import minimize

results = {}
for (temperature,), group in spectra.sort("temperature").group_by(
    ["temperature"], maintain_order=True
):
    group = group.filter(col("freq").is_between(0.1, 2.0))

    n_weights = (group["freq"].is_between(0.2, 2.2)).cast(pl.Float32).to_numpy()
    κ_weights = (group["freq"].is_between(0.7, 2.2)).cast(pl.Float32).to_numpy()

    result = minimize(
        objective,
        x0=[4, 20.0, 0.1],
        args=(group, n_weights, κ_weights),
        method="Nelder-Mead",
    )
    results[temperature] = result

# fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4))
freq = np.arange(0, 5, 0.01)
cmap = sns.color_palette("flare", as_cmap=True)
for temperature, result in results.items():
    eps_inf, strength, gamma = result.x
    eps = drude_lorentz(freq, eps_inf, strength, gamma, freq0)
    n = np.sqrt(eps)
    ax[0].plot(freq, n.real, c=cmap(temperature / 220))
    ax[1].plot(freq, n.imag, c=cmap(temperature / 220))

ax[0].set(yscale="linear", ylim=(4, 6))
ax[1].set(yscale="linear", ylim=(0, 0.5))

# %%
#| label: fig-substrate-fit-wide
#| fig-cap: A wide view of the fit result.

indices = {"eps_inf": 0, "strength": 1}
results_dict = {
    key: np.mean([result.x[value] for result in results.values()])
    for key, value in indices.items()
}
eps_inf, strength = [results_dict[key] for key in ["eps_inf", "strength"]]

gamma = np.array([result.x[2] for result in results.values()])

cmap = sns.color_palette("flare", as_cmap=True)
fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4))
freq = np.arange(0, 10, 0.01)
for i, g in enumerate(gamma):
    eps = drude_lorentz(freq, eps_inf, strength, g, freq0)
    ax[0].plot(freq, np.sqrt(eps).real, c=cmap(i / len(gamma)))
    ax[1].plot(freq, np.sqrt(eps).imag, c=cmap(i / len(gamma)))

ax[0].set(xlim=(0.1, 10.2))
ax[1].set(xlim=(0.1, 10.2))

eps = drude_lorentz(freq, eps_inf, strength, gamma[0], freq0)
pl.DataFrame({"freq": freq, "eps_real": eps.real, "eps_imag": eps.imag}).write_csv(
    PROJECT_PATHS.processed_data / "substrate" / "lsat_fit_experiment.csv"
)
plt.show()
