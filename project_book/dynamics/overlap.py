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
# title: "Pump-probe overlap"
# ---

# %%
# | output: false

import re
from pathlib import Path

import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import mplstylize
import seaborn as sns
from driven_nickelate.conductivity_mapping.overlap import (
    DATAFILES as OVERLAP_DATAFILES,
)
from driven_nickelate.conductivity_mapping.temperature import (
    DATAFILES as TEMPERATURE_DATAFILES,
)
from driven_nickelate.simulations.gapfield import (
    DATAFILES as GAPFIELD_DATAFILES,
)
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.scripts.pump_probe.field_from_hwp import field_from_hwp
from driven_nickelate.tools import create_dataset
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from polars import col
from mplstylize import mpl_tools, colors
from scipy.interpolate import LinearNDInterpolator
from scipy.special import erf

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(10)

# %% [markdown]
# ## Experiment
#
# Waveforms transmitted through the sample is recorded for a set of pump-probe delays $\tau$ (the common delay line of the optical gate and THz probe, with respect to the THz pump), sampled densely around $\tau = 0$. The pump is modulated, so that the signal is the differential signal (pump on minus pump off). A reference waveform without pump is measured as well, so that we have both $\Delta E$ and $E$.
#
# We load the raw data from experiment, and pad with zeros to interpolate in the frequency domain. Note that these measurements were quite noisy, since a lot of time-traces had to be performed for combinations of the waveplate position (pump power) and pump-probe delay. The dataset is structured as follows:

# %%
waveforms = OVERLAP_DATAFILES["waveforms_meas"].load()
display(waveforms)

# %%
# | label: fig-waveforms
# | fig-cap: Summary of raw waveforms, recorded after transmission through the sample for various values of the THz pump field strength, and the pump-probe delay.

fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.set_box_aspect(1)

delay_norm = Normalize(*waveforms.extrema("pp_delay"))
sns.lineplot(
    waveforms.filter(col("time").is_between(-1, 10)).with_columns(col("dX") * 1e3),
    x="time",
    y="dX",
    hue="pp_delay",
    hue_norm=delay_norm,
    palette="Spectral",
    legend=False,
).set(xlabel=r"$t$ (ps)", ylabel=r"EOS (mV)")
sns.lineplot(
    waveforms.select("time", "X")
    .unique()
    .filter(col("time").is_between(-1, 10))
    .with_columns(col("X") * 1e3),
    x="time",
    y="X",
    color="black",
    linewidth=0.5,
    legend=False,
    zorder=-1,
)
fig.colorbar(
    plt.cm.ScalarMappable(norm=delay_norm, cmap="Spectral"),
    label=r"$\Delta\tau$ (ps)",
    aspect=30,
    ax=ax,
)
plt.show()

# %% [markdown]
# The time-domain data is transformed to the frequency domain. The measured signal is the differential signal (pump on minus pump off), and one trace of the absolute waveform absent pump. We therefore look at the relative transmission amplitude in the frequency domain, defined as
#
# $$
# \hat{t}_r(\tau) \equiv \frac{\Delta\hat{t}}{\hat{t}_0} = \frac{\Delta\hat{S}(\tau)}{\hat{S}_0}
# $$ {#eq-relative_transmission}
#
# Here, $\hat{S}$ is the discrete Fourier transform of a waveform, $\tau$ is the pump-probe delay, and $\mathcal{E}$ the pump field strength. The dependence on frequency is omitted in the notation. $\Delta\hat{S}$ is the Fourier transform of the differential waveform, and $\hat{S}_0$ is the transform of the full waveform absent pump.
#
# ## Calculations
#
# The transmission amplitude of the sample can be calculated theoretically, with the simplifying assumption that any field-induced variation in conductivity occurs exclusively in the capacitor-like gap of the split-ring resonator.
#
# The calculated data is aligned with the measured data, on the frequency axis (same frequency sampling), and upsampled along the conductivity axis through spline interpolation.
#
# Then, it is relativized with respect to the low-temperature conductivity of the film (and resonator gap), as determined from the linear spectroscopy temperature study of the sample. This coincides with the negative-delay spectrum of the experiment, so that the reference spectra are the same for both measured and calculated data.

# %%
trel_joint = OVERLAP_DATAFILES["trel_joint"].load()
display(trel_joint)

# %%
# | label: fig-trel-summary
# | fig-cap: "Relative transmission amplitude"
# | fig-subcap:
# |   - "Experimental data, with transmission relative to an unexcited sample at base temperature."
# |   - "Calculated data, with transmission relative to a sample with homogeneous conductivity, corresponding to base temperature."
# | layout: [ [1], [1] ]

meas_data = (
    trel_joint.select("freq", "pp_delay", "^t_meas.*$")
    .unique()
    .filter(col("pp_delay") > -5)
)
delay_norm = Normalize(*meas_data.extrema("pp_delay"))
g = sns.FacetGrid(
    meas_data.unpivot(index=["freq", "pp_delay"], value_name="t_meas"),
    col="variable",
    col_wrap=2,
    despine=False,
)
g.map(sns.lineplot, "freq", "t_meas", "pp_delay", hue_norm=delay_norm, palette="flare")
g.figure.colorbar(
    plt.cm.ScalarMappable(norm=delay_norm, cmap="flare"),
    label=r"$\Delta\tau$ (ps)",
    aspect=30,
    ax=g.axes,
)

cond_sample = trel_joint.coord("cond_gap").gather_every(50)
calc_data = (
    trel_joint.select("freq", "cond_gap", "^t_calc.*$")
    .unique()
    .filter(col("cond_gap").is_in(cond_sample))
)
cond_norm = LogNorm(cond_sample.min(), cond_sample.max())
g = sns.FacetGrid(
    calc_data.unpivot(index=["freq", "cond_gap"], value_name="t_calc"),
    col="variable",
    col_wrap=2,
    despine=False,
)
g.map(
    sns.lineplot, "freq", "t_calc", "cond_gap", hue_norm=cond_norm, palette="Spectral"
)
g.figure.colorbar(
    plt.cm.ScalarMappable(norm=cond_norm, cmap="Spectral"),
    label=r"$\sigma$ (S/m)",
    aspect=30,
    ax=g.axes,
)
plt.show()

# %% [markdown]
# ## Mapping
#
# Finally, we can map from the experimental parameter ($\tau$) to the compuational one ($\sigma$). The minimization norm is a residual sum of squares (RSS), but the minimization is done in a single step “grid search”, rather than iteratively. This is possible because the calculations are done on a conductivity grid a priori, and interpolation is reliable. The minimization is done as follows:
#
# 1. The residual sum of squares (RSS) is calculated by grouping by $\tau$ and $\sigma$, squaring and summing the residual over frequencies.
# 2. Grouping by $\tau$ only, the RSS is sorted in ascending order (together with $\sigma$) and the smallest value is picked.
#

# %%
# | label: fig-mapping
# | fig-cap: In black, the least-residual $\sigma$, and in color the residual sum of squares (RSS).
residuals = (
    OVERLAP_DATAFILES["error"]
    .load()
    .sort("pp_delay", "cond_gap")
    .filter(col("pp_delay") > -5)
)

interpolator = LinearNDInterpolator(
    residuals.select("pp_delay", "cond_gap"),
    residuals.select((col("error")).log().alias("log-error")).get_column("log-error"),
)

ci = np.geomspace(*residuals["cond_gap"].sort()[[0, -1]].to_list(), 100)
pi = np.linspace(*residuals["pp_delay"].sort()[[0, -1]].to_list(), 100)
pi, ci = np.meshgrid(pi, ci)
color_value = interpolator(pi, ci)

fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.set(yscale="log", xlabel=r"$\tau$ (ps)", ylabel=r"$\sigma$ (S/m)")
ax.set_box_aspect(1)

im = ax.pcolormesh(pi, ci, color_value, cmap="Reds")
fig.colorbar(im, ax=ax, label=r"log-error")

mapping = OVERLAP_DATAFILES["mapping"].load().filter(col("pp_delay") > -5)
sns.scatterplot(mapping, x="pp_delay", y="cond_gap", color="black", ax=ax)

cond_limits = (
    TEMPERATURE_DATAFILES["solution"]
    .load()
    .fetch("direction", "temperature", "cond")
    .group_by("direction")
    .agg(col("cond").max().alias("max"), col("cond").min().alias("min"))
    .mean()
)
cond_I = cond_limits["min"].item()
cond_M = cond_limits["max"].item()
ax.axhline(cond_I, color="black", linestyle="--", linewidth=0.5)
ax.axhline(cond_M, color="black", linestyle="--", linewidth=0.5)

ax.annotate(
    r"Insulating phase",
    fontsize=6,
    xy=(1.5, cond_I),
    ha="right",
    va="bottom",
)
ax.annotate(
    r"Metallic phase",
    fontsize=6,
    xy=(1.5, cond_M),
    ha="right",
    va="bottom",
)
ax.set(ylim=(3e3, 2e5))

plt.show()

# %% [markdown]
# ## Scan type illustrations

# %%
fig = plt.figure(figsize=(6.8, 3.4))

ax = fig.subplot_mosaic([["1Dexpl", "2Dexpl"]])


# - Sampling time illustration
def pulse(x, f, tau):
    t = 6.67 * (x + 0.16)
    chisq = np.exp(-0.5 * (t / tau) ** 2) * sigmoid(t, 0, 1)
    return -np.cos(2 * np.pi * (f - 0.5 * f**2) * t) * chisq


def sigmoid(x, mi, mx):
    return mi + (mx - mi) * (lambda t: (1 + 200 ** (-t + 0.5)) ** (-1))(
        (x - mi) / (mx - mi)
    )


def impulse(x, tau):
    t = 6.67 * x
    return np.exp(-0.5 * (t / tau) ** 2) / 2


line_colors = dict(zip(["pump", "probe", "sampler"], sns.color_palette("deep", 3)))

x = np.arange(-1, 1, 1e-3)
amp_pump, amp_probe, amp_sampler = 1, 0.3, 0.3

x0 = {"1Dexpl": 0, "2Dexpl": -0.1}
for key in ["1Dexpl", "2Dexpl"]:
    ax[key].plot(x, amp_pump * pulse(x + 0.4, 1, 1.5), c=line_colors["pump"])
    ax[key].plot(x, 1 + amp_probe * pulse(x, 1, 1.5), c=line_colors["probe"])
    ax[key].plot(x, 2 + amp_sampler * impulse(x - x0[key], 0.05), c=line_colors["sampler"])
    ax[key].set(ylim=(-1, 2.5), yticks=[], xticks=[])

steps = np.linspace(0, 0.8, 5)
for dx in steps:
    ax["1Dexpl"].fill_between(
        *[x, 2],
        2 + amp_sampler * impulse(x + dx, 0.1),
        fc=line_colors["sampler"],
        alpha=(1 - dx / steps[-1]) / 3,
    )
    ax["1Dexpl"].fill_between(
        *[x, 1],
        1 + amp_probe * pulse(x + dx, 1, 1.5),
        fc=line_colors["probe"],
        alpha=(1 - dx / steps[-1]) / 3,
    )
    ax["2Dexpl"].fill_between(
        *[x, 2],
        2 + amp_sampler * impulse(x + dx - x0["2Dexpl"], 0.1),
        fc=line_colors["sampler"],
        alpha=(1 - dx / steps[-1]) / 3,
    )
    ax["2Dexpl"].fill_between(
        *[x, 1],
        1 + amp_probe * pulse(x + dx, 1, 1.5),
        fc=line_colors["probe"],
        alpha=(1 - dx / steps[-1]) / 3,
    )

ax["1Dexpl"].text(-0.03, 1.5, r"$t_0$", va="bottom", ha="right")

offset = -0.6
for key in ["1Dexpl", "2Dexpl"]:
    ax[key].plot(
        [-0.01, -0.01 + x0[key]], [1 + abs(amp_probe), 1.9], c="k", ls="--", lw=0.6
    )
    ax[key].plot([-1, 1], [offset, offset], c="black", lw=0.6)
    ax[key].plot([-0.5, -0.5], [offset, offset + 0.1], c="black", lw=0.6)
    ax[key].plot([0, -0], [offset, offset + 0.1], c="black", lw=0.6)
    ax[key].text(0, offset * 1.05, r"$\tau$", va="top", ha="center")
ax["2Dexpl"].plot(
    [x0["2Dexpl"], x0["2Dexpl"], 0, 0], [2.25, 2.3, 2.3, 2.25], c="black", lw=0.6
)
ax["2Dexpl"].text(0.5 * x0["2Dexpl"], 2.35, r"$t$", va="bottom", ha="center")

pad = 1e-1
props = dict(arrowstyle="-|>", fc="k", lw=0.5)
for key in ["1Dexpl", "2Dexpl"]:
    ax[key].text(0.9, 2 + pad, "Sampler", fontsize="small", va="bottom", ha="right")
    ax[key].text(0.9, 1 + pad, "Probe", fontsize="small", va="bottom", ha="right")
    ax[key].text(0.9, 0 + pad, "Pump", fontsize="small", va="bottom", ha="right")
    ax[key].spines[:].set_visible(False)
    ax[key].annotate(
        "", xy=(0.3 + x0[key], 1.58), xytext=(0.05 + x0[key], 1.58), arrowprops=props
    )

plt.show()

# %% [markdown]
# ## 1D scan

# %%
# | label: fig-1D-scan
# | fig-cap: A 1D-scan, probing a singular point in time on the probe waveform, as a function of the delay time $\tau$ between the THz pump and THz probe pulses.

# - Data import, processing and fitting
data = pl.read_csv(
    PROJECT_PATHS.raw_data
    / "09.02.23/00h14m30s_Pump-induced (166 Hz) peak-track xPP_3.00-4.84_HWP=30.00_Sam=9.33.txt",
    comment_prefix="#",
    separator="\t",
    has_header=False,
    new_columns=["delay", "X"],
).select(
    ((2 / 0.299792) * (163.75 - col("delay"))).alias("time"), col("X") / col("X").max()
)

from lmfit.models import StepModel, LinearModel

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
    xlabel=r"$\tau$ (ps)", ylabel=r"$-\Delta E(t_0)$ (norm.)", xlim=xlim, ylim=(0, 1)
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
from scipy.integrate import cumulative_simpson

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
# fig.savefig()

import svgutils.compose as sc
from cairosvg import svg2pdf

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
    url="_tmp.svg", write_to=str(PROJECT_PATHS.figures / "fast_dynamics/1D-scan.pdf")
)
Path("_tmp.svg").unlink()

plt.show()


# %% [markdown]
# ## Rise and decay time scaling
#
# ### Field strength dependence 

# %%
def powerlaw_model(t, t0, amp, alpha, tau0, sigma):
    step_pump = 0.5 * (1 + erf((t - t0) / sigma))
    decay = amp / (1 + (np.abs(t - t0) / tau0) ** alpha)
    return decay * step_pump


def exponential_decay(t, t0, amp, sigma, decay_time):
    step = 0.5 * (1 + erf((t - t0) * sigma))
    decay = amp * np.exp(-(t - t0) / decay_time)
    return decay * step


model = lm.Model(exponential_decay)
params = model.make_params()
params.add_many(
    ("t0", 0.0, True, None, None, None),
    ("amp", 1, True, 0, None, None),
    ("decay_time", 1, True, 0.1, 10, None),
    ("sigma", 0.1, True, 0.01, 10),
    ("rise_time", 1, True, None, None, "0.238/sigma"),
)


def fit_group(group, model, params, results=[]):
    t, E = group.select("time", "X").get_columns()
    result = model.fit(E, params, t=t, method="nelder-mead")
    results.append(result)
    return group.with_columns(
        pl.lit(result.best_fit).alias("best_line"),
        pl.lit(result.params["decay_time"].value).alias("decay_time"),
        pl.lit(result.params["rise_time"].value).alias("rise_time"),
    )



# %%
# - Field strength dependence
filenames = [
    "14h55m38s_Pump-induced (166 Hz) peak-track xHWP xPP_5.00-6.37_HWP=25.00_Tem=5.00_Sam=9.33.txt",
    "15h45m30s_Pump-induced (166 Hz) peak-track xHWP xPP_5.01-6.37_HWP=30.00_Tem=5.00_Sam=9.33.txt",
    "16h35m10s_Pump-induced (166 Hz) peak-track xHWP xPP_5.02-6.36_HWP=35.00_Tem=5.00_Sam=9.33.txt",
    "17h24m58s_Pump-induced (166 Hz) peak-track xHWP xPP_4.98-6.35_HWP=40.00_Tem=5.00_Sam=9.33.txt",
]
paths = list(map(lambda s: str(PROJECT_PATHS.raw_data / "09.02.23" / s), filenames))
hwp = list(map(lambda path: float(re.search(r"HWP=(\d+)", path).group(1)), paths))

paths = pl.DataFrame({"hwp": hwp, "path": paths}).select(
    col("hwp")
    .map_elements(field_from_hwp, return_dtype=pl.Float64)
    .alias("field_strength"),
    col("path"),
)

data_fs = (
    create_dataset(
        paths,
        column_names=["delay", "X"],
        index="delay",
        lockin_schema={"X": "X"},
        id_schema={"field_strength": pl.Float64},
    )
    .with_columns(
        col("field_strength").round(2),
        ((2 / 0.299792) * (163.6 - col("delay"))).alias("time"),
        col("X") / col("X").max(),
    )
    .set(index="time")
    .drop("delay")
)

g = sns.lineplot(
    data_fs.with_columns(col("field_strength").round(2)),
    x="time",
    y="X",
    hue="field_strength",
    legend=True,
    palette="Set2",
)
plt.show()

# %%
results_fs = []
data_fs = (
    data_fs.group_by("field_strength")
    .map_groups(lambda g: fit_group(g, model, params, results_fs))
    .sort("field_strength")
)
params_fs = (
    data_fs.select(pl.all().exclude("time", "X", "best_line"))
    .unique()
    .sort("field_strength")
)

# %% [markdown]
# ### Temperature dependence

# %%
filenames = [
    "01h18m32s_Pump-induced (166 Hz) xPP-scan xT_4.99-6.37_Tem=5.00_HWP=35.00_Sam=9.66.txt",
    "03h29m11s_Pump-induced (166 Hz) xPP-scan xT_50.00-51.74_Tem=50.00_HWP=35.00_Sam=9.66.txt",
    "05h50m44s_Pump-induced (166 Hz) xPP-scan xT_100.00-101.30_Tem=100.00_HWP=35.00_Sam=9.66.txt",
    "08h26m39s_Pump-induced (166 Hz) xPP-scan xT_150.00-150.90_Tem=150.00_HWP=35.00_Sam=9.66.txt",
]
paths = list(map(lambda s: str(PROJECT_PATHS.raw_data / "15.02.23" / s), filenames))
tempr = list(map(lambda path: float(re.search(r"Tem=(\d+)", path).group(1)), paths))
paths = pl.DataFrame({"tempr": tempr, "path": paths})
data_tempr = (
    create_dataset(
        paths,
        column_names=["delay", "X"],
        index="delay",
        lockin_schema={"X": "X"},
        id_schema={"temperature": pl.Float64},
    )
    .with_columns(
        ((2 / 0.29979) * (163.6 - col("delay"))).alias("time"),
        col("X") / col("X").max(),
    )
    .set(index="time")
    .drop("delay")
)

g = sns.lineplot(
    data_tempr,
    x="time",
    y="X",
    hue="temperature",
    legend=True,
    palette="Reds",
)
plt.show()

# %%
results_tempr = []
data_tempr = (
    data_tempr.group_by("temperature")
    .map_groups(lambda g: fit_group(g, model, params, results_tempr))
    .sort("temperature")
)

# %%
label_map = {
    "field_strength": r"$E_0/E_{0,\text{max}}$",
    "temperature": r"$T$ (K)",
    "rise_time": r"$\tau_0$ (ps)",
    "decay_time": r"$\tau_1$ (ps)",
}


def plot_panel(fig, data, fit_results, id_var, cmap="Reds"):
    axs = fig.subplot_mosaic(
        [["data", "decay_time"], ["data", "rise_time"]], width_ratios=(2, 1)
    )

    plot_kwargs = dict(x="time", hue=id_var, palette=cmap, ax=axs["data"])
    sns.lineplot(
        data,
        y="best_line",
        units=id_var,
        estimator=None,
        color="#333333",
        legend=False,
        **plot_kwargs,
    ).set(ylabel=r"$\Delta E(t_0)$ (norm.)", xlabel=r"$\tau$ (ps)", xlim=(-1.1, 3.1))
    g = sns.scatterplot(data, y="X", legend=True, ec="white", lw=0.6, **plot_kwargs)
    g.legend(title=label_map[id_var], loc="upper left")

    params = (
        data.select(pl.all().exclude("time", "X", "best_line")).unique().sort(id_var)
    ).unpivot(index=id_var)

    for (variable,), group in params.group_by(["variable"]):
        ylim = [group["value"].min(), group["value"].max()]
        ylim = np.asarray(ylim) * np.array([0.5, 1.5])

        sns.scatterplot(
            group,
            x=id_var,
            y="value",
            hue=id_var,
            palette=cmap,
            legend=False,
            s=30,
            ax=axs[variable],
        ).set(ylim=ylim, ylabel=label_map[variable], xlabel="")

    list(axs.values())[-2].tick_params(labelbottom=False)
    list(axs.values())[-1].set(xlabel=label_map[id_var])

    mpl_tools.breathe_axes(axs.values(), axis="y")
    for ax in axs.values():
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

    fig.align_ylabels()


fig = plt.figure(figsize=(6.8, 6.8))
subfigs = fig.subfigures(2, 1)

red_cmap = sns.blend_palette(["#0B4E7D", "#81BACD"], as_cmap=True)
plot_panel(subfigs[0], data_fs, results_fs, "field_strength", cmap=red_cmap)

blue_cmap = sns.blend_palette(["#BB342F", "#FFCAB1"], as_cmap=True)
plot_panel(subfigs[1], data_tempr, results_tempr, "temperature", cmap=blue_cmap)

mpl_tools.enumerate_axes([ax for ax in fig.axes if ax.get_label() == "data"])

fig.align_ylabels()
fig.savefig(PROJECT_PATHS.figures / "fast_dynamics/overlap_dependence.pdf")

plt.show()
