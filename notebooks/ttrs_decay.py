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
# title: "Decay dynamics"
# ---

# %%
# | output: false


import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from lmfit import Model
from mplstylize import colors, mpl_tools
from polars import col
from scipy.special import erf

from driven_nickelate.conductivity_mapping.dynamics_slow import (
    DATAFILES as SLOW_DATAFILES,
)
from driven_nickelate.conductivity_mapping.temperature import (
    DATAFILES as TEMPERATURE_DATAFILES,
)
from driven_nickelate.config import paths as PROJECT_PATHS

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(10)

# %% [markdown]
# ## 2D scans

# %%
axis_kwargs = dict(
    xscale="log",
    ylim=(1, 1e5),
    yscale="linear",
    ylabel=r"$\Delta\sigma$ (S/m)",
    xlabel=r"$\tau$ (ps)",
    xlim=(0.5, 2e3),
)
time = np.geomspace(0.1, 1e4, 10_000)
scatter_kwargs = dict(x="pp_delay", y="delta_cond", s=20)

best_lines = pl.DataFrame({"time": time})

# %%
mapping = (
    SLOW_DATAFILES["solution"]
    .load()
    .select(
        col("pp_delay"), col("cond_gap"), (col("cond_gap") - 6e3).alias("delta_cond")
    )
    .unique()
    .sort("pp_delay")
)

t, cond = mapping.select("pp_delay", "delta_cond").get_columns()

fig, ax = plt.subplots(figsize=(3.4, 3.4))

sns.lineplot(
    mapping, x="pp_delay", y="delta_cond", color=colors("deep_gray"), marker="o", ax=ax
)

ax.set(**axis_kwargs)

plt.show()


# %% [markdown]
# ### Second-order kinetics
#
# $$
#     \frac{\mathrm{d}\sigma}{\mathrm{d}t} = -k \sigma^2
# $$


# %%
def kinetic_model(t, t_pump, k, sigma0, sigma_inf, amp_echo, t_echo):
    step_pump = np.heaviside(t - t_pump, 0)
    decay = sigma0 / (np.abs(t - t_pump) * k * sigma0 + 1) * step_pump

    step_echo = 0.5 * (1 + erf((t - t_echo) / 3))
    decay += amp_echo * (sigma0 / (np.abs(t - t_echo) * k * sigma0 + 1)) * step_echo

    return decay + sigma_inf * step_pump


model = Model(kinetic_model)
params = model.make_params()
params.add_many(
    ("t_pump", 0, False, -1, 1),
    ("t_echo", 16, True, 14, 19),
    ("amp_echo", 0.3, False, 0, 1e12),
    ("sigma0", 1e5, True, 0, 1e6),
    ("sigma_inf", False, 0, 0, 1e4),
    ("k", 1e-4, True, 0, 1),
)

result = model.fit(
    cond.to_numpy(),
    t=t.to_numpy(),
    params=params,
    method="nelder-mead",
)

fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.set(**axis_kwargs)

sns.scatterplot(mapping, ax=ax, label="2nd order kinetics", **scatter_kwargs)
ax.plot(time, model.eval(t=time, params=result.params), color=colors("deep_gray"))

plt.show()


# %% [markdown]
# ### Power-law decay
#
# $$
#     \sigma(t) = \frac{\sigma_0}{(t/\tau_0)^\alpha}
# $$


# %%
def kinetic_model(t, t_pump, sigma0, alpha, amp_echo, t_echo, tau0, delta_sigma):
    step_pump = np.heaviside(t - t_pump, 0)
    decay = sigma0 / np.abs((t - t_pump) / tau0) ** (alpha) * step_pump
    step_echo = 0.5 * (1 + erf((t - t_echo) / 3.0))
    decay *= 1 + step_echo * amp_echo
    return decay + delta_sigma * step_pump


model = Model(kinetic_model)
params = model.make_params()
params.add_many(
    ("t_pump", 0, True, -1, 1),
    ("sigma0", 1.2e5, True, 6e4, 1e7),
    ("alpha", 0.5, True, 0, 1e2),
    ("amp_echo", 1, True, 0, 10),
    ("t_echo", 16, True, 14, 19),
    ("tau0", 1, True, 0, 10),
    ("delta_sigma", 0, False, 0, 1e4),
)

result = model.fit(
    cond.to_numpy(),
    t=t.to_numpy(),
    params=params,
    method="nelder-mead",
)

best_lines = best_lines.with_columns(powerlaw=model.eval(t=time, params=result.params))

fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.set(**axis_kwargs)

sns.scatterplot(mapping, ax=ax, label="Power-law", **scatter_kwargs)

ax.plot(time, model.eval(t=time, params=result.params), color=colors("deep_gray"))

plt.show()


# %% [markdown]
# ### Multi-exponential decay, segregated


# %%
def kinetic_model(t, t0, t1, amp1, amp2, amp3, tau1, tau2, tau3, delta_sigma):
    decay = (
        amp1 * np.exp(-(t - t0) / tau1) + amp2 * np.exp(-(t - t0) / tau2) + delta_sigma
    )
    decay *= 0.5 * (1 + erf((t - t0) * 5))

    echo = amp3 * 0.5 * (1 + erf((t - t1) / 2)) * np.exp(-(t - t1) / tau3)
    return decay + echo


model = Model(kinetic_model)
params = model.make_params()
params.add_many(
    ("t0", -0.5, True, -1, 1),
    ("t1", 16, True, 15, 20),
    ("amp1", 8e4, False, 1e3, 1e6),
    ("amp2", 4e4, True, 1e3, 1e6),
    ("amp3", 1e3, True, 1e3, 1e6),
    ("tau1", 2, True, 1e-6, 10),
    ("tau2", 400, True, 100, 1e4),
    ("tau3", 400, True, 100, 1e4),
    ("delta_sigma", 0, False, 0, 1e4),
)
result = model.fit(
    cond.to_numpy(),
    t=t.to_numpy(),
    params=params,
    method="nelder-mead",
)

fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.set(**axis_kwargs)
ax.set(ylim=(0, None))

sns.scatterplot(mapping, ax=ax, label="multiexpo", **scatter_kwargs)
ax.plot(time, model.eval(t=time, params=result.params), color=colors("deep_gray"))

plt.show()


# %% [markdown]
# ### Multi-exponential decay


# %%
def kinetic_model(
    t, t_pump, amp1, amp2, amp3, tau1, tau2, tau3, delta_sigma, r, t_echo, delta_t
):
    step_pump = np.heaviside(t - t_pump, 0)
    step_echo = 0.5 * (1 + erf((t - t_echo) * 0.3))
    step_total = step_pump + r * step_echo

    t0 = t_pump
    decay = (
        amp1 * np.exp(-(t - t0) / tau1)
        + amp2 * np.exp(-(t - t0) / tau2)
        + amp3 * np.exp(-(t - t0) / tau3)
    )
    decay *= step_total

    return decay + delta_sigma


model = Model(kinetic_model)
params = model.make_params()
params.add_many(
    ("t_pump", 0, True, -1, 1),
    ("amp1", 7e4, True, 1e3, 1e6),
    ("amp2", 2e4, True, 1e3, 1e6),
    ("amp3", 1e4, True, 0, 1e4),
    ("tau1", 3, True, 1e-1, 10),
    ("tau2", 200, True, 50, 1e5),
    ("tau3", 1000, True, 50, 1e5),
    ("delta_sigma", 0, True, 0, 1e4),
    ("t_echo", 17, True, 14, 19),
    ("r", 0.1, True, 0, 1),
)

result = model.fit(
    cond.to_numpy(),
    t=t.to_numpy(),
    params=params,
    method="nelder-mead",
    # weights=(t < 16).to_numpy(),
)

display(result.params)

best_lines = best_lines.with_columns(multiexpo=model.eval(t=time, params=result.params))

fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.set(**axis_kwargs)
ax.set(xlim=(1e-1, 2e3))

sns.scatterplot(mapping, ax=ax, label="multiexpo", **scatter_kwargs)
ax.plot(time, model.eval(t=time, params=result.params), color=colors("deep_gray"))

params_early = result.params.copy()
params_early["r"].value = 0
# ax.fill_between(
#     time, model.eval(t=time, params=params_early), alpha=0.2, color="tomato"
# )

params_late = result.params.copy()
params_late["amp1"].value = 0
params_late["amp2"].value = 0
# ax.fill_between(
#     time, model.eval(t=time, params=params_late), alpha=0.2, color="forestgreen"
# )

plt.show()

# %%
temperature_data = (
    TEMPERATURE_DATAFILES["solution"]
    .load()
    .select(col("cond", "direction", "temperature"))
    .unique()
    .sort("direction", "temperature")
    .with_columns(
        col("temperature").cum_count().over("direction").alias("index"),
    )
    .select(
        col("cond", "temperature").mean().over("index"),
    )
    .unique("temperature")
    .sort("cond")
)

from scipy.interpolate import PchipInterpolator

temperature_interpolator = PchipInterpolator(*temperature_data)


# %% [markdown]
# ### Paper plot

# %%
fig: plt.Figure = plt.figure(figsize=(6.8, 3.4), layout="none")
gs = fig.add_gridspec(2, 1, hspace=0, wspace=0.2, height_ratios=[0.1, 1])

ax = {
    "signal": fig.add_subplot(gs[1]),
    "pnpa": fig.add_subplot(gs[0]),
}
ax["pnpa"].sharex(ax["signal"])
ax["pnpa"].axis("off")

xlim = (0.09, 1.1e3)

ax["signal"].plot(
    *best_lines.select("time", "multiexpo"), color=colors("deep_gray"), zorder=-1
)
sns.scatterplot(
    mapping,
    x="pp_delay",
    y="delta_cond",
    ax=ax["signal"],
    s=25,
    lw=1,
    color=colors("deep_gray"),
)
ax["signal"].set(
    yscale="linear",
    ylim=(0, 1e5),
    ylabel=r"$\Delta\sigma$ (S/m)",
    xlabel=r"$\tau$ (ps)",
)
ax["signal"].set_xscale("symlog", linthresh=0.01)

path = "10.02.23/10h40m29s_Pump (333 Hz) xPP_3.99-5.60_HWP=40.00_Tem=4.00_Sam=9.33.txt"
data_PNPA = (
    pl.read_csv(
        PROJECT_PATHS.raw_data / path,
        separator="\t",
        comment_prefix="#",
        has_header=False,
    )
    .select(
        (6.67 * (163.73 - col("column_1"))).alias("time"),
        (col("column_2").mean() - col("column_2")).alias("signal"),
    )
    .filter(col("time") > xlim[0])
)
pp_delay, signal = data_PNPA
time = np.geomspace(*xlim, 10_000)
signal = np.interp(time, pp_delay, signal)

ax["pnpa"].fill_between(
    time, signal, clip_on=False, fc=colors("deep_gray"), ec="none", alpha=0.1
)
ax["pnpa"].plot(time, signal, clip_on=False, color=colors("deep_gray"))
ax["pnpa"].set(ylim=(0, 1e-1), xlim=xlim)

params_noecho = result.params.copy()
params_noecho["r"].value = 0
ax["signal"].fill_between(
    time,
    model.eval(t=time, params=params_noecho),
    alpha=0.2,
    color=colors("desert_sand"),
    lw=0,
)
ax["signal"].fill_between(
    time,
    model.eval(t=time, params=params_noecho),
    model.eval(t=time, params=result.params),
    alpha=0.2,
    color=colors("cambridge_blue"),
    lw=0,
)

fig.tight_layout(pad=0)
fig.savefig(PROJECT_PATHS.figures / "slow_dynamics" / "exponential_decay.pdf")

plt.show()

# %% [markdown]
# ## One-dimensional scans

# %%
# | label: fig-decay-echos

read_kwargs = dict(separator="\t", comment_prefix="#", has_header=False)

data_1D = pl.read_csv(
    PROJECT_PATHS.raw_data
    / "08.02.23/23h11m23s_Pump-induced (166 Hz) peak-track xPP_3.02-4.84_HWP=30.00_Sam=9.33.txt",
    **read_kwargs,
).select(
    (6.67 * (163.66 - col("column_1"))).alias("time"),
    (-1 * col("column_2")).alias("signal"),
)

# path = "08.02.23/16h36m02s_PNPA_3.02-4.84_Min=30.00_Sam=9.54.txt"
path = "10.02.23/10h40m29s_Pump (333 Hz) xPP_3.99-5.60_HWP=40.00_Tem=4.00_Sam=9.33.txt"
data_PNPA = pl.read_csv(
    PROJECT_PATHS.raw_data / path,
    **read_kwargs,
).select(
    (6.67 * (163.66 - col("column_1"))).alias("time"),
    (col("column_2")).alias("V_pump"),
)


def decay_model(t, t0, tau_fast, tau_slow, amp_fast, amp_slow):
    step = 0.5 * (1 + erf((t - t0) * 2))
    decay_fast = amp_fast * np.exp(-t / tau_fast)
    decay_slow = amp_slow * np.exp(-t / tau_slow)
    return (decay_fast + decay_slow) * step


model = lm.Model(decay_model)
params = lm.Parameters()

params.add_many(
    ("t0", 0, True, -1, 1),
    ("tau_fast", 1, True, 0, 10),
    ("tau_slow", 100, True, 0, 1000),
    ("amp_fast", -1e-3, True, -1e-2, 0),
    ("amp_slow", -1e-3, True, -1e-2, 0),
)
result_1D = model.fit(
    data_1D["signal"].to_numpy(),
    t=data_1D["time"].to_numpy(),
    params=params,
    weights=(data_1D["time"] < 16).to_numpy(),
)

display(result_1D.params)

data_1D = data_1D.with_columns(
    model=result_1D.eval(t=data_1D["time"], params=result_1D.params)
)

fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.4))
sns.lineplot(data_1D, x="time", y="signal", ax=ax[0])
sns.lineplot(data_1D, x="time", y="model", color=colors("deep_gray"), ls="--", ax=ax[0])
sns.lineplot(data_PNPA, x="time", y="V_pump", ax=ax[1])

plt.show()

# %%
# | label: fig-decay-cond

mapping = (
    SLOW_DATAFILES["solution"]
    .load()
    .select(
        col("pp_delay") + 1.7,
        col("cond_gap"),
        (col("cond_gap") - 6e3).alias("delta_cond"),
    )
    .unique()
    .sort("pp_delay")
)


def decay_model(t, t0, tau_fast, tau_slow, amp_fast, amp_slow):
    step = 0.5 * (1 + erf((t - t0) * 3))
    decay_fast = amp_fast * np.exp(-(t - t0) / tau_fast)
    decay_slow = amp_slow * np.exp(-(t - t0) / tau_slow)
    return (decay_fast + decay_slow) * step


model = lm.Model(decay_model)
params = lm.Parameters()

params.add_many(
    ("t0", 0, False, -1, 1),
    ("tau_fast", 3, True, 0, 10),
    ("tau_slow", np.inf, False, 100, None),
    ("amp_fast", 6e4, True, 1e3, 1e6),
    ("amp_slow", 2e4, True, 1e3, 1e6),
)
t, cond = mapping.select("pp_delay", "delta_cond").get_columns()

cond_result = model.fit(cond.to_numpy(), t=t.to_numpy(), params=params)

display(cond_result.params)

mapping = mapping.with_columns(
    model=cond_result.eval(t=mapping["pp_delay"], params=cond_result.params),
    weights=(t < 16).to_numpy(),
)

fig, ax = plt.subplots()

t_cont = np.linspace(-5, 15, 10_000)
ax.plot(
    t_cont,
    cond_result.eval(t=t_cont, params=cond_result.params),
    color=colors("deep_gray"),
    zorder=-1,
    ls="--",
)

sns.scatterplot(
    mapping.filter(col("pp_delay") < 16), x="pp_delay", y="delta_cond", s=20, ax=ax
)
ax.set(xlim=(-2, 15), ylim=(0, 1e5))
mpl_tools.breathe_axes(ax)

plt.show()


# %%
# | label: fig-decay-offset-temperature
# | fig-cap: "Temperature dependence of $\\Delta E$ at a fixed pump-probe delay of 13 ps."

filenames = (
    "19h34m16s_xT offset_210.02-209.71_HWP=30.00_Sam=9.28_Del=145.00.txt",
    "17h27m44s_xT offset_44.88-46.11_Del=145.00_Sam=9.28_HWP=30.00.txt",
    "18h07m18s_xT offset_135.04-135.60_Del=145.00_Sam=9.28_HWP=30.00.txt",
)
data_tempr_scan = []
for name in filenames:
    data_tempr_scan.append(
        pl.scan_csv(
            PROJECT_PATHS.raw_data / "05.03.23" / name,
            **read_kwargs,
        ).select(
            col("column_1").alias("temperature"), (-1 * col("column_2")).alias("signal")
        )
    )
data_tempr_scan = pl.concat(data_tempr_scan).collect()

g = sns.lineplot(
    data_tempr_scan, x="temperature", y="signal", marker="o", color="darkorange"
)
g.set(xlabel=r"$T$ (K)", ylabel=r"$\Delta E(t_0)$ (a.u.)", ylim=(None, 0))

plt.show()

# %%
fig = plt.figure(figsize=(6.8, 3.4))
axes = fig.subplot_mosaic(
    [["probe", "temperature"]],
    gridspec_kw={"wspace": 0.05},
    width_ratios=(3, 2),
)

xlim = (-3, 16)
xlabel = r"$\tau$ (ps)"

axes["probe"].set(
    xlabel=xlabel,
    xlim=xlim,
    ylabel=r"$\Delta E (t_0)$ (a.u.)",
)

# Probe 1D scan
sns.lineplot(data_1D, x="time", y="signal", color=colors("bice"), ax=axes["probe"])
sns.lineplot(
    data_1D, x="time", y="model", color=colors("deep_gray"), ls="--", ax=axes["probe"]
)

x, y = 12.6, model.eval(t=12.6, params=result_1D.params)

offset_color = colors("caramel")
axes["probe"].scatter(x, y, c=offset_color, ec="white", s=40, zorder=np.inf)

# Temperature scan
axes["temperature"].fill_between(
    *data_tempr_scan.with_columns(col("signal").rolling_mean(4, center=True))
    .filter(col("temperature").is_between(10, 200))
    .sort("temperature")
    .drop_nulls(),
    color=offset_color,
    alpha=0.1,
)
g = sns.scatterplot(
    data_tempr_scan,
    x="temperature",
    y="signal",
    ec="white",
    s=20,
    color=offset_color,
    ax=axes["temperature"],
    zorder=np.inf,
)
g.set(
    xlabel=r"$T$ (K)",
    ylabel=r"$\Delta E(t_0)$ (a.u.)",
    ylim=(None, 0),
    xlim=(10, 200),
)

mpl_tools.enumerate_axes(axes.values())
fig.savefig(PROJECT_PATHS.figures / "slow_dynamics/decay_offset.pdf")

plt.show()
