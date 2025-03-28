# %% [markdown]
# # Photodoping a film

# %%
from pathlib import Path

import lmfit as lm
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import svgutils.compose as sc
from cairosvg import svg2pdf
from IPython.display import display
from matplotlib import rcParams
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import ArrowStyle, FancyArrowPatch
from mplstylize import colors, mpl_tools
from polars import col
from scipy.constants import c as c_const
from scipy.special import erf

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.photodoping.film import DATAFILES as FILM_DATAFILES
from driven_nickelate.tools import create_dataset

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(10)

# %%
waveforms = FILM_DATAFILES["waveforms"].load()
display(waveforms)

# %%
fig: plt.Figure = plt.figure(figsize=(3.4, 3.4))
ax: plt.Axes = fig.add_subplot()

sns.lineplot(
    waveforms.filter(col("time").is_between(-2, 8)).select("time", "X").unique(),
    x="time",
    y="X",
    color=colors("deep_gray"),
    linestyle="--",
    linewidth=0.5,
    ax=ax,
)
sns.lineplot(
    waveforms.filter(col("time").is_between(-2, 8)),
    x="time",
    y="dX",
    hue="fluence",
    palette="Spectral",
    ax=ax,
)
ax.legend(title=r"$F$ (mJ/cm$^2$)", loc="upper left")
ax.set(xlabel=r"$t$ (ps)", ylabel=r"$\Delta X$ (nm)")

plt.show()

# %% [markdown]
# In order to obtain the complex conductivity, we make use of the following thin film approximation:
#
# $$
#     \Delta\sigma = -\frac{2}{Z_0 d}\frac{\Delta \hat{S}}{\hat{S} + \Delta \hat{S}}
# $$ {#eq-thin-film}
#
# Here $\hat{S}$ denotes the Fourier transform of the electric waveform $E(t)$, and $\Delta \hat{S}$ denotes the Fourier transform of the electro-optic differential waveform $\Delta E(t)$.

# %%
spectra = FILM_DATAFILES["spectra"].load()
display(spectra)

# %% [markdown]
# We know the initial conductivity from separate measurements, so that we can calculate the final conductivity as a function of fluence $F$, i.e. $\sigma(F) = \sigma_0 + \Delta \sigma(F)$. The full conductivity in the detectable frequency band is shown in @fig-conductivity.


# %%
def label_func(colname):
    match colname:
        case "dsig.real":
            return r"$\Delta\sigma_1$"
        case "dsig.imag":
            return r"$\Delta\sigma_2$"
    return colname


g = sns.relplot(
    spectra.filter(col("freq").is_between(0.2, 2.3))
    .select_data("dsig.real", "dsig.imag")
    .unpivot(index=["freq", "fluence"])
    .with_columns(col("variable").map_elements(label_func, return_dtype=pl.String)),
    x="freq",
    y="value",
    hue="fluence",
    col="variable",
    palette="Spectral",
    kind="line",
    facet_kws=dict(despine=False),
)


g.set_titles("{col_name}")
g.figure.set_size_inches(6.8, 3.4)
g.set(xlabel=r"$f$ (THz)", ylabel=r"$\Delta \sigma$ (S/m)")
g.legend.set(title=r"$F$ (mJ/cm$^2$)", loc="upper right", bbox_to_anchor=(0.95, 0.99))

plt.show()

# %% [markdown]
# In @fig-conductivity-scaling the average conductivity within the experimental frequency band is shown as a function of pump fluence. Extrapolating, it appears that the full metallic state can be reached by pumping at a fluence of around $1\,\mathrm{mJ/cm^2}$. We are looking for the number of absorbed photons this corresponds to.
#
# The (pseudocubic) unit cell area is $a^2 \approx 0.16\,\mathrm{nm^2}$, and there are 30 unit cells in the film thickness direction. That means the number of unit cells in the excited region is
#
# $$
# N_\mathrm{uc} = 30 \times A_\mathrm{ex} / a^2 \approx 1.3 \times 10^{15}
# $$
#
# The average number of photons in the pulse is
#
# $$
# N_\mathrm{ph} = E_\mathrm{pulse} / \hbar\omega \approx 6.5 \times 10^{16}
# $$
#
# where the energy of the pulse is
#
# $$
#     E_\mathrm{pulse} = 1.0\,\mathrm{mJ/cm^2} \times 7\,\mathrm{mm}^2 = 6.2\times 10^{16}\,\mathrm{eV}
# $$
#
# and the average photon energy is $\hbar\omega = 0.95\,\mathrm{eV}$. The number of photons per unit cell is therefore
#
# $$
#     N_\mathrm{ph} / N_\mathrm{uc} \approx 50
# $$
#
# However, only a fraction of the photons are absorbed. The fraction of the pulse that is absorbed is
#
# $$
#     1 - (1 - R) \mathrm{e}^{-d/\delta} \approx 0.3
# $$
#
# where we used $R = 0.3$, based on measurements by T. Katsufuji et al.

# %%
fig, ax = plt.subplots(figsize=(3.4, 3.4))

sns.lineplot(
    spectra.filter(col("freq").is_between(0.2, 2.3)).group_by("fluence").mean(),
    x="fluence",
    y="dsig.real",
    marker="o",
    mec=None,
    label=r"$\sigma_1$",
    ax=ax,
    color=colors("deep_gray"),
    linestyle="-",
)

sns.lineplot(
    spectra.filter(col("freq").is_between(0.2, 2.3)).group_by("fluence").mean(),
    x="fluence",
    y="dsig.imag",
    marker="v",
    mec=None,
    label=r"$\sigma_2$",
    ax=ax,
    color=colors("deep_gray"),
    linestyle="--",
)

ax.set(
    xlabel=r"Pump fluence (mJ/cm$^2$)",
    ylabel=r"$\overline{\sigma}$ (S/m)",
    xlim=(0, None),
)

ax.legend(loc="upper left")
plt.show()

# %%
import numpy as np

omega = 2 * np.pi * 1e12  # [rad/s]
sigma = 1e5  # [S/m]
mu0 = 4 * np.pi * 1e-7  # [H/m]

thickness = 11e-9  # [m]

delta = np.sqrt(2 / (mu0 * omega * sigma))

reflected = 0.2
transmitted = (1 - reflected) * np.exp(-thickness / delta)

absorbed = 1 - transmitted


pump_energy = 6.24e15  # [eV/cm^2], (1.0 mJ/cm^2)
photon_energy = 0.95  # [eV]

photon_density = pump_energy / photon_energy  # [1/cm^2]

area_excited = np.pi * 3.5e-3**2  # [m^2]
num_photons = photon_density * area_excited

print(num_photons)

a = 0.4e-9  # [m]
num_uc = 30 * area_excited / a**2

print(num_photons / num_uc)

# %%
root = PROJECT_PATHS.raw_data / "06.03.23"
files = [
    "11h09m39s_NNO film 1D scan_4.02-5.18_HWP=20.60_Sam=9.51.txt",
    "10h53m33s_NNO film 1D scan_3.99-5.24_HWP=26.70_Sam=9.51.txt",
    "11h25m17s_NNO film 1D scan_4.02-5.25_HWP=29.73_Sam=9.51.txt",
]
powers = [80, 125, 150]
paths = pl.DataFrame(
    {"power": pwr, "path": root / fn} for (fn, pwr) in zip(files, powers)
)

data = (
    create_dataset(
        paths,
        index="delay",
        column_names=["delay", "signal"],
        lockin_schema={"signal": "signal"},
        id_schema={"power": pl.Float64},
    )
    .with_columns(((2e9 / c_const) * (147.3 - col("delay"))).alias("time"))
    .set(index="time")
    .select(col("power", "time", "signal"))
)

sns.lineplot(data, x="time", y="signal", hue="power")

# %%
root = PROJECT_PATHS.raw_data / "06.03.23"
file = "09h57m00s_NNO film pump power dependence_4.01-5.29_Min=147.22_Sam=9.33.txt"
data_power_dependence = pl.read_csv(
    root / file, separator="\t", has_header=False, comment_prefix="#"
).select(
    col("column_1").alias("hwp_pos"),
    col("column_2").alias("signal"),
)


def waveplate_power(x, x0, y0, A, P):
    return y0 + A * np.sin(2 * np.pi * (x - x0) / P) ** 2


model = lm.Model(waveplate_power)
params = model.make_params(x0=8, y0=40, A=250, P=120)

power_data = pl.read_csv(
    root / "09h04m00s_HWPvsPowerOPASignalNoND.csv", comment_prefix="#"
).select(
    col("HWP").alias("hwp_pos"),
    col("Power").alias("power"),
)

result = model.fit(power_data["power"], params, x=power_data["hwp_pos"])

data_power_dependence = data_power_dependence.with_columns(
    pl.lit(result.eval(x=data_power_dependence["hwp_pos"])).alias("power")
)

fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.fill_between(
    *data_power_dependence.select(col.power, col.signal),
    alpha=0.1,
)
ax.plot(
    *data_power_dependence.select(col.power, col.signal),
)
ax.set(xlim=(0, None), xlabel=r"Power (mW)", ylabel=r"$\Delta E$")
plt.show()

# %% [markdown]
# ## Sample comparison

# %%
root = PROJECT_PATHS.raw_data

files = [
    "05.03.23/11h36m14s_Pump chopped gate scan long delay_3.02-4.37_Del=140.00_HWP=30.00.txt",
    "06.03.23/07h19m30s_NNO film pump chopped_4.01-5.21_Tem=4.00_HWP=30.00_Del=140.00.txt",
    "06.03.23/10h32m45s_NNO film transient probe scan_4.02-5.22_Del=147.00_HWP=26.70.txt",
]
files = [root / fn for fn in files]
samples = ["esrr", "film", "film"]
powers = [38, 38, 125]
fluence = (
    1e-3 * np.array(powers) / (np.pi * 0.23**2) * (1 - np.exp(-2 * 0.15**2 / 0.23**2))
)


paths = pl.DataFrame({"sample": samples, "power": powers, "files": files})

data = (
    create_dataset(
        paths,
        index="delay",
        column_names=["delay", "signal"],
        lockin_schema={"signal": "signal"},
        id_schema={"sample": pl.Utf8, "power": pl.Float64},
    )
    .with_columns(((2e9 / c_const) * (9.75 - col("delay"))).alias("time"))
    .set(index="time")
    .sort("time")
    .select(col("sample", "power", "time", "signal"))
)


fig = plt.figure(figsize=(6.8, 6.8))
axs = fig.subplot_mosaic(
    [
        ["film_svg", "film_lp", "film_hp"],
        [".", "power", "power"],
        ["esrr_svg", "esrr", "."],
    ],
    sharey=True,
)

axs["esrr"].plot(
    *data.filter(col("sample") == "esrr").select("time", "signal").get_columns()
)
axs["film_lp"].plot(
    *data.filter(col("sample") == "film")
    .filter(col("power") == 38)
    .select("time", "signal")
    .get_columns()
)
axs["film_hp"].plot(
    *data.filter(col("sample") == "film")
    .filter(col("power") == 125)
    .select("time", "signal")
    .get_columns()
)


axs["power"].fill_between(
    *data_power_dependence.select("power", "signal").get_columns(), alpha=0.1
)
axs["power"].plot(*data_power_dependence.select("power", "signal").get_columns())

transform = axs["power"].transData
axs["power"].annotate(
    "",
    xy=[38, -3.2e-3],
    xytext=[38, -2.5e-3],
    xycoords=transform,
    textcoords=transform,
    arrowprops=dict(facecolor=colors("deep_gray"), arrowstyle="-|>", linestyle="-"),
)
axs["power"].annotate(
    "",
    xy=[38, 3.2e-3],
    xytext=[38, 2.5e-3],
    xycoords=transform,
    textcoords=transform,
    arrowprops=dict(facecolor=colors("deep_gray"), arrowstyle="-|>", linestyle="-"),
)
axs["power"].annotate(
    "",
    xy=[125, 3.2e-3],
    xytext=[125, 2.5e-3],
    xycoords=transform,
    textcoords=transform,
    arrowprops=dict(facecolor=colors("deep_gray"), arrowstyle="-|>", linestyle="-"),
)

axs["power"].axvline(38, linestyle="--", color=colors("deep_gray"), lw=0.6)
axs["power"].axvline(125, linestyle="--", color=colors("deep_gray"), lw=0.6)

data_axs = [axs["film_lp"], axs["film_hp"], axs["esrr"]]
for ax in data_axs:
    ax.set(xlabel=r"$t$ (ps)", ylabel=r"$\Delta E\,(t)$", ylim=(-2.5e-3, 2.5e-3))
axs["power"].set(xlim=(0, 210), xlabel=r"Pump power (mW)", ylabel=r"$\Delta E\,(t_0)$")

for ax in [axs["film_svg"], axs["esrr_svg"]]:
    ax.tick_params(
        labelleft=False,
        labelright=False,
        labelbottom=False,
        labeltop=False,
        left=False,
        right=False,
        bottom=False,
        top=False,
    )
    ax.spines[:].set_visible(False)

data_axs.insert(2, axs["power"])
mpl_tools.enumerate_axes(data_axs)


x, y = 72 * fig.get_size_inches()
sc.Figure(
    f"{x}pt",
    f"{y}pt",
    sc.MplFigure(fig),
    sc.SVG(PROJECT_PATHS.figures / "illustrations/no_resonator.svg")
    .scale(0.06)
    .move(15, 15),
    sc.SVG(PROJECT_PATHS.figures / "illustrations/one_resonator.svg")
    .scale(0.06)
    .move(15, 340),
).save("_tmp.svg")

svg2pdf(
    url="_tmp.svg", write_to=str(PROJECT_PATHS.figures / "photodoping/film_vs_esrr.pdf")
)

Path("_tmp.svg").unlink()

plt.show()

# %%
fig = plt.figure(figsize=(6.8, 3.4))
axs = fig.subplot_mosaic(
    [["time", "fluence", "cbar"]],
    width_ratios=(1, 1, 0.05),
)

dotted_kwargs = dict(color="k", marker="o", mec="white")
fluence_norm = Normalize(0, 0.8)
color_palette = sns.color_palette("crest_r", as_cmap=True)

cbar = fig.colorbar(
    ScalarMappable(norm=fluence_norm, cmap=color_palette),
    cax=axs["cbar"],
    label=r"$F\,{\rm (mJ/cm^2)}$",
)


def power_to_fluence(power):
    return (
        1e-3 * power / (np.pi * 0.23**2) * (1 - np.exp(-2 * 0.15**2 / 0.23**2))
    )  # mJ/cm^2


# - eSRR

files = [
    "12h02m07s_ZnTe F21059_eSRR gate scan xHWP_4.00-5.14_Del=146.60_HWP=45.00.txt",
    "12h11m45s_ZnTe F21059_eSRR gate scan xHWP_3.99-5.16_Del=146.60_HWP=41.82.txt",
    "12h21m23s_ZnTe F21059_eSRR gate scan xHWP_4.01-5.16_Del=146.60_HWP=39.09.txt",
    "12h31m01s_ZnTe F21059_eSRR gate scan xHWP_3.98-5.18_Del=146.60_HWP=36.82.txt",
    "12h40m39s_ZnTe F21059_eSRR gate scan xHWP_3.99-5.19_Del=146.60_HWP=35.00.txt",
    "12h50m18s_ZnTe F21059_eSRR gate scan xHWP_3.99-5.18_Del=146.60_HWP=33.18.txt",
    "14h03m18s_ZnTe F21059_eSRR gate scan xHWP_4.02-5.19_Del=146.60_HWP=31.36.txt",
    "13h48m33s_ZnTe F21059_eSRR gate scan xHWP_3.98-5.19_Del=146.60_HWP=30.00.txt",
    "14h19m02s_ZnTe F21059_eSRR gate scan xHWP_4.01-5.20_Del=146.60_HWP=28.18.txt",
    "14h28m41s_ZnTe F21059_eSRR gate scan xHWP_4.00-5.20_Del=146.60_HWP=26.82.txt",
    "14h38m19s_ZnTe F21059_eSRR gate scan xHWP_4.00-5.22_Del=146.60_HWP=25.00.txt",
]
powers = np.arange(25, 25 + 5 * len(files), 5)
paths = pl.DataFrame(
    {
        "fluence": power_to_fluence(powers),
        "paths": [PROJECT_PATHS.raw_data / "07.03.23" / fn for fn in files],
    }
)


data = (
    create_dataset(
        paths,
        index="delay",
        column_names=["delay", "signal"],
        lockin_schema={"signal": "signal"},
        id_schema={"fluence": pl.Float64},
    )
    .with_columns(((2e9 / c_const) * (9.75 - col("delay"))).alias("time"))
    .set(index="time")
    .select(col("fluence", "time", "signal"))
)

data_ref = pl.read_csv(
    PROJECT_PATHS.raw_data
    / "07.03.23/11h36m46s_ZnTe F21059_eSRR  38 uJ gate scan xPP PUMP BLOCKED_4.01-5.13_Del=0.00_HWP=37.66.txt",
    separator="\t",
    comment_prefix="#",
    has_header=False,
).select(
    ((2e9 / c_const) * (9.75 - col("column_1"))).alias("time"),
    col("column_2").alias("signal_eq"),
)
data = data.join(data_ref, on="time").with_columns(
    (col("signal") - col("signal_eq")).alias("signal_diff")
)
data_energy = (
    data.group_by("fluence")
    .agg(col("signal_diff").pow(2).mean().sqrt().alias("rms"))
    .sort("fluence")
)

sns.lineplot(
    data.with_columns(col("signal_diff") + 2e-3),
    x="time",
    y="signal_diff",
    hue="fluence",
    hue_norm=fluence_norm,
    legend=False,
    palette=color_palette,
    ax=axs["time"],
)
axs["fluence"].plot(
    *data_energy,
    c="k",
    ls="-",
    zorder=-1,
    label="Metasurface",
)
sns.scatterplot(
    data_energy,
    x="fluence",
    y="rms",
    hue="fluence",
    hue_norm=fluence_norm,
    palette=color_palette,
    s=30,
    legend=False,
    ax=axs["fluence"],
)


# - Film

files = [
    "12h20m54s_NNO film transient probe scan xHWP pump 20ps_3.99-5.16_Del=144.30_HWP=15.13.txt",
    "12h38m37s_NNO film transient probe scan xHWP pump 20ps_3.98-5.19_Del=144.30_HWP=20.59.txt",
    "12h56m21s_NNO film transient probe scan xHWP pump 20ps_4.00-5.21_Del=144.30_HWP=24.77.txt",
    "13h14m05s_NNO film transient probe scan xHWP pump 20ps_4.01-5.24_Del=144.30_HWP=28.51.txt",
    "13h31m49s_NNO film transient probe scan xHWP pump 20ps_4.00-5.26_Del=144.30_HWP=32.21.txt",
    "14h02m24s_NNO film transient probe scan xHWP pump 20ps_4.01-5.28_Del=144.30_HWP=36.08.txt",
    "14h26m47s_NNO film transient probe scan xHWP pump 20ps_4.01-5.30_Del=144.30_HWP=38.92.txt",
]
powers = np.asarray([50, 80, 110, 140, 170, 200, 220])
paths = pl.DataFrame(
    {
        "fluence": power_to_fluence(powers),
        "paths": [PROJECT_PATHS.raw_data / "06.03.23" / fn for fn in files],
    }
)

data = (
    create_dataset(
        paths,
        index="delay",
        column_names=["delay", "signal_diff"],
        lockin_schema={"signal_diff": "signal_diff"},
        id_schema={"fluence": pl.Float64},
    )
    .with_columns(
        ((2e9 / c_const) * (9.75 - col("delay"))).alias("time"),
        (col("signal_diff") - col("signal_diff").mean()).over("fluence"),
    )
    .set(index="time")
    .select(col("fluence", "time", "signal_diff"))
)


sns.lineplot(
    data.with_columns(col("signal_diff") - 2e-3),
    x="time",
    y="signal_diff",
    hue="fluence",
    hue_norm=fluence_norm,
    legend=False,
    palette=color_palette,
    ax=axs["time"],
)

data_energy = (
    data.group_by("fluence")
    .agg(col("signal_diff").pow(2).mean().sqrt().alias("rms"))
    .sort("fluence")
)

axs["fluence"].plot(
    *data_energy,
    c="k",
    ls="--",
    zorder=-1,
    label="Blank film",
)
sns.scatterplot(
    data_energy,
    x="fluence",
    y="rms",
    hue="fluence",
    hue_norm=fluence_norm,
    palette=color_palette,
    s=30,
    legend=False,
    ax=axs["fluence"],
)

axs["time"].annotate(
    "Metasurface", xy=(7, 1.0e-3), xycoords="data", ha="center", va="center", color="k"
)
axs["time"].annotate(
    "Blank film", xy=(7, -1.5e-3), xycoords="data", ha="center", va="center", color="k"
)


axs["fluence"].set(
    xlim=(0, 0.8),
    ylabel=r"$\Delta E_\text{rms}$ (a.u.)",
    xlabel=r"$F\,{\rm (mJ/cm^2)}$",
    ylim=(0, 1.2e-3),
)
axs["time"].set(xlim=(-2, 9), xlabel=r"$t$ (ps)", ylabel=r"$\Delta E$ (a.u.)")

axs.pop("cbar")
mpl_tools.enumerate_axes(axs.values())
mpl_tools.breathe_axes(axs["fluence"])

import svgutils.compose as sc
from cairosvg import svg2pdf

x, y = 72 * fig.get_size_inches()
sc.Figure(
    f"{x}pt",
    f"{y}pt",
    sc.MplFigure(fig),
    sc.SVG(PROJECT_PATHS.figures / "illustrations/no_resonator.svg")
    .scale(0.03)
    .move(370, 40),
    sc.SVG(PROJECT_PATHS.figures / "illustrations/one_resonator.svg")
    .scale(0.03)
    .move(290, 20),
).save("_tmp.svg")

svg2pdf(
    url="_tmp.svg", write_to=str(PROJECT_PATHS.figures / "photodoping/film_vs_esrr.pdf")
)

Path("_tmp.svg").unlink()

plt.show()


# %%
def decay_expr(t, t0, total_amp, amp1, amp2, sig0, tau1, tau2, offset):
    t = t - t0
    rise = total_amp * (1 + erf(t * sig0)) * 0.5
    return rise * (amp1 * np.exp(-t / tau1) + amp2 * np.exp(-t / tau2) + offset)


model = lm.Model(decay_expr)
params = model.make_params(
    t0=0, total_amp=1e-3, amp1=1, amp2=0.5, sig0=4, tau1=3, tau2=500, offset=0
)
params["sig0"].set(min=1e-2)
params["tau1"].set(min=1e-2)
params.add("tau0", expr="0.238/sig0")

fname = (
    PROJECT_PATHS.raw_data
    / "05.03.23/13h39m17s_Pump chopped pump-probe scan nonlinear steps_3.01-4.37_Sam=9.14_HWP=30.00.txt"
)
data = (
    pl.read_csv(fname, separator="\t", comment_prefix="#", has_header=False)
    .select(
        ((2e9 / c_const) * (147 - col("column_1"))).alias("time"),
        col("column_2").alias("signal"),
    )
    .with_columns(col("signal") - col("signal").filter(col("time") < -0.2).mean())
)

result = model.fit(data["signal"], params, t=data["time"])

# %%
# | label: fig-decay-timescales
# | fig-cap: Decay times after photoexcitation of the eSRR sample.


fig = plt.figure(figsize=(6.8, 3.4), layout="none")
axs = fig.subplot_mosaic([["short", "long"]])

big_ax = plt.Subplot(fig, axs["short"].get_gridspec()[-1, :], frameon=False)
big_ax.set(xticks=[], yticks=[])
big_ax.patch.set_facecolor("none")
big_ax.set_xlabel(r"$\tau\,({\rm ps})$", labelpad=20)

fig.add_subplot(big_ax)

time_stop = 5
time_start = 20
axs["short"].scatter(
    *data.filter(col("time") < time_stop), ec="white", c=colors("bice"), s=20
)
axs["long"].scatter(
    *data.filter(col("time") > time_start), ec="white", c=colors("bice"), s=20
)

time_short = np.linspace(-3, time_stop, 1000)
time_long = np.linspace(time_start, 700, 1000)
axs["short"].plot(time_short, result.eval(t=time_short), color=colors("deep_gray"))
axs["long"].plot(time_long, result.eval(t=time_long), color=colors("deep_gray"))

axs["inset"] = axs["long"].inset_axes([0.4, 0.5, 0.5, 0.4])
axs["inset"].plot(
    data["time"].log10(),
    data["signal"],
    color=colors("deep_gray"),
    # mec="white",
    mew=0.6,
    ms=2,
)
axs["inset"].set(xlabel=r"$\log_{10}(\tau)$", ylabel=r"$\Delta E$ (a.u.)")


def create_breaklines(ax_left, ax_right):
    d = 0.02
    kwargs = dict(
        transform=ax_left.transAxes,
        color="k",
        clip_on=False,
        lw=rcParams["axes.linewidth"],
    )
    theta = np.deg2rad(45)
    dx, dy = d * np.array([np.cos(theta), np.sin(theta)])

    ax_left.plot([1.01 - dx, 1.01 + dx], [0 - dy, 0 + dy], **kwargs)
    ax_left.plot([1.01 - dx, 1.01 + dx], [1 - dy, 1 + dy], **kwargs)

    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot([0 - dx, 0 + dx], [0 - dy, 0 + dy], **kwargs)
    ax_right.plot([0 - dx, 0 + dx], [1 - dy, 1 + dy], **kwargs)


create_breaklines(axs["short"], axs["long"])


def annotate_timescales(ax_left, ax_right):
    double_bracket = ArrowStyle.BarAB(widthA=2, widthB=2)
    left_bracket = ArrowStyle.BarAB(widthA=2, widthB=0)
    right_bracket = ArrowStyle.BarAB(widthA=0, widthB=2)
    right_arrow = ArrowStyle.CurveFilledB(head_width=2, head_length=2)

    t0, tau1, tau2 = (
        result.params["t0"],
        result.params["tau1"],
        result.params["tau2"],
    )

    y0 = 3e-4
    ax_left.add_patch(
        FancyArrowPatch((t0, y0), (t0 + tau1, y0), arrowstyle=double_bracket)
    )
    ax_left.text(t0 + tau1, y0, r"$\tau_1$", ha="left", va="center")

    y0 -= 1e-4
    ax_left.add_patch(FancyArrowPatch((t0, y0), (5, y0), arrowstyle=left_bracket))
    ax_right.add_patch(
        FancyArrowPatch((t0, y0), (t0 + tau2, y0), arrowstyle=right_bracket)
    )
    ax_right.text(t0 + tau2, y0, r"$\tau_2$", ha="left", va="center")

    y0 -= 1e-4
    ax_right.add_patch(FancyArrowPatch((t0, y0), (5, y0), arrowstyle=left_bracket))
    ax_right.add_patch(FancyArrowPatch((t0, y0), (600, y0), arrowstyle=right_arrow))
    ax_right.text(600, 1e-4, r"$\tau_3 = \infty$", ha="left", va="center")


annotate_timescales(axs["short"], axs["long"])

axs["long"].sharey(axs["short"])
axs["long"].spines["left"].set_visible(False)
axs["long"].yaxis.set_tick_params(labelleft=False, left=False)

axs["short"].spines["right"].set_visible(False)
axs["short"].yaxis.set_tick_params(labelright=False, right=False)
axs["short"].set(ylabel=r"$\Delta E(t_0)$ (a.u.)")

axs["long"].set(xlim=(time_start, None))
axs["short"].set(xlim=(None, time_stop), ylim=(-1e-4, 1.5e-3))

fig.tight_layout(pad=0.5)
fig.savefig(PROJECT_PATHS.figures / "photodoping/decay_timescales.pdf")

plt.show()
