import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from lmfit import Model
from mplstylize import colors
from polars import col
from scipy.special import erf

from driven_nickelate.conductivity_mapping.dynamics_slow import (
    DATAFILES as SLOW_DATAFILES,
)
from driven_nickelate.config import paths as PROJECT_PATHS


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


if __name__ == "__main__":
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

    mapping = (
        SLOW_DATAFILES["solution"]
        .load()
        .select(
            col("pp_delay"),
            col("cond_gap"),
            (col("cond_gap") - 6e3).alias("delta_cond"),
        )
        .unique()
        .sort("pp_delay")
    )

    t, cond = mapping.select("pp_delay", "delta_cond").get_columns()

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

    best_lines = best_lines.with_columns(
        multiexpo=model.eval(t=time, params=result.params)
    )

    params_early = result.params.copy()
    params_early["r"].value = 0

    params_late = result.params.copy()
    params_late["amp1"].value = 0
    params_late["amp2"].value = 0

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

    path = (
        "10.02.23/10h40m29s_Pump (333 Hz) xPP_3.99-5.60_HWP=40.00_Tem=4.00_Sam=9.33.txt"
    )
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
