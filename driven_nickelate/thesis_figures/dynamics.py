import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from lmfit import Model, Parameters
from mplstylize import colors, mpl_tools
from polars import col
from scipy.special import erf

from driven_nickelate.config import paths as PROJECT_PATHS

READ_KWARGS = dict(separator="\t", comment_prefix="#", has_header=False)


def decay_model(t, t0, tau_fast, tau_slow, amp_fast, amp_slow):
    step = 0.5 * (1 + erf((t - t0) * 2))
    decay_fast = amp_fast * np.exp(-t / tau_fast)
    decay_slow = amp_slow * np.exp(-t / tau_slow)
    return (decay_fast + decay_slow) * step


if __name__ == "__main__":
    data_1D = pl.read_csv(
        PROJECT_PATHS.raw_data
        / "08.02.23/23h11m23s_Pump-induced (166 Hz) peak-track xPP_3.02-4.84_HWP=30.00_Sam=9.33.txt",
        **READ_KWARGS,
    ).select(
        (6.67 * (163.66 - col("column_1"))).alias("time"),
        (-1 * col("column_2")).alias("signal"),
    )

    model = Model(decay_model)
    params = Parameters()
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

    data_1D = data_1D.with_columns(
        model=result_1D.eval(t=data_1D["time"], params=result_1D.params)
    )

    path = (
        "10.02.23/10h40m29s_Pump (333 Hz) xPP_3.99-5.60_HWP=40.00_Tem=4.00_Sam=9.33.txt"
    )
    data_PNPA = pl.read_csv(
        PROJECT_PATHS.raw_data / path,
        **READ_KWARGS,
    ).select(
        (6.67 * (163.66 - col("column_1"))).alias("time"),
        (col("column_2")).alias("V_pump"),
    )

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
        data_1D,
        x="time",
        y="model",
        color=colors("deep_gray"),
        ls="--",
        ax=axes["probe"],
    )

    x, y = 12.6, model.eval(t=12.6, params=result_1D.params)

    offset_color = colors("caramel")
    axes["probe"].scatter(x, y, c=offset_color, ec="white", s=40, zorder=np.inf)

    # Temperature scan

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
                **READ_KWARGS,
            ).select(
                col("column_1").alias("temperature"),
                (-1 * col("column_2")).alias("signal"),
            )
        )
    data_tempr_scan = pl.concat(data_tempr_scan).collect()

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
