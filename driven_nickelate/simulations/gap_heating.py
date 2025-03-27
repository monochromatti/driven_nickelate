import logging

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.colors import LogNorm
from polars_dataset import Datafile, Dataset
from scipy.fft import irfft
from scipy.integrate import simpson

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.simulations.pnpa import DATAFILES as PNPA_DATAFILES

PROJECT_ROOT = PROJECT_PATHS.root / "simulations"
MAX_FIELDSTRENGTH = 0.192  # [MV/cm]

STORE_DIR = PROJECT_PATHS.processed_data / "simulations/gap_heating"
STORE_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_PATHS.figures / "simulations").mkdir(parents=True, exist_ok=True)

DATAFILES = {
    "spatio_spectral": Datafile(
        path=STORE_DIR / "spatio_spectral.csv", index="freq", id_vars=["cond"]
    ),
    "gap_temporal": Datafile(
        path=STORE_DIR / "gap_temporal.csv", index="time", id_vars=["cond"]
    ),
}


def scale_fieldstrength(x: pl.Series) -> pl.Series:
    x = x / x.max() * MAX_FIELDSTRENGTH
    return x


def import_spatial_data():
    data_folder = (
        PROJECT_PATHS.root / "simulations/data/01.03.24/gapfield_pnpa_cond_many"
    )
    paths = [f for f in data_folder.glob("*.csv") if "spatial_data" in f.name]

    def read_coords(file):
        lines = []
        while not (line := file.readline()).startswith("% Data"):
            lines.append(line.strip())
        return lines

    def read_data(file):
        lines = []
        while not (line := file.readline()).startswith("% Data"):
            if not line:
                break
            lines.append(line.strip())
        return lines

    def process_header(line):
        name, freq = line.strip("% |\n").split(" @ ")
        name, unit = name.replace("emw.", "").split(" ")
        unit = unit.strip("()")
        freq = float(freq.strip("freq="))
        return name, unit, freq

    lazyframes = []
    for path in paths:
        logging.info(f"Importing {path.name} ...")
        datagroup = []
        condgroup = []
        with open(path) as f:
            while not f.readline().startswith("% Grid"):
                continue

            x_str, y_str = read_coords(f)
            y = pl.Series("y", y_str.split(",")).cast(pl.Float64)

            while line := f.readline():
                name, unit, freq = process_header(line)
                data = read_data(f)
                data = [s.strip().split(",") for s in data]

                data = (
                    pl.LazyFrame(data, schema=x_str.split(","))
                    .with_columns(y)
                    .unpivot(
                        index="y",
                        variable_name="x",
                    )
                    .select(
                        pl.lit(name).alias("variable"),
                        pl.lit(unit).alias("unit"),
                        pl.lit(freq).alias("freq"),
                        pl.col("x").cast(pl.Float64),
                        pl.col("y"),
                        pl.col("value").cast(pl.Float64),
                    )
                )
                if name == "real(sigmabnd)":
                    cond = data.select(
                        pl.col("x", "y", "freq"), pl.col("value").alias("cond")
                    )
                    condgroup.append(cond)
                else:
                    datagroup.append(data)
        datagroup = pl.concat(datagroup).join(
            pl.concat(condgroup), on=["x", "y", "freq"]
        )
        lazyframes.append(datagroup)

    data = pl.concat(lazyframes).collect()
    data = data.vstack(  # Handle missing zero-frequency value
        data.filter(pl.col("freq") == pl.col("freq").min()).with_columns(
            pl.lit(0.0).alias("freq")
        )
    ).sort("variable", "cond", "freq", "x", "y")

    DATAFILES["spatio_spectral"].write(data)


def window(t):
    t1, t2 = -2, 3
    left = np.min([np.ones(len(t)), np.exp((t - t1) / 0.5)], axis=0)
    right = np.min([np.ones(len(t)), np.exp(-(t - t2) / 2)], axis=0)
    return left + right - 1


def to_time(gap_data):
    pnpa_data = PNPA_DATAFILES["waveform"].load()
    time = pnpa_data["time"].to_numpy()

    data_time = []
    for (cond,), data in gap_data.group_by(["cond"]):
        electric_field = data.select("real(Ey)", "imag(Ey)").to_numpy()
        electric_field = np.fft.irfft(
            electric_field[:, 0] + 1j * electric_field[:, 1]
        )  # [V/m], “field per distance (y)”

        surface_current_density = data.select("real(Jsupy)", "imag(Jsupy)").to_numpy()
        surface_current_density = np.fft.ifft(
            surface_current_density[:, 0] + 1j * surface_current_density[:, 1]
        )  # [A/m], “current per width”

        Qsrh = data["Qsrh"].to_numpy()
        Qsrh = np.fft.ifft(Qsrh)

        group = pl.DataFrame(
            {
                "time": time,
                "cond": cond,
                "surface_current_density": surface_current_density,
                "electric_field": electric_field,
                "Qsrh": Qsrh,
            }
        )
        data_time.append(group)
    data_time = pl.concat(data_time)
    return data_time


def struct_integral(s: pl.Series):
    x, y = s.struct[0], s.struct[1]
    integral = simpson(y=y, x=x)
    return integral


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data = pl.read_csv(
        PROJECT_PATHS.root / "simulations/data/06.03.24/gapfield_q3D/spectral_data.csv"
    ).with_columns(
        (pl.col("qrh") * 300e-9 * 3e-6 * 12e-9).alias("qrh"),
    )
    data = data.vstack(  # Handle missing zero-frequency value
        data.filter(pl.col("freq") == pl.col("freq").min()).with_columns(
            pl.lit(0.0).alias("freq"),
            pl.lit(0.0).alias("Ey.imag"),
            pl.lit(0.0).alias("Jy.imag"),
        )
    ).sort("freq")
    data = Dataset(data, index="freq", id_vars=["cond"])

    import seaborn as sns

    sns.lineplot(
        data,
        x="freq",
        y="Qrhy",
        hue="cond",
        hue_norm=LogNorm(),
        palette="Spectral",
    ).set(xlim=(0, 4), yscale="log")
    plt.show()

    data_em = data.filter((pl.col("freq") - 1.1).abs() < 1e-2).df.select(
        "cond", r"^Ey.*$", r"^Jy.*$"
    )
    pnpa_data = PNPA_DATAFILES["waveform"].load()
    time = pnpa_data["time"].to_numpy()  # [ps]

    def ifft(array):
        array = np.r_[array[0], array[1:], np.conj(array[1:][::-1])]
        array_ifft = np.fft.ifft(array, n=2 * len(time))
        return array_ifft

    data_time = []
    for (cond,), group in data.sort("freq", "cond").group_by(["cond"]):
        freq = group.get_column("freq").to_numpy()
        source_distance = 10e-3  # [mm]
        phase = 2 * np.pi * freq * source_distance / 0.3

        electric_field = group.select("Ey.real", "Ey.imag").to_numpy()
        electric_field = electric_field[:, 0] + 1j * electric_field[:, 1]
        electric_field = irfft(electric_field)

        current_density = group.select("Jy.real", "Jy.imag").to_numpy()
        current_density = current_density[:, 0] + 1j * current_density[:, 1]
        current_density = irfft(current_density)

        # volume = 300e-9 * 3e-6 * 12e-9
        energy_density = simpson(
            y=current_density * electric_field,  # [W/m^3]
            x=1e-12 * time,  # [s]
        )  # [J/m^3]

        specific_heat = 0.6e5  # J/m^3 K
        final_temperature = energy_density / specific_heat

        data_time.append(
            pl.DataFrame(
                {
                    "time": time,
                    "cond": cond,
                    "Ey": electric_field,
                    "Jy": current_density,
                    "energy_density": energy_density,
                    "final_temperature": final_temperature,
                }
            )
        )

    data_time = pl.concat(data_time)

    volume = 300e-9 * 3e-6 * 12e-9
    data_time = data_time.with_columns(
        (pl.col("Ey") * pl.col("Jy") * volume).alias("Q")
    )

    sns.lineplot(data_time, x="cond", y="final_temperature").set(xscale="log")
    plt.show()

    from scipy.interpolate import interp1d

    df = pl.read_csv(
        PROJECT_PATHS.root / "literature_data/heat_capacity.csv",
        separator=",",
        comment_prefix="#",
    )

    T, C = df["T (K)"], df["C (J/mol K)"]

    density = 7.24  # [g/cm^3]
    molar_mass = 250.9  # [g/mol]
    C = C / molar_mass * density * 1e6  # [J/m^3 K]

    T_c = np.arange(T.min(), T.max(), 1e-3)
    C_c = interp1d(T, C, kind="cubic")(T_c)

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.plot(T, C, "o", markersize=5)
    ax.plot(T_c, C_c)
    ax.set(
        xlabel="Temperature (K)",
        ylabel=r"Specific heat, $c$ (J/ m^3 K)",
    )
    plt.show()

    plt.plot(T_c[1:], np.diff(C_c))
    plt.show()
