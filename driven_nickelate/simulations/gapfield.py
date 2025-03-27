import numpy as np
import polars as pl
from polars_complex import ccol
from polars_dataset import Datafile, Dataset

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.simulations.pnpa import DATAFILES as PNPA_DATAFILES
from driven_nickelate.tools import pl_fft

PROJECT_ROOT = PROJECT_PATHS.root / "simulations"
MAX_FIELDSTRENGTH = 0.192  # [MV/cm]

STORE_DIR = PROJECT_PATHS.processed_data / "simulations"
STORE_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_PATHS.figures / "simulations").mkdir(parents=True, exist_ok=True)

DATAFILES = {
    "spectra": Datafile(
        path=STORE_DIR / "gapfield_spectral.csv",
        index="freq",
        id_vars=["cond"],
    ),
    "temporal": Datafile(
        path=STORE_DIR / "gapfield_temporal.csv",
        index="time",
        id_vars=["cond"],
    ),
}


def ifft(data: pl.DataFrame, var: str) -> np.ndarray:
    data_subset = data.select("freq", f"{var}.mag", f"{var}.pha").to_numpy(
        structured=True
    )
    complex_amplitude = data_subset[f"{var}.mag"] * np.exp(
        -1j * data_subset[f"{var}.pha"]
    )

    t_shift = 1.0  # [ps]
    complex_amplitude *= np.exp(-1j * data_subset["freq"] * 2 * np.pi * t_shift)
    return np.fft.irfft(complex_amplitude, norm="backward")


def scale_fieldstrength(x: pl.Series) -> pl.Series:
    x = x / x.max() * MAX_FIELDSTRENGTH
    return x


def import_data(data_path):
    data = (
        pl.scan_csv(data_path, comment_prefix="#", has_header=True)
        .sort("cond.real", "freq")
        .with_columns(
            ccol("Ey.real", "Ey.imag").alias("Ey[c]", fields="Ey"),
            ccol("E0y.real", "E0y.imag").alias("E0y[c]", fields="E0y"),
        )
        .select(
            pl.col("freq"),
            pl.col("cond.real").cast(pl.Float64).alias("cond"),
            pl.col("Ey[c]").complex.arg().alias("Ey.pha"),
            pl.col("Ey[c]").complex.modulus().alias("Ey.mag"),
            pl.col("E0y[c]").complex.arg().alias("E0y.pha"),
            pl.col("E0y[c]").complex.modulus().alias("E0y.mag"),
        )
        .collect()
    )

    return data


def window(t):
    t1, t2 = -2, 3
    left = np.min([np.ones(len(t)), np.exp((t - t1) / 0.5)], axis=0)
    right = np.min([np.ones(len(t)), np.exp(-(t - t2) / 2)], axis=0)
    return left + right - 1


def calculate_ifft(data):
    time_data = []
    for (cond,), data_subset in data.group_by(["cond"], maintain_order=True):
        data_ifft = {var: ifft(data_subset, var).real for var in ["Ey", "E0y"]}

        M = (len(data_subset["freq"]) - 1) * 2
        time = np.arange(M) / (M * data_subset["freq"][[0, 1]].diff(1, "drop").item())

        data_ifft = (
            pl.DataFrame(data_ifft)
            .with_columns(pl.lit(time).alias("time"))
            .select("time", "Ey", "E0y")
            .with_columns(pl.lit(cond).alias("cond"))
        )
        time_data.append(data_ifft)

    time_data = pl.concat(time_data)
    # time_data = time_data.with_columns(
    #     (pl.col("E0y") / pl.col("E0y").abs().max() * MAX_FIELDSTRENGTH).alias("E0y"),
    #     (pl.col("Ey") / pl.col("E0y").abs().max() * MAX_FIELDSTRENGTH).alias("Ey"),
    # )
    return time_data


if __name__ == "__main__":
    data = import_data(PROJECT_ROOT / "data/04.09.23/spectral_data.csv")
    data = Dataset(data, index="freq", id_vars=["cond"]).sort_columns()
    # DATAFILES["spectra"].write(data)

    time_data = calculate_ifft(data)
    time_data = Dataset(time_data, index="time", id_vars=["cond"]).sort_columns()
    # DATAFILES["temporal"].write(time_data)

    pnpa_data = PNPA_DATAFILES["spectrum"].load()
    pnpa_data = pnpa_data.with_columns(ccol("value*").modulus().alias("mag"))
    pnpa_data = (
        pl.read_csv(
            PROJECT_PATHS.raw_data
            / "12.01.23/14h09m31s_Field dependence dry air_292.05-292.23_HWP=50.00_Sam=0.00.txt",
            separator="\t",
            comment_prefix="#",
            has_header=False,
            columns=range(2),
            new_columns=("time", "value"),
        )
        .with_columns(
            (6.67 * (178.45 - pl.col("time"))).alias("time"),
            pl.col("value").map_batches(scale_fieldstrength).alias("value"),
        )
        .with_columns(pl.col("value") * pl.col("time").map_batches(window))
        .sort("time")
    )
    pnpa_data = Dataset(pnpa_data, index="time")
    pnpa_data = pnpa_data.regrid(
        pl.Series("time", np.arange(-50, 50, 1e-3)), fill_value=0, method="catmullrom"
    )

    pnpa_fft = Dataset(pl_fft(pnpa_data, "time"), "freq")
    pnpa_fft = pnpa_fft.regrid(data.coord("freq")).drop_nulls()
    pnpa_fft = pnpa_fft.with_columns(
        ccol("value*").modulus().alias("mag")
    ).with_columns((pl.col("mag") / pl.col("mag").pow(2).mean().sqrt()).alias("mag"))

    norm = pl.col("E0y.mag").pow(2).mean().sqrt()
    data = data.with_columns(
        (pl.col("Ey.mag") / norm).alias("Ey.mag"),
        (pl.col("E0y.mag") / norm).alias("E0y.mag"),
    )
