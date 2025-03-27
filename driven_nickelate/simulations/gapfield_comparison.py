import operator as oper
from functools import reduce

import numpy as np
import polars as pl

from driven_nickelate.config import paths as PROJECT_PATHS

SIMULATION_ROOT = PROJECT_PATHS.root / "simulations"

DATA_FILE = SIMULATION_ROOT / "data/06.09.23/spectral_data.csv"
MAX_FIELDSTRENGTH = 0.192  # [MV/cm], from laboratory calibration

# scriptname = Path(__file__).stem
# FIGUREPATH_TEMPORAL = f"figures/{scriptname}_temporal.png"
# FIGUREPATH_SPECTRAL = f"figures/{scriptname}_spectral.png"

# FILEPATH_TEMPORAL = f"processed_data/{scriptname}_temporal.csv"
# FILEPATH_SPECTRAL = f"processed_data/{scriptname}_spectral.csv"


def calculate_phase(x: pl.Struct) -> pl.Series:
    phase = np.arctan2(x.struct[0], x.struct[1])
    phase = np.unwrap(phase)
    return pl.Series(phase, dtype=pl.Float64)


def calculate_amplitude(x: pl.Struct) -> pl.Series:
    amplitude = np.linalg.norm((x.struct[0], x.struct[1]), axis=0)
    return pl.Series(amplitude, dtype=pl.Float64)


def ifft(data: pl.DataFrame, var: str) -> np.ndarray:
    data_subset = data.select("freq", f"{var}.mag", f"{var}.pha").to_numpy(
        structured=True
    )
    complex_amplitude = data_subset[f"{var}.mag"] * np.exp(
        -1j * data_subset[f"{var}.pha"]
    )

    t_shift = -14  # [ps]
    complex_amplitude *= np.exp(-1j * data_subset["freq"] * 2 * np.pi * t_shift)
    return np.fft.irfft(complex_amplitude, norm="backward")


def scale_fieldstrength(x: pl.Series) -> pl.DataFrame:
    x = x / pl.max(x) * MAX_FIELDSTRENGTH
    return x


def import_data():
    data = pl.scan_csv(
        DATA_FILE,
        comment_prefix="#",
        has_header=True,
    ).sort("cond.real", "freq")

    norm = data.select(
        pl.struct("E0y.real", "E0y.imag").map_batches(calculate_amplitude).max()
    ).collect()[0, 0]

    data = data.with_columns(
        (pl.col("E0y.real") / norm).alias("E0y.real"),
        (pl.col("E0y.imag") / norm).alias("E0y.imag"),
        (pl.col("Ey.real") / norm).alias("Ey.real"),
        (pl.col("Ey.imag") / norm).alias("Ey.imag"),
    )
    data = (
        data.with_columns(
            pl.struct("Ey.real", "Ey.imag")
            .map_batches(calculate_phase)
            .alias("Ey.pha"),
            pl.struct("Ey.real", "Ey.imag")
            .map_batches(calculate_amplitude)
            .alias("Ey.mag"),
            pl.struct("E0y.real", "E0y.imag")
            .map_batches(calculate_phase)
            .alias("E0y.pha"),
            pl.struct("E0y.real", "E0y.imag")
            .map_batches(calculate_amplitude)
            .alias("E0y.mag"),
        )
        .with_columns(
            (pl.col("Ey.mag") / pl.col("E0y.mag")).alias("FE"),
        )
        .collect()
    )

    return data


def calculate_ifft(data):
    time_data = []
    for (cond,), data_subset in data.group_by(["cond.real"], maintain_order=True):
        data_ifft = {var: ifft(data_subset, var).real for var in ["Ey", "E0y"]}

        M = (len(data_subset["freq"]) - 1) * 2
        time = np.arange(M) / (M * reduce(oper.sub, data_subset["freq"][[1, 0]]))

        data_ifft = (
            pl.DataFrame(data_ifft)
            .with_columns(pl.lit(time).alias("time"))
            .select("time", "Ey", "E0y")
            .with_columns(pl.lit(cond).alias("cond.real"))
        )
        time_data.append(data_ifft)

    time_data = pl.concat(time_data).with_columns(
        (pl.col("E0y") / pl.col("E0y").abs().max() * MAX_FIELDSTRENGTH).alias("E0y"),
        (pl.col("Ey") / pl.col("E0y").abs().max() * MAX_FIELDSTRENGTH).alias("Ey"),
    )
    return time_data


if __name__ == "__main__":
    data = import_data()
    # data.write_csv(FILEPATH_SPECTRAL)

    time_data = calculate_ifft(data)
    # time_data.write_csv(FILEPATH_TEMPORAL)
