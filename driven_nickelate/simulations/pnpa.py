import numpy as np
import polars as pl
from polars_dataset import Datafile, Dataset

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.tools import pl_fft

PROJECT_ROOT = PROJECT_PATHS.root / "simulations"
MAX_FIELDSTRENGTH = 0.192 * 1e8  # [V/m]

STORE_DIR = PROJECT_PATHS.processed_data / "simulations/pnpa"
STORE_DIR.mkdir(parents=True, exist_ok=True)

DATAFILES = {
    "spectrum": Datafile(path=STORE_DIR / "pnpa_spectrum.csv", index="freq"),
    "waveform": Datafile(path=STORE_DIR / "pnpa_waveform.csv", index="time"),
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


def time_window(t):
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
    return time_data


if __name__ == "__main__":
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
        .with_columns(pl.col("value") * pl.col("time").map_batches(time_window))
        .sort("time")
    )
    pnpa_data = Dataset(pnpa_data, index="time")
    pnpa_data = pnpa_data.regrid(
        pl.Series("time", np.arange(-5, 15, 6e-2)), fill_value=0, method="catmullrom"
    )
    DATAFILES["waveform"].write(pnpa_data)

    pnpa_fft = Dataset(pl_fft(pnpa_data, "time", rfft=True), "freq")
    DATAFILES["spectrum"].write(pnpa_fft)

    num_samples = (len(pnpa_fft.coord("freq")) - 1) * 2
    time = np.arange(num_samples) / (
        num_samples * pnpa_fft.coord("freq")[[0, 1]].diff(1, "drop").item()
    )

    signal = pnpa_fft.fetch("value.real", "value.imag").to_numpy()
    signal = signal[:, 0] + 1j * signal[:, 1]
    signal = np.fft.irfft(signal)
