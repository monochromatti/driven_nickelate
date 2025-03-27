import logging

import numpy as np
import polars as pl
import polars.selectors as cs
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.tools import create_dataset, pl_fft

STORE_DIR = PROJECT_PATHS.root / "processed_data/linear_spectroscopy/film"
STORE_DIR.mkdir(exist_ok=True, parents=True)

DATAFILES = {
    "waveforms": Datafile(
        path=STORE_DIR / "waveforms.csv",
        index="delay",
        id_vars=["direction", "temperature"],
    ),
    "spectra": Datafile(
        path=STORE_DIR / "spectra.csv",
        index="freq",
        id_vars=["direction", "temperature"],
    ),
}


def process_waveforms() -> Dataset:
    column_names = ["delay"] + [f"{var}{s}" for var in ["X", "Y"] for s in ("", " SEM")]
    import_kwargs = dict(
        column_names=column_names,
        index="delay",
        lockin_schema={"X": ("X", "Y")},
        separator="\t",
        comment_prefix="#",
        has_header=False,
    )
    time_coord = pl.Series("time", np.arange(-5, 50, 6e-3))

    waveforms = create_dataset(
        (
            pl.read_csv(
                PROJECT_PATHS.file_lists / "linear_probe/xT_ZnTe_NNO.csv",
                comment_prefix="#",
            ).select(
                col("Get temperature").alias("temperature"),
                col("Direction").alias("direction"),
                col("Path")
                .map_elements(
                    lambda x: f"{str(PROJECT_PATHS.root)}/{x}", return_dtype=pl.Utf8
                )
                .alias("path"),
            )
        ),
        id_schema={"temperature": pl.Float64, "direction": pl.Float64},
        **import_kwargs,
    )

    waveforms = (
        waveforms.with_columns(
            (6.67 * (9.8 - col("delay"))).alias("delay"),
            (col("X") - col("X").mean().over("temperature", "direction")).alias("X"),
        )
        .rename({"delay": "time"})
        .sort()
        .regrid(time_coord, method="catmullrom", fill_value=0)
    )
    waveforms_ref = (
        create_dataset(
            pl.read_csv(
                PROJECT_PATHS.file_lists / "linear_probe/xT_ZnTe_LSAT.csv",
                comment_prefix="#",
            ).select(
                col("Get temperature").alias("temperature"),
                col("Path")
                .map_elements(
                    lambda x: f"{str(PROJECT_PATHS.root)}/{x}", return_dtype=pl.Utf8
                )
                .alias("path"),
            ),
            id_schema={"temperature": pl.Float64},
            **import_kwargs,
        )
        .with_columns(
            (6.67 * (9.8 - col("delay"))).alias("delay"),
            col("X") - col("X").mean().over("temperature"),
        )
        .rename({"delay": "time"})
        .sort()
        .regrid(
            pl.Series("time", np.arange(-1, 10, 1e-2)),
            method="catmullrom",
            fill_value=0,
        )
    )

    temperatures = waveforms_ref.coord("temperature")
    temperatures = pl.Series(
        "temperature", np.arange(temperatures.min(), temperatures.max(), 5)
    )
    waveforms_ref = waveforms_ref.regrid(temperatures, method="cosine")
    waveforms_ref = waveforms_ref.regrid(time_coord, method="catmullrom", fill_value=0)
    waveforms_ref = waveforms_ref.vstack(
        waveforms_ref.filter(col("temperature").eq(temperatures.min()))
        .with_columns(pl.lit(0.0).alias("temperature"))
        .df
    ).sort()

    frames = []
    for (direction,), df in waveforms.group_by(["direction"]):
        df = Dataset(df, index="time", id_vars=("temperature", "direction")).sort(
            "temperature", "direction", "time"
        )
        T_min, T_max = df.extrema("temperature")
        temperatures = pl.Series("temperature", np.arange(T_min, T_max + 1, 5))

        frames.append(
            df.regrid(temperatures, method="cosine")
            .join(
                waveforms_ref.regrid(temperatures, method="cosine").sort(
                    "temperature", "time"
                ),
                on=("temperature", "time"),
                suffix="_ref",
            )
            .select("temperature", "direction", "time", "X", "X_ref")
            .sort("temperature", "direction", "time")
        )

    waveforms = Dataset(frames, waveforms.index, waveforms.id_vars).drop_nulls().sort()
    return waveforms


def process_fourier(waveforms: Dataset) -> Dataset:
    spec = pl_fft(waveforms, waveforms.index, waveforms.id_vars)
    spec = Dataset(spec, index="freq", id_vars=waveforms.id_vars)

    spec = (
        spec.select_data(
            ccol("X.real", "X.imag").alias("X[c]", fields="X"),
            ccol("X_ref.real", "X_ref.imag").alias("X_ref[c]", fields="X_ref"),
        )
        .select_data(
            (ccol("X[c]") / ccol("X_ref[c]")).alias("t[c]", fields="t"),
        )
        .filter(col("freq").is_between(0.3, 2.5))
        .with_columns(
            ccol("t[c]").modulus().alias("t.mag"),
            ccol("t[c]").arg().alias("t.pha"),
        )
        # .with_columns(
        #     ccol("t[c]").arg(unwrap=True).over("temperature", "direction"),
        # )
    )
    return spec


def append_index(spectra: Dataset) -> Dataset:
    # LSAT substrate data from doi.org/10.1116/1.4960356
    eps_dict = {
        "L1": (6.3, 156.9, 12.8),
        "L2": (1.5, 222, 35),
        "L3": (2.6, 248, 42),
        "L4": (4.3, 285.9, 28),
        "L5": (0.46, 330, 46),
        "L6": (1.89, 395, 44),
        "L7": (0.51, 436.4, 18.6),
        "L8": (0.646, 659.8, 36.5),
        "L9": (0.0045, 787, 26),
    }
    freq = spectra.coord("freq").to_numpy()
    eps = 4.0  # eps_inf
    for key, val in eps_dict.items():
        a = eps_dict[key][0]
        f0 = eps_dict[key][1] / 33.356
        g = eps_dict[key][2] / 33.356
        eps += a * f0**2 / (f0**2 - freq**2 - 1j * g * freq)

    index_substrate = pl.DataFrame(
        {"freq": freq, "n_sub.real": np.sqrt(eps).real, "n_sub.imag": np.sqrt(eps).imag}
    ).select(
        col("freq"),
        ccol("n_sub.real", "n_sub.imag").alias("n_sub[c]", fields="n_sub"),
    )
    return spectra.join(index_substrate, on="freq")


def calculate_conductivity(
    spectra: Dataset,
    d: float = 12e-9,
    x: float = -0.023,
) -> Dataset:
    """
    Calculate the complex conductivity of the film using the substrate as a reference.

    Parameters
    ----------
    spectra : Dataset
        The complex transmission spectra of the film. Must include a column `n_sub[c]`
        for the substrate index.
    d : float, optional
        The thickness of the film.
    x : float, optional
        The thickness mismatch between the film and the substrate.

    """
    φ = (2 * np.pi * col("freq")) * col("n_sub[c]") * x / 0.3  # ⍵n(Δd)/c
    prop_factor = φ.complex.real()
    decay_factor = (-1 * φ.complex.imag()).exp()  # exp(-Im[φ])
    spectra = (
        spectra.with_columns(
            (prop_factor.cos() * decay_factor).alias("prop_factor.real"),
            (prop_factor.sin() * decay_factor).alias("prop_factor.imag"),
        )
        .with_columns(col("prop_factor").complex.into())
        .with_columns((ccol("t[c]") * ccol("prop_factor[c]")).alias("t[c]", fields="t"))
    )

    d = 11e-9
    sigma = ((col("n_sub[c]") + 1) / (d * 377)).complex.multiply(
        (col("t[c]").complex.inverse() - 1)
    )
    spectra = spectra.with_columns(sigma.alias("sigma[c]", fields="sigma"))
    return spectra


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Processing waveforms ...")
    waveforms = process_waveforms()
    DATAFILES["waveforms"].write(waveforms.filter(col("time").is_between(-5, 20)))

    logging.info("Processing Fourier spectra ...")
    spec = process_fourier(waveforms)

    logging.info("Appending substrate index to data ...")
    spec = append_index(spec)

    # logging.info("Calculating conductivity ...")
    # spec = calculate_conductivity()

    DATAFILES["spectra"].write(spec.filter(col("freq") < 4).unnest(cs.contains("[c]")))

    logging.info("Done.")


waveforms

pl.DataFrame(waveforms.filter(col("time") < 15))
