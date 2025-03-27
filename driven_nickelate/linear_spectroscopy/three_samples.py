import logging

import numpy as np
import polars as pl
import polars.selectors as cs
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.tools import create_dataset, pl_fft

STORE_DIR = PROJECT_PATHS.root / "processed_data/linear_spectroscopy/three_samples"
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
        id_vars=["direction", "temperature", "sample", "sample_ref"],
    ),
    "film_conductivity": Datafile(
        path=STORE_DIR / "film_conductivity.csv",
        index="freq",
        id_vars=["temperature", "direction", "sample", "sample_ref"],
    ),
}


def process_waveforms() -> Dataset:
    column_names = ["delay"] + [
        c for s in ("", ".1") for c in [f"X{s}", f"X{s} SEM", f"Y{s}", f"Y{s} SEM"]
    ]

    esrr_files = pl.read_csv(
        PROJECT_PATHS.file_lists / "linear_probe/xT_NNO_eSRR.csv",
        comment_prefix="#",
        schema={"Set temperature": pl.Float64, "Direction": pl.Int8, "Path": pl.Utf8},
        new_columns=["temperature", "direction", "path"],
    ).with_columns(
        col("path").map_elements(
            lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8
        )
    )

    esrr_data = create_dataset(
        esrr_files,
        column_names,
        "delay",
        {"X": ("X", "Y"), "X.sem": ("X SEM", "Y SEM")},
    ).with_columns(pl.lit("esrr").alias("sample"))
    esrr_data.id_vars += ["sample"]

    film_files = pl.read_csv(
        PROJECT_PATHS.file_lists / "linear_probe/xT_NNO.csv",
        comment_prefix="#",
        schema={"Set temperature": pl.Float64, "Direction": pl.Int8, "Path": pl.Utf8},
        new_columns=["temperature", "direction", "path"],
    ).with_columns(
        col("path").map_elements(
            lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8
        )
    )

    film_data = create_dataset(
        film_files,
        column_names,
        "delay",
        {"X": ("X", "Y"), "X.sem": ("X SEM", "Y SEM")},
    ).with_columns(pl.lit("film").alias("sample"))
    film_data.id_vars += ["sample"]

    lsat_files = pl.read_csv(
        PROJECT_PATHS.file_lists / "linear_probe/xT_LSAT.csv",
        comment_prefix="#",
        schema={"Set temperature": pl.Float64, "Path": pl.Utf8},
        new_columns=["temperature", "path"],
    ).with_columns(
        col("path").map_elements(
            lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8
        )
    )
    lsat_data = create_dataset(
        lsat_files,
        column_names,
        "delay",
        {"X": ("X", "Y"), "X.sem": ("X SEM", "Y SEM")},
    ).with_columns(
        pl.lit(1).cast(pl.Int8).alias("direction"), pl.lit("lsat").alias("sample")
    )
    lsat_data.id_vars += ["direction", "sample"]

    lsat_esrr_files = pl.read_csv(
        PROJECT_PATHS.file_lists / "linear_probe/xT_LSAT_eSRR.csv",
        comment_prefix="#",
        schema={"Set temperature": pl.Float64, "Path": pl.Utf8},
        new_columns=["temperature", "path"],
    ).with_columns(
        col("path").map_elements(
            lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8
        )
    )
    lsat_esrr_data = create_dataset(
        lsat_esrr_files,
        column_names,
        "delay",
        {"X": ("X", "Y"), "X.sem": ("X SEM", "Y SEM")},
    ).with_columns(
        pl.lit(1).cast(pl.Int8).alias("direction"), pl.lit("lsat_esrr").alias("sample")
    )
    lsat_esrr_data.id_vars += ["direction", "sample"]

    waveforms = Dataset(
        [esrr_data, film_data, lsat_data, lsat_esrr_data], index="delay"
    )

    waveforms = (
        waveforms.with_columns(
            (6.67 * (10.4 - col("delay"))).alias("delay"),
            (col("X") - col("X").mean()).over("sample").alias("X"),
        )
        .rename({"delay": "time"})
        .regrid(pl.Series("time", np.arange(-10, 50, 6e-3)), method="catmullrom")
        .with_columns(
            col("X").fill_null(0),
            col("X.sem").fill_null(col("X.sem").mean().over("sample")),
        )
        .sort()
    )

    temperatures = pl.Series("temperature", np.arange(5, 290, 5))
    waveforms = waveforms.regrid(temperatures, method="cosine").drop_nulls()

    return waveforms


def process_fourier(waveforms) -> Dataset:
    spectra = pl_fft(waveforms, "time", waveforms.id_vars).filter(col("freq") < 4)
    spectra = Dataset(spectra, "freq", waveforms.id_vars).select_data(
        ccol("X.real", "X.imag").alias("X[c]", fields="X"),
        ccol("X.sem.real", "X.sem.imag").alias("X.sem[c]", fields="X.sem"),
    )
    spec_ref = (
        spectra.filter(col("sample") == "lsat")
        .unique(["temperature", "freq"])
        .drop("direction")
    )
    spectra = spectra.join(
        spec_ref,
        on=("freq", "temperature"),
        suffix="_ref",
    ).select_data(
        col("sample_ref", "X[c]", "X.sem[c]"),
        (ccol("X[c]") / ccol("X[c]_ref"))
        .over("temperature", "direction")
        .alias("t[c]", fields="t")
        .struct.rename_fields(("t.real", "t.imag")),
        (ccol("X.sem[c]") / ccol("X.sem[c]_ref"))
        .over("temperature", "direction")
        .alias("t.sem[c]", fields="t.sem")
        .struct.rename_fields(("t.sem.real", "t.sem.imag")),
    )
    spectra.id_vars += ["sample_ref"]

    return spectra.sort_columns()


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
        col("freq"), ccol("n_sub.real", "n_sub.imag").alias("n_sub[c]", fields="n_sub")
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
    k0 = (2 * np.pi * col("freq")) * x / 0.3  # ⍵(Δd)/c
    spectra = spectra.with_columns(
        pl.struct(k0.alias("real"), pl.lit(0).alias("imag")).alias(
            "k0[c]", fields="k0"
        ),
    ).select_data(
        (ccol("n_sub[c]") * ccol("k0[c]")).alias(
            "prop_factor[c]", fields="prop_factor"
        ),
        col("n_sub[c]"),
        col("t[c]"),
    )
    print(spectra)
    spectra = spectra.select_data(
        (ccol("t[c]") * ccol("prop_factor[c]")).alias("t[c]", fields="t"),
        col("n_sub[c]"),
    )

    d = 12e-9
    sigma = ((ccol("n_sub[c]") + 1) / (d * 377)) * (ccol("t[c]").inverse() - 1)

    spectra = spectra.with_columns(sigma.alias("sigma[c]", fields="sigma"))
    return spectra


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Processing waveforms ...")
    waveforms = process_waveforms()
    DATAFILES["waveforms"].write(waveforms.filter(col("time").is_between(-5, 20)))

    logging.info("Transforming to spectra ...")
    spectra = process_fourier(waveforms)
    DATAFILES["spectra"].write(spectra.unnest(cs.contains("[c]")))

    logging.info("Calculating film conductivity ...")
    cond = append_index(
        spectra.select_data(col("t[c]")).filter(col("sample").eq("film"))
    )
    cond = calculate_conductivity(cond)
    DATAFILES["film_conductivity"].write(cond.unnest(cs.contains("[c]")))

    logging.info("Done.")
