import logging

import polars as pl
import polars.selectors as cs
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset

from driven_nickelate.config import paths as PROJECT_PATHS

STORE_DIR = PROJECT_PATHS.processed_data / "conductivity_mapping/calculation/"
STORE_DIR.mkdir(exist_ok=True, parents=True)
DATAFILES = {
    "spectra": Datafile(
        path=STORE_DIR / "spectra.csv", index="freq", id_vars=["cond_gap", "cond_film"]
    )
}


def import_calculation(data_file, reference_file):
    n_freq = 53
    columns = ["freq"] + [
        f"{name}.{comp}"
        for name in ["Ey_surf", "cond_gap", "cond_film", "Ey_sub"]
        for comp in ["real", "imag"]
    ]
    columns_ref = ["freq", "Ey_sub.real", "Ey_sub.imag"]
    data_ref = (
        pl.read_csv(
            reference_file,
            schema={k: pl.Float32 for k in columns_ref},
        )
        .select(
            col("freq"),
            col("Ey_sub.real").alias("E_ref.real"),
            col("Ey_sub.imag").alias("E_ref.imag"),
        )
        .with_columns(
            ccol("E_ref.real", "E_ref.imag").alias("E_ref[c]", fields="E_ref")
        )
    )
    if data_ref.select("freq").unique().height != n_freq:
        raise ValueError("Reference data has wrong length.")

    data = (
        pl.read_csv(data_file, schema={k: pl.Float32 for k in columns})
        .select(
            col("freq"),
            col("Ey_sub.real").alias("E.real"),
            col("Ey_sub.imag").alias("E.imag"),
            col("cond_gap.real").alias("cond_gap"),
            col("cond_film.real").alias("cond_film"),
        )
        .with_columns(col("freq").count().over("cond_gap", "cond_film").alias("n_freq"))
        .filter(col("n_freq").eq(n_freq))
        .with_columns(
            ccol("E.real", "E.imag").alias("E[c]", fields="E"),
        )
        .join(data_ref, on="freq")
        .select(
            col("cond_gap"),
            col("cond_film"),
            col("freq"),
            (ccol("E[c]") / ccol("E_ref[c]")).alias("t[c]", fields="t"),
        )
        .sort("cond_film", "cond_gap", "freq")
    )

    data = Dataset(
        data.unnest(cs.contains("[c]")),
        index="freq",
        id_vars=["cond_gap", "cond_film"],
    )
    return data


def load() -> Dataset:
    try:
        return DATAFILES["spectra"].load()
    except FileNotFoundError:
        logging.warning("Calculation data not found.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    files = [
        dict(
            CALCULATION_DATA=(
                PROJECT_PATHS.root
                / "simulations/data/06.11.23/2023-11-05_21h17m48s_spectral_data.csv"
            ),
            REFERENCE_DATA=PROJECT_PATHS.root
            / "simulations/data/06.11.23/spectral_data_reference.csv",
        ),
        dict(
            CALCULATION_DATA=(
                PROJECT_PATHS.root
                / "simulations/data/27.11.23/2023-11-26_19h28m23s_spectral_data.csv"
            ),
            REFERENCE_DATA=PROJECT_PATHS.root
            / "simulations/data/27.11.23/spectral_data_reference.csv",
        ),
    ]

    logging.info("Loading calculation data ...")
    frames = []
    for x in files:
        data_file, reference_file = x.values()
        df = import_calculation(data_file, reference_file)
        frames.append(import_calculation(data_file, reference_file))
    spectra_calc = Dataset(frames, index="freq").sort()
    DATAFILES["spectra"].write(spectra_calc)

    logging.info("Done.")
