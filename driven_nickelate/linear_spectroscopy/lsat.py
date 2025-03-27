import logging
import re
from datetime import datetime

import polars as pl
from polars import col
from polars_dataset import Datafile, Dataset

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.tools import create_dataset

STORE_DIR = PROJECT_PATHS.root / "processed_data/linear_spectroscopy/lsat/"
STORE_DIR.mkdir(exist_ok=True, parents=True)

DATAFILES = {
    "waveforms": Datafile(
        path=STORE_DIR / "waveforms.csv",
        index="delay",
        id_vars=["detector", "date", "temperature"],
    )
}


def generate_column_names(num_averages: int = 0) -> list[str]:
    suffixes = [""] + [f".{i}" for i in range(1, num_averages + 1)]

    return ["delay"] + [
        n for s in suffixes for n in [f"X{s}", f"X{s} SEM", f"Y{s}", f"Y{s} SEM"]
    ]


def find_date(filename: str):
    match = re.search(r"\d{2}\.\d{2}\.\d{2}", filename)
    if match:
        value = match.group()
        return datetime.strptime(value, "%d.%m.%y").date()
    return None


def import_data(paths: pl.DataFrame, column_names: list):
    assert set(paths.columns) == set(["detector", "date", "temperature", "path"]), (
        "The `paths` argument does not have correct columns."
    )

    data = create_dataset(
        paths,
        column_names,
        index="delay",
        lockin_schema={"X": ("X", "Y"), "X.std": ("X SEM", "Y SEM")},
    ).with_columns((col("X") - col("X").mean()).alias("X"))

    return data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    data_total = []

    logging.info("Processing: Data using ZnTe detector. Measured 23-24 January 2023.")
    paths = pl.read_csv(
        PROJECT_PATHS.file_lists / "linear_probe/xT_ZnTe_LSAT.csv", comment_prefix="#"
    ).select(
        pl.lit("ZnTe").alias("detector"),
        pl.col("Path").map_elements(find_date, return_dtype=pl.Date).alias("date"),
        pl.col("Get temperature").alias("temperature"),
        pl.col("Path")
        .map_elements(lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8)
        .alias("path"),
    )

    column_names = generate_column_names(num_averages=1)
    data = import_data(paths, column_names)

    data_total.append(data)

    logging.info("Processing: Data using GaP detector. Measured 23 February 2023.")
    paths = pl.read_csv(
        PROJECT_PATHS.file_lists / "pump_probe/220223_xT_LSAT.csv", comment_prefix="#"
    ).select(
        pl.lit("GaP").alias("detector"),
        pl.col("Path").map_elements(find_date, return_dtype=pl.Date).alias("date"),
        pl.col("Temperature").alias("temperature"),
        pl.col("Path")
        .map_elements(lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8)
        .alias("path"),
    )
    data = import_data(paths, generate_column_names(num_averages=0))

    data_total.append(data)

    logging.info("Processing: Data using GaP detector. Measured 30-31 January 2023.")
    paths = pl.read_csv(
        PROJECT_PATHS.file_lists / "linear_probe/xT_LSAT.csv", comment_prefix="#"
    ).select(
        pl.lit("GaP").alias("detector"),
        pl.col("Path").map_elements(find_date, return_dtype=pl.Date).alias("date"),
        pl.col("Set temperature").alias("temperature"),
        pl.col("Path")
        .map_elements(lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8)
        .alias("path"),
    )
    data = import_data(paths, generate_column_names(num_averages=1))

    data_total.append(data)

    logging.info("Processing: Singular scan on 2 March 2023.")
    paths = pl.DataFrame(
        {
            "path": str(
                PROJECT_PATHS.raw_data
                / "02.03.23/15h02m06s_LSAT scan_292.80-292.99_.txt"
            )
        }
    ).select(
        pl.lit("GaP").alias("detector"),
        pl.col("path").map_elements(find_date, return_dtype=pl.Date).alias("date"),
        pl.lit(293.0).alias("temperature"),
        pl.col("path"),
    )

    data = import_data(paths, generate_column_names(num_averages=0))
    data_total.append(data)

    # Store data
    data_total = Dataset(data_total, index="delay")
    DATAFILES["waveforms"].write(data_total)

    logging.info("Done.")
