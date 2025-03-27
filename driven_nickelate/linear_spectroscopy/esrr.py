import logging

import hvplot.polars

# +
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.linear_spectroscopy.lsat import DATAFILES as LSAT_DATAFILES
from driven_nickelate.tools import create_dataset, pl_fft

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

hvplot.extension("matplotlib")

# +
STORE_DIR = PROJECT_PATHS.root / "processed_data/linear_spectroscopy/esrr"
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


# -


def process_waveforms() -> Dataset:
    time = pl.Series("time", np.arange(-5, 30, 6e-3))

    paths = pl.read_csv(
        PROJECT_PATHS.file_lists / "linear_probe/xT_weakfield_ZnTe.csv",
        comment_prefix="#",
    ).select(
        col("Get temperature").alias("temperature"),
        col("Direction").cast(pl.Int8).alias("direction"),
        col("Path")
        .map_elements(lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8)
        .alias("path"),
    )
    column_names = ["delay"] + [
        c for s in ("", ".1") for c in [f"X{s}", f"X{s} SEM", f"Y{s}", f"Y{s} SEM"]
    ]
    lockin_schema = {"X": ("X", "Y"), "X.sem": ("X SEM", "Y SEM")}

    waveforms = (
        create_dataset(paths, column_names, index="delay", lockin_schema=lockin_schema)
        .with_columns(
            (6.67 * (9.86 - col("delay"))).alias("delay"),
            (col("X") - col("X").mean()).alias("X"),
        )
        .rename({"delay": "time"})
    ).regrid(time, method="catmullrom", fill_value=0)

    lsat_data = (
        LSAT_DATAFILES["waveforms"]
        .load()
        .filter(
            (pl.col("detector") == "ZnTe")
            & (
                pl.col("date")
                .cast(pl.Date)
                .is_between(pl.date(2023, 1, 23), pl.date(2023, 1, 24))
            )
        )
        .unique(["temperature", "date", "detector", "delay"])
        .with_columns(
            (6.67 * (9.86 - col("delay"))).alias("delay"),
            (col("X") - col("X").mean()).alias("X"),
        )
        .rename({"delay": "time"})
        .regrid(time, method="cosine", fill_value=0)
    )

    temperatures = waveforms.coord("temperature").sort()
    lsat_frames = []
    for (time,), group in lsat_data.group_by(["time"]):
        T, X = group.select("temperature", "X").get_columns()

        p = np.polyfit(T, X, 4)
        X_fit = np.polyval(p, temperatures)

        lsat_frames.append(
            pl.DataFrame({"time": time, "temperature": temperatures, "X": X_fit})
        )
    lsat_data = pl.concat(lsat_frames).sort("time", "temperature")
    waveforms = waveforms.join(
        lsat_data.rename({"X": "X_sub"}), on=["time", "temperature"]
    )
    return waveforms


logging.info("Processing waveforms ...")
waveforms = process_waveforms()
DATAFILES["waveforms"].write(waveforms.filter(col("time").is_between(-5, 20)))


logging.info("Processing spectra ...")
spectra = pl_fft(waveforms, "time", waveforms.id_vars).filter(
    col("freq").is_between(0.1, 4)
)
spectra = Dataset(spectra, "freq", waveforms.id_vars)
spectra = spectra.select_data(
    ccol("X.real", "X.imag").alias("X[c]", fields="X"),
    ccol("X_sub.real", "X_sub.imag").alias("X_sub[c]", fields="X_sub"),
)
spectra = spectra.with_columns(
    (ccol("X[c]") / ccol("X_sub[c]"))
    .over("temperature")
    .alias("t[c]", fields="t")
    .struct.rename_fields(("t.real", "t.imag"))
)
DATAFILES["spectra"].write(spectra.filter(col("freq") < 4).unnest(cs.contains("[c]")))

sns.relplot(data=waveforms, x="time", y="X", row="direction", hue="temperature")
