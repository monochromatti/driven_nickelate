import numpy as np
import polars as pl
import polars.selectors as cs
from polars import col
from polars_complex import ccol
from polars_dataset import Datafile, Dataset

from driven_nickelate.conductivity_mapping.temperature import cond_from_temperatures
from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.tools import create_dataset, pl_fft

STORE_DIR = PROJECT_PATHS.root / "processed_data/photodoping/film"
STORE_DIR.mkdir(exist_ok=True, parents=True)

DATAFILES = {
    "waveforms": Datafile(
        path=STORE_DIR / "waveforms.csv",
        index="time",
        id_vars=["fluence"],
    ),
    "spectra": Datafile(
        path=STORE_DIR / "spectra.csv",
        index="freq",
        id_vars=["fluence"],
    ),
}

if __name__ == "__main__":

    def to_fluence(power):
        return 1e-3 * power / (np.pi * 0.23**2) * (1 - np.exp(-2 * 0.15**2 / 0.23**2))

    temp_I, temp_M = 4.40, 200  # [K]
    cond_I, cond_M = cond_from_temperatures([temp_I, temp_M])

    label = "NNO film transient probe scan xHWP pump 20ps"
    files = [
        f"12h20m54s_{label}_3.99-5.16_Del=144.30_HWP=15.13.txt",
        f"12h38m37s_{label}_3.98-5.19_Del=144.30_HWP=20.59.txt",
        f"12h56m21s_{label}_4.00-5.21_Del=144.30_HWP=24.77.txt",
        f"13h14m05s_{label}_4.01-5.24_Del=144.30_HWP=28.51.txt",
        f"13h31m49s_{label}_4.00-5.26_Del=144.30_HWP=32.21.txt",
        f"14h02m24s_{label}_4.01-5.28_Del=144.30_HWP=36.08.txt",
        f"14h26m47s_{label}_4.01-5.30_Del=144.30_HWP=38.92.txt",
    ]
    powers = [50, 80, 110, 140, 170, 200, 220]
    fluences = [to_fluence(power) for power in powers]
    paths = pl.DataFrame(
        [
            {
                "power": fluence,
                "path": str(PROJECT_PATHS.root / "raw_data/06.03.23/" / fn),
            }
            for fluence, fn in zip(fluences, files)
        ]
    )

    time = pl.Series("time", np.arange(-5, 20, 6e-3))
    waveforms = (
        create_dataset(
            paths,
            ["delay", "X", "X SEM", "Y", "Y SEM"],
            "delay",
            lockin_schema={"dX": ("X", "Y")},
            id_schema={"fluence": pl.Float64},
        )
        .with_columns(
            (6.67 * (9.77 - col("delay"))).alias("delay"),
            (col("dX") - col("dX").mean()).alias("dX"),
        )
        .rename({"delay": "time"})
        .regrid(time, method="catmullrom", fill_value=0)
    )

    paths = pl.DataFrame(
        {
            "path": PROJECT_PATHS.raw_data
            / "06.03.23/16h10m17s_Probe chopped 20ps film THz pump_4.01-5.17_HWP=63.50_Del=160.66.txt"
        }
    )
    waveform_ref = (
        create_dataset(
            paths,
            ["delay", "X", "X SEM", "Y", "Y SEM"],
            "delay",
            lockin_schema={"X": ("X", "Y")},
        )
        .with_columns(
            (6.67 * (9.77 - col("delay"))).alias("delay"),
            (col("X") - col("X").mean()).alias("X"),
        )
        .rename({"delay": "time"})
        .regrid(time, method="catmullrom", fill_value=0)
    )

    waveforms = waveforms.join(waveform_ref, on="time")
    DATAFILES["waveforms"].write(waveforms)

    spectra = pl_fft(waveforms, "time", waveforms.id_vars)
    spectra = Dataset(spectra, "freq", waveforms.id_vars).select_data(
        ccol("X.real", "X.imag").alias("X[c]", fields="X"),
        ccol("dX.real", "dX.imag").alias("dX[c]", fields="dX"),
    )

    t_reldiff = (ccol("dX[c]") / ccol("X[c]")).alias("t.reldiff[c]", fields="t.reldiff")
    delta_sigma = (
        pl.lit(2 / (377 * 12e-9)).complex.into()
        * (ccol("dX[c]") / (ccol("dX[c]") + ccol("X[c]")))
    ).alias("dsig[c]", fields="dsig")

    spectra = spectra.with_columns(t_reldiff, delta_sigma)

    DATAFILES["spectra"].write(spectra.unnest(cs.contains("[c]")))
