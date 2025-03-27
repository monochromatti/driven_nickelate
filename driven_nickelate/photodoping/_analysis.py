import numpy as np
import polars as pl
from polars import col
from scipy.constants import c as c_const

from driven_nickelate.config import paths as PROJECT_PATHS
from driven_nickelate.tools import create_dataset, pl_fft

files = pl.read_csv(
    PROJECT_PATHS.file_lists / "nir_pump" / "xPP_ProbeScan_PumpNIR.csv",
    comment_prefix="#",
).with_columns(
    col("path").map_elements(
        lambda s: str(PROJECT_PATHS.root / s), return_dtype=pl.Utf8
    )
)

waveforms = (
    create_dataset(
        files, ["delay", "X"], "delay", {"X": "X"}, id_schema={"pp_delay": pl.Float64}
    )
    .with_columns(
        ((2e9 / c_const) * (147.4 - col("pp_delay"))).alias("pp_delay"),
        ((2e9 / c_const) * (9.53 - col("delay"))).alias("time"),
        (col("X") - col("X").mean()).alias("signal"),
    )
    .set(index="time", id_vars=["pp_delay"])
    .select(col("pp_delay", "time", "signal"))
)

time = pl.Series("time", np.arange(-10, 30, 0.16))
spectra = pl_fft(waveforms.regrid(time, fill_value=0.0), "time", ["pp_delay"])
