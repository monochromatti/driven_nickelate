import logging
import re

import polars as pl
from polars_complex import ccol
from polars_dataset import Datafile

from driven_nickelate.config import paths as PROJECT_PATHS

STORE_DIR = PROJECT_PATHS.root / "processed_data/simulations/field_enhancement"
STORE_DIR.mkdir(parents=True, exist_ok=True)
DATAFILES = {
    "surface_data": Datafile(path=STORE_DIR / "surface_data.csv"),
    "field_enhancement": Datafile(path=STORE_DIR / "field_enhancement.csv"),
}


def read_surface_data(filename):
    with open(filename) as f:
        header = ""
        data_lines = []
        for line in f:
            if line.startswith("%"):
                header = line
            else:
                values = line.strip("\n").replace("i", "j").split(",")
                values[:2] = [float(v) for v in values[:2]]
                values[2:] = [
                    x for v in map(complex, values[2:]) for x in (v.real, v.imag)
                ]
                data_lines.append(values)

    file_colnames = header.strip("% ").split(",")
    schema = {"x": pl.Float64, "y": pl.Float64}
    for colname in file_colnames[2:]:
        colname = re.sub(r" @.*", "", colname.strip("emw."))
        varname, unit = re.search(r"([^ ]+) \(([^)]+)\)", colname).groups()
        schema[f"{varname}.real ({unit})"] = pl.Float64
        schema[f"{varname}.imag ({unit})"] = pl.Float64

    df = pl.DataFrame(data_lines, schema=schema)

    return df.unpivot(index=["x", "y"])


def load_data():
    data_folder = PROJECT_PATHS.root / "simulations/data/22.08.23"
    data_files = {
        "TBC_ON": data_folder / "resonatorON/spatial_data.csv",
        "TBC_OFF": data_folder / "resonatorOFF/spatial_data.csv",
    }
    data = []
    for tbc_status, f in data_files.items():
        data.append(
            read_surface_data(f)
            .sort("x", "y", "variable")
            .filter(pl.col("variable").str.contains(r"Ey.* \(V/m\)"))
            .pivot(
                index=["x", "y"],
                columns="variable",
                values="value",
                aggregate_function=None,
            )
            .with_columns(
                ccol("Ey.real (V/m)", "Ey.imag (V/m)").modulus().alias("E (V/m)"),
                pl.lit(tbc_status).replace({"TBC_ON": 1, "TBC_OFF": 0}).alias("TBC"),
            )
        )
    return pl.concat(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("Loading data ...")
    surface_data = load_data()
    DATAFILES["surface_data"].write(surface_data)

    logging.info("Calculating field enhancement ...")
    field_enhancement = surface_data.pivot(
        index=("x", "y"), values=["E (V/m)"], columns="TBC"
    ).select(pl.col("x", "y"), (pl.col("1") / pl.col("0")).alias("field_enhancement"))
    DATAFILES["field_enhancement"].write(field_enhancement)

    logging.info("Done.")
