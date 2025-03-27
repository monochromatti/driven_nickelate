import os

import numpy as np
import polars as pl


# Get conductivity from filenames. (e.g. smatrix_3.90E+05.csv -> 3.90E+05)
def get_num_from_filename(filename):
    num = filename.split("_")[1].split(".csv")[0]
    return float(num)


def struct_norm(x):
    return pl.Expr.sqrt(x.struct[0].pow(2) + x.struct[1].pow(2))


def struct_phase(x):
    return pl.Series(np.unwrap(np.arctan2(x.struct[0], x.struct[1])))


# %%

data = pl.concat(
    pl.read_csv("data/26.05.23/gap_conductivity/" + file, comment_char="#")
    .rename({"freq (THz)": "freq"})
    .with_columns(
        pl.lit(get_num_from_filename(file)).cast(pl.Float64).alias("cond"),
        pl.struct("s11.real", "s11.imag").map(struct_norm).alias("s11.mag"),
        pl.struct("s11.real", "s11.imag").map(struct_phase).alias("s11.pha"),
        pl.struct("s21.real", "s21.imag").map(struct_norm).alias("s21.mag"),
        pl.struct("s21.real", "s21.imag").map(struct_phase).alias("s21.pha"),
    )
    for file in filter(
        lambda x: x.endswith(".csv"), os.listdir("data/26.05.23/gap_conductivity")
    )
).sort("cond", "freq")

data = data.unpivot(index=("cond", "freq"), variable_name="param", value_name="value")
with open("processed_data/gap_conductivity.csv", "wb") as f:
    f.write("# Flat substrate index of 4.8 without absorption.\n".encode("utf-8"))
    f.write("# Zero conductivity outside rectangular gap.\n".encode("utf-8"))
    data.write_csv(f)

# %%
data = pl.concat(
    pl.read_csv("data/26.05.23/film_conductivity/" + file, comment_char="#")
    .rename({"freq (THz)": "freq"})
    .with_columns(
        pl.lit(get_num_from_filename(file)).cast(pl.Float64).alias("cond"),
        pl.struct("s11.real", "s11.imag").map(struct_norm).alias("s11.mag"),
        pl.struct("s11.real", "s11.imag").map(struct_phase).alias("s11.pha"),
        pl.struct("s21.real", "s21.imag").map(struct_norm).alias("s21.mag"),
        pl.struct("s21.real", "s21.imag").map(struct_phase).alias("s21.pha"),
    )
    for file in filter(
        lambda x: x.endswith(".csv"), os.listdir("data/26.05.23/film_conductivity")
    )
).sort("cond", "freq")

data = data.unpivot(index=("cond", "freq"), variable_name="param", value_name="value")
with open("processed_data/film_conductivity.csv", "wb") as f:
    f.write("# Flat substrate index of 4.8 without absorption.\n".encode("utf-8"))
    f.write("# Homogeneous conductivity across sample.\n".encode("utf-8"))
    data.write_csv(f)
