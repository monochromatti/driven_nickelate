import numpy as np
import polars as pl

from driven_nickelate.config import paths as PROJECT_PATHS

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
freq = np.arange(0, 20, 1e-3)  # THz
eps = 4.0  # eps_inf
for key, val in eps_dict.items():
    a = eps_dict[key][0]
    f0 = eps_dict[key][1] / 33.356
    g = eps_dict[key][2] / 33.356
    eps += a * f0**2 / (f0**2 - freq**2 - 1j * g * freq)

pl.DataFrame({"f (THz)": freq, "eps.real": eps.real, "eps.imag": eps.imag}).write_csv(
    PROJECT_PATHS.root / "literature_data/lsat_model.csv"
)
