# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Parameters from Appl. Opt. 24, 4493-4499 (1985)
wt = 2.15e2  # [1/cm]
wp = 7.28e4  # [1/cm]

w = np.arange(1e-2, 333.6, 1e-2)
f = w / (33.36 * 1e-12)

eps_inf = 1
eps = eps_inf - wp**2 / (w**2 + 1j * w * wt)

eps0 = 8.854e-12  # F/m

sig = (2j * np.pi * f * eps0) * (eps_inf - eps)

df = pl.DataFrame(
    {
        "freq": f,
        "eps.real": eps.real,
        "eps.imag": eps.imag,
        "sig.real": sig.real,
        "sig.imag": sig.imag,
    }
)
df.write_csv("Au_response_functions.csv")

fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.4), layout="constrained")
ax[0].plot(1e-12 * f, -eps.real, label="-eps.real")
ax[0].plot(1e-12 * f, eps.imag, label="eps.imag")
ax[0].set(xscale="log", yscale="log")
ax[0].legend()

ax[1].plot(1e-12 * f, sig.real, label="sig.real")
ax[1].plot(1e-12 * f, sig.imag, label="sig.imag")
ax[1].set(xscale="log", yscale="log")
ax[1].legend()

ax[0].set(xlabel=r"$f$ (THz)", ylabel=r"$\epsilon$")
ax[1].set(xlabel=r"$f$ (THz)", ylabel=r"$\sigma$ (S/m)")

fig.savefig("eps_sigma.png")
