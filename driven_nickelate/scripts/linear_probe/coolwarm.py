import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

project_dir = "/Users/mmatthiesen/surfdrive/Measurements/2023/eSRR NNO"
sys.path.insert(0, project_dir)
from support_functions import (
    c_const,
    get_slice,
    nearest_index,
    read_kwargs,
    time_window,
)

os.chdir(project_dir)

x0 = 9.86  # Time zero
t_c = np.arange(-10, 50, 3e-3)
time_lim = get_slice(t_c, -2, 25)
window = time_window(t_c, -2, 15)

f_c = np.fft.rfftfreq(t_c.size, t_c[1] - t_c[0])
freq_lim = get_slice(f_c, 0.1, 2.5)

# Import file paths
files = pd.read_csv("File lists/linear_probe/weakfield_ZnTe_tempdep.dat", comment="#")

# Import temperature-interpolated reference data
ref_data = pd.read_csv("Processed data/linear_probe/xT_ZnTe_LSAT_interp.dat")
ref_tempr = ref_data.columns[1:].to_numpy(dtype="float")
t_ref = ref_data["Delay"]

# Initialize empty data arrays
N_time, N_freq = len(t_c[time_lim]), len(f_c[freq_lim])
waveforms = np.zeros((len(files), N_time, 4), dtype="float")
spectra = np.zeros((len(files), N_freq, 4), dtype="float")

# File empty data arrays with columns from datafiles in `files'
pbar = tqdm(range(2 + len(files["Path"])), desc="Processing files")
for i in range(len(files)):
    pbar.set_description("Processing %s" % files["Path"].iloc[i])

    df = pd.read_csv(files["Path"].iloc[i], **read_kwargs)
    a, b = (2 / c_const) * (x0 - df["Delay"])[::-1], df["X"][::-1] - df["X"].mean()
    E = np.interp(t_c, a, b, left=0, right=0)
    E *= window

    temperature = files["Get temperature"].iloc[i]
    direction = files["Direction"].iloc[i]

    X_ref = ref_data[f"{ref_tempr[nearest_index(ref_tempr, temperature)]:.0f}"]
    a, b = (2 / c_const) * (x0 - t_ref)[::-1], X_ref[::-1] - X_ref.mean()
    E_ref = np.interp(t_c, a, b, left=0, right=0)
    E_ref *= window

    S = np.abs(np.fft.rfft(E)) / np.abs(np.fft.rfft(E_ref))

    waveforms[i, :, 0] = np.repeat(direction, len(t_c[time_lim]))
    waveforms[i, :, 1] = np.repeat(temperature, len(t_c[time_lim]))
    waveforms[i, :, 2] = t_c[time_lim]
    waveforms[i, :, 3] = E[time_lim]

    spectra[i, :, 0] = np.repeat(direction, len(f_c[freq_lim]))
    spectra[i, :, 1] = np.repeat(temperature, len(f_c[freq_lim]))
    spectra[i, :, 2] = f_c[freq_lim]
    spectra[i, :, 3] = S[freq_lim]

    pbar.update(1)


# Create DataFrame from data
pbar.set_description("Creating DataFrames")


def split(ndarr):
    return np.transpose([np.ravel(arr) for arr in np.split(ndarr, 4, axis=2)])


df_time = pd.DataFrame(
    split(waveforms), columns=["Direction", "Temperature", "t", "EOS"]
)
df_freq = pd.DataFrame(
    split(spectra), columns=["Direction", "Temperature", "f", "FFT EOS"]
)

pbar.set_description("Saving time data")
df_time.to_csv("Processed data/linear_probe/xT_ZnTe_coolwarm.dat", index=False)
pbar.update(1)

pbar.set_description("Saving frequency data")
df_freq.to_csv("Processed data/linear_probe/xT_FFT-ZnTe_coolwarm.dat", index=False)
pbar.update(1)

pbar.close()
