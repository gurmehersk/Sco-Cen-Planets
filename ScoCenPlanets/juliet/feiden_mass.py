
import pickle as pkl
import numpy as np
from feiden_grid import FeidenGrid, best_fit_ages, best_fit_masses
 
MODELS_DIR = "./MagneticUpperSco/models"  # point at your clone
RSUN = 6.957e8
MSUN = 1.98847e30
TEFF_OBS, SIG_TEFF = 3168.4951, 55.0  # AstroARIADNE; SIG_TEFF ~ symmetrized (+36/-71)
 
# ---- your mass posterior, unchanged ----
with open("posterior_samples.pkl", "rb") as f:
    params = pkl.load(f)
 
density = params["rho"]
N = len(density)
 
Rstar_mean = 0.7375 * RSUN
Rstar_sigma = 0.024 * RSUN
Rstar = np.random.normal(Rstar_mean, Rstar_sigma, size=N)
 
 
def stellar_mass(radius, density):
    mass_kg = (4.0 / 3.0) * np.pi * radius**3 * density
    return mass_kg / MSUN
 
 
M_star = stellar_mass(Rstar, density)
p16, p50, p84 = np.percentile(M_star, [16, 50, 84])
print(f"M_star = {p50:.4f} (+{p84-p50:.4f}/-{p50-p16:.4f}) M_sun")
 
# ---- feed straight into the Feiden comparison ----
Rstar_Rsun = Rstar / RSUN  # tracks are tabulated in R/Rsun, convert back
sig_r = 0.024  # same sigma we used to draw Rstar, in Rsun

m_lo = np.percentile(M_star, 0.1) - 0.04
m_hi = np.percentile(M_star, 99.9) + 0.04
std_grid = FeidenGrid(MODELS_DIR, "std_tracks.txt", "std", m_lo, m_hi)
mag_grid = FeidenGrid(MODELS_DIR, "mag_tracks.txt", "mag", m_lo, m_hi)
 
'''
m_lo = 0.2229
m_hi = 0.2600
std_grid = FeidenGrid(MODELS_DIR, "std_tracks.txt", "std", m_lo, m_hi)
mag_grid = FeidenGrid(MODELS_DIR, "mag_tracks.txt", "mag", m_lo, m_hi)
'''
ages_std, chi_array = best_fit_ages(std_grid, M_star, Rstar_Rsun, TEFF_OBS, SIG_TEFF, sig_r)
ages_mag, chi_array_2 = best_fit_ages(mag_grid, M_star, Rstar_Rsun, TEFF_OBS, SIG_TEFF, sig_r)
 
for label, ages in [("standard", ages_std), ("magnetic (Feiden 2016)", ages_mag)]:
    ok = np.isfinite(ages)
    med, lo, hi = np.percentile(ages[ok], [50, 16, 84])
    print(f"Best-fit age, {label}: {med:.2f} +{hi-med:.2f} -{med-lo:.2f} Myr "
          f"({ok.sum()}/{len(ages)} draws in-grid)")

print(f"Standard chisquare median = {np.nanmedian(chi_array)}")
print(f"Magnetic chisquare median = {np.nanmedian(chi_array_2)}")
print(f"lowest magnetic chisquare = {np.nanmin(chi_array_2)}")
print("Compare to the lithium/EAGLES age: 18.2 +3.2/-15.7 Myr")

best_index = np.argmin(chi_array_2) # assuming chi_array_2 is the magnetic one
best_mass = M_star[best_index]
best_age = ages_mag[best_index]

print(f"The model's favorite mass is {best_mass:.4f} M_sun (Age: {best_age:.2f} Myr)")

import matplotlib.pyplot as plt

# Get the median mass from samples
m_med = np.percentile(M_star, 50)
#m_med = 0.2429
# Pull the evolutionary tracks for that exact median mass
age_mag, R_mag, Teff_mag = mag_grid.track_at_mass(m_med)
age_std, R_std, Teff_std = std_grid.track_at_mass(m_med)

plt.figure(figsize=(10, 6))

# 1. Plot observed data (with error bars)
plt.errorbar(TEFF_OBS, 0.7375, xerr=SIG_TEFF, yerr=0.024, 
             fmt='o', color='black', label='Your Observation', zorder=5)

# 2. Plot the Magnetic Track
age_mag_myr = age_mag * 1000
sc = plt.scatter(Teff_mag, R_mag, c=age_mag_myr, cmap='plasma', 
                 s=15, zorder=4, label=f'Magnetic Track (M={m_med:.3f} Msun)')
cbar = plt.colorbar(sc)
cbar.set_label('Age [Myr]')

# 3. Plot the Standard Track as a dashed line for comparison
plt.plot(Teff_std, R_std, '--', color='gray', alpha=0.7, 
         label=f'Standard Track (M={m_med:.3f} Msun)', zorder=3)


plt.gca().invert_xaxis()

plt.xlabel("Effective Temperature (K)")
plt.ylabel("Radius (R_sun)")
plt.title("Why is the Magnetic Age So Uncertain?")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
plt.savefig("feiden_visualization.pdf", bbox_inches = "tight")

import pandas as pd
import numpy as np

# 1. Load the CSV 
# Replace 'eagles_posterior.csv' with your actual filename.
# We use comment='#' to ignore the header line if it starts with a hash.
df = pd.read_csv('/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/PFS/eagles/tic88297141_age_pos.csv', comment='#', names=['log_age', 'probability'])

# 2. Convert log(Age/yr) to linear Age in Gyr
# 10^log_age gives years. Divide by 1e9 to get Gigayears.
ages_gyr = (10 ** df['log_age']) / 1e9

# 3. Normalize the probabilities 
# np.random.choice requires the probabilities to sum to exactly 1.0
weights = df['probability'] / df['probability'].sum()

# 4. Draw the ages
# This randomly selects ages from your EAGLES results, favoring the ones with higher probability.
EAGLES_AGE_GYR = np.random.choice(ages_gyr, size=len(Rstar_Rsun), p=weights)

# EAGLES_AGE = 18.2  # Myr


# # Convert 18.2 Myr to 0.0182 Gyr, and create an array the same size as MCMC draws
# EAGLES_AGE_GYR = np.full(len(Rstar_Rsun), EAGLES_AGE / 1000.0)

# Calculate the isochronal mass distributions
iso_masses_std, chi_array_mass = best_fit_masses(std_grid, EAGLES_AGE_GYR, Rstar_Rsun, TEFF_OBS, SIG_TEFF, sig_r)
iso_masses_mag, chi_array_mass2 = best_fit_masses(mag_grid, EAGLES_AGE_GYR, Rstar_Rsun, TEFF_OBS, SIG_TEFF, sig_r)

print("\n--- ISOCHRONAL MASS RESULTS (Fed in EAGLES age of 18.2 Myr with associated uncertainties) ---")

print(f"Median chisquare standard = {np.nanmedian(chi_array_mass)}")
print(f"Median chisquare magnetic = {np.nanmedian(chi_array_mass2)}")

# 1. Print Standard Model Mass
p16_s, p50_s, p84_s = np.percentile(iso_masses_std, [16, 50, 84])
print(f"Standard Model Isochronal Mass: {p50_s:.4f} (+{p84_s-p50_s:.4f}/-{p50_s-p16_s:.4f}) M_sun")

# 2. Print Magnetic Model Mass
p16_m, p50_m, p84_m = np.percentile(iso_masses_mag, [16, 50, 84])
print(f"Magnetic Model Isochronal Mass: {p50_m:.4f} (+{p84_m-p50_m:.4f}/-{p50_m-p16_m:.4f}) M_sun")

# 3. Reminder of the empirical mass for comparison
print(f"Empirical Mass (from transit/SED): {p50:.4f} M_sun (median)")

