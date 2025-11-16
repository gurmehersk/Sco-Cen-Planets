### Per-transit-window polynomial detrending + transit fitting
### Like the betty example - each transit gets its own local polynomial

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import pickle

### For exoplanet package transit fitting & mcmc
import pymc as pm
import exoplanet as xo
import pytensor.tensor as pt
import arviz as az

def clean_arrays(time, flux):
    mask = (~np.isnan(time)) & (~np.isnan(flux))
    return np.asarray(time[mask], dtype=np.float64), np.asarray(flux[mask], dtype=np.float64)

def slide_clip(time, data, window_length, low=3, high=3, method='mad', center='median'):
    """Sliding time-windowed outlier clipper."""
    def clipit(data, low, high, method, center):
        if center == 'median':
            mid = np.nanmedian(data)
        else:
            mid = np.nanmean(data)
        data = np.nan_to_num(data)
        diff = data - mid
        if method == 'mad':
            cutoff = np.nanmedian(np.abs(data - mid))
        else:
            cutoff = np.nanstd(data)
        data[diff > high * cutoff] = np.nan
        data[diff < -low * cutoff] = np.nan
        return data
    
    low_index = np.min(time)
    hi_index = np.max(time)
    idx_start = 0
    idx_end = 0
    size = len(time)
    half_window = window_length / 2
    clipped_data = np.full(size, np.nan)
    for i in range(size-1):
        if time[i] > low_index and time[i] < hi_index:
            while time[idx_start] < time[i] - half_window:
                idx_start += 1
            while time[idx_end] < time[i] + half_window and idx_end < size-1:
                idx_end += 1
            clipped_data[idx_start:idx_end] = clipit(
                data[idx_start:idx_end], low, high, method, center)
    return clipped_data

# =============================================================================
# DATA LOADING
# =============================================================================

tic_id = 88297141
path2 = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025127075000-s0092-0000000088297141-0289-s/tess2025127075000-s0092-0000000088297141-0289-s_lc.fits"
hdu_list2 = fits.open(path2)
data2 = hdu_list2[1].data
time = data2['TIME']
pdcsap_flux = data2['PDCSAP_FLUX']/np.nanmedian(data2['PDCSAP_FLUX'])

time, pdcsap_flux = clean_arrays(time, pdcsap_flux)

# Clip outliers
clipped_flux = slide_clip(time, pdcsap_flux, window_length=1, low=100, high=2)
time_clean, flux_clean = clean_arrays(time, 1.*clipped_flux)

print(f"TIC {tic_id}")
print(f"Total data points: {len(time_clean)}")

flux_err = np.asarray(np.ones_like(flux_clean) * np.nanstd(flux_clean), dtype=np.float64)

# =============================================================================
# IDENTIFY TRANSIT WINDOWS
# =============================================================================

t0_initial = 3803.23
period_initial = 4.6370567442246955
r_star = 0.7375

# Calculate expected transit times in the data
t_min = time_clean.min()
t_max = time_clean.max()

# Find all transit centers in the data range
transit_times = []
n_transit = int((t_min - t0_initial) / period_initial) - 1
while True:
    t_transit = t0_initial + n_transit * period_initial
    if t_transit > t_max + period_initial:
        break
    if t_transit >= t_min - period_initial:
        transit_times.append(t_transit)
    n_transit += 1

print(f"\nFound {len(transit_times)} potential transits in data range")

# Define transit windows (±4 hours from center)
window_half_width = 4.0 / 24.0  # 4 hours in days
transit_windows = []

for t_transit in transit_times:
    # Get data in this window
    mask = np.abs(time_clean - t_transit) < window_half_width
    if np.sum(mask) > 10:  # Need at least 10 points
        t_window = time_clean[mask]
        f_window = flux_clean[mask]
        ferr_window = flux_err[mask]
        transit_windows.append({
            't': t_window,
            'f': f_window,
            'ferr': ferr_window,
            't_center': t_transit,
            'mask': mask
        })

print(f"Using {len(transit_windows)} transit windows with data")
for i, tw in enumerate(transit_windows):
    print(f"  Window {i}: {len(tw['t'])} points around t={tw['t_center']:.2f}")

# =============================================================================
# BUILD MODEL WITH PER-WINDOW POLYNOMIALS
# =============================================================================

with pm.Model() as transit_model:
    
    # =========================================================================
    # SHARED TRANSIT PARAMETERS
    # =========================================================================
    
    t0_var = pm.Normal("t0", mu=t0_initial, sigma=0.1)
    period_var = pm.Normal("period", mu=period_initial, sigma=0.01)
    log_ror_var = pm.Uniform("log_ror", lower=np.log(0.01), upper=np.log(0.3), initval=np.log(0.05))
    ror_var = pm.Deterministic("ror", pt.exp(log_ror_var))
    b_var = pm.Uniform("b", lower=0.0, upper=1.3, initval=0.5)
    rho_star_var = pm.Normal("rho_star", mu=1.0, sigma=0.5)
    u_var = xo.distributions.QuadLimbDark("u")
    
    # Orbit
    orbit_var = xo.orbits.KeplerianOrbit(
        period=period_var, t0=t0_var, b=b_var,
        rho_star=rho_star_var, r_star=r_star
    )
    
    star = xo.LimbDarkLightCurve(u_var)
    
    # Compute transit model for ALL data at once (avoids shape issues)
    transit_lc_full = star.get_light_curve(
        orbit=orbit_var, 
        r=ror_var * r_star, 
        t=time_clean
    )
    transit_lc_full = pt.sum(transit_lc_full, axis=-1).flatten()
    
    # Jitter (shared)
    log_jitter = pm.Normal("log_jitter", mu=np.log(np.median(flux_err)), sigma=2)
    
    # =========================================================================
    # PER-WINDOW POLYNOMIALS + LIKELIHOODS
    # =========================================================================
    
    for i, tw in enumerate(transit_windows):
        t_window = tw['t']
        f_window = tw['f']
        ferr_window = tw['ferr']
        mask_window = tw['mask']
        
        # Extract transit model for this window
        transit_lc_i = transit_lc_full[mask_window]
        
        # Normalize time for this window
        t_mid = np.median(t_window)
        t_norm = (t_window - t_mid) / np.std(t_window)
        
        # Per-window polynomial coefficients (2nd order)
        mean_i = pm.Normal(f"mean_{i}", mu=0.0, sigma=0.1)
        a1_i = pm.Normal(f"a1_{i}", mu=0, sigma=0.1, initval=0)
        a2_i = pm.Normal(f"a2_{i}", mu=0, sigma=0.1, initval=0)
        
        # Polynomial trend for this window
        trend_i = mean_i + a1_i * t_norm + a2_i * t_norm**2
        
        # Combined model for this window
        model_flux_i = trend_i + transit_lc_i
        
        # Likelihood for this window
        pm.Normal(
            f"obs_{i}",
            mu=model_flux_i,
            sigma=pt.sqrt(ferr_window**2 + pt.exp(2 * log_jitter)),
            observed=f_window
        )
    
    # =========================================================================
    # SAMPLING
    # =========================================================================
    
    print("\nStarting MCMC sampling...")
    trace_transit = pm.sample(
        tune=2000,
        draws=2000,
        target_accept=0.9,
        return_inferencedata=True,
        cores=2,
        init='jitter+adapt_diag'
    )

# =============================================================================
# POST-PROCESSING
# =============================================================================

print("\nSampling complete! Processing results...")

az.plot_trace(trace_transit, var_names=["t0", "period", "ror", "b", "rho_star"])
plt.savefig(f"trace_plot_perwindow_TIC{tic_id}_s92.pdf", bbox_inches="tight")
plt.close()

summary = az.summary(trace_transit, var_names=["t0", "period", "ror", "b", "rho_star"])
print(summary)

# Extract posterior medians
posterior = trace_transit.posterior
period_med = posterior["period"].median().item()
t0_med = posterior["t0"].median().item()
ror_med = posterior["ror"].median().item()
b_med = posterior["b"].median().item()
rho_star_med = posterior["rho_star"].median().item()
u_med = [posterior["u"][:, :, i].median().item() for i in range(2)]

# Build best-fit transit model
orbit_best = xo.orbits.KeplerianOrbit(
    period=period_med, t0=t0_med, b=b_med,
    rho_star=rho_star_med, r_star=r_star
)
ld_model = xo.LimbDarkLightCurve(u_med)

# Reconstruct the full detrended light curve
time_detrended = []
flux_detrended = []
transit_model_full = []

for i, tw in enumerate(transit_windows):
    t_window = tw['t']
    f_window = tw['f']
    
    # Get polynomial for this window
    t_mid = np.median(t_window)
    t_norm = (t_window - t_mid) / np.std(t_window)
    
    mean_i = posterior[f"mean_{i}"].median().item()
    a1_i = posterior[f"a1_{i}"].median().item()
    a2_i = posterior[f"a2_{i}"].median().item()
    
    trend_i = mean_i + a1_i * t_norm + a2_i * t_norm**2
    
    # Transit model for this window
    transit_i = ld_model.get_light_curve(
        orbit=orbit_best, r=ror_med * r_star, t=t_window
    ).eval().flatten()
    
    # Detrend: remove polynomial, keep transit + baseline of 1.0
    flux_det = f_window - trend_i + 1.0
    
    time_detrended.extend(t_window)
    flux_detrended.extend(flux_det)
    transit_model_full.extend(transit_i + 1.0)

time_detrended = np.array(time_detrended)
flux_detrended = np.array(flux_detrended)
transit_model_full = np.array(transit_model_full)

# Phase fold
phase_data = ((time_detrended - t0_med + 0.5 * period_med) % period_med) / period_med - 0.5

# =============================================================================
# PLOTTING
# =============================================================================

def phase_bin(phase, flux, bins=100):
    bin_edges = np.linspace(-0.5, 0.5, bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    digitized = np.digitize(phase, bin_edges) - 1
    binned_flux = [np.nanmedian(flux[digitized == i]) if np.any(digitized == i) else np.nan 
                   for i in range(bins)]
    return bin_centers, np.array(binned_flux)

sort_idx = np.argsort(phase_data)
phase_sorted = phase_data[sort_idx]
flux_sorted = flux_detrended[sort_idx]
transit_sorted = transit_model_full[sort_idx]

bin_phase, bin_flux = phase_bin(phase_sorted, flux_sorted, 100)

# Full phase plot
plt.figure(figsize=(19, 10))
plt.scatter(phase_sorted, flux_sorted, s=5, color="k", alpha=0.3, label="Detrended Data")
plt.scatter(bin_phase, bin_flux, s=40, color='dodgerblue', label='Binned Data', zorder=2)
plt.plot(phase_sorted, transit_sorted, color="C1", linewidth=2, label="Transit Model")
plt.xlabel("Phase", fontsize=14)
plt.ylabel("Normalized Flux", fontsize=14)
plt.legend(fontsize=12)
plt.title(f"TIC {tic_id} - Phase-folded (Per-window polynomial detrending)", fontsize=16)
plt.savefig(f"phase_fold_perwindow_TIC{tic_id}_s92.pdf", bbox_inches="tight")
plt.close()

# Zoomed ±5 hours
plt.figure(figsize=(12, 8))
phase_hours = phase_sorted * period_med * 24
mask_zoom = np.abs(phase_hours) <= 5
plt.scatter(phase_hours[mask_zoom], flux_sorted[mask_zoom], s=8, color="k", alpha=0.3, label="Detrended Data")
plt.plot(phase_hours[mask_zoom], transit_sorted[mask_zoom], color="C1", linewidth=3, label="Transit Model", zorder=3)
plt.xlabel("Time from Transit [hours]", fontsize=14)
plt.ylabel("Normalized Flux", fontsize=14)
plt.legend(fontsize=12)
plt.title(f"TIC {tic_id} - Transit Zoom (±5 hours)", fontsize=16)
plt.xlim(-5, 5)
plt.grid(alpha=0.3)
plt.savefig(f"transit_zoom_5hr_perwindow_TIC{tic_id}_s92.pdf", bbox_inches="tight")
plt.close()

# Corner plot
import corner
samples = np.vstack([
    posterior["t0"].values.flatten(),
    posterior["period"].values.flatten(),
    posterior["ror"].values.flatten(),
    posterior["b"].values.flatten(),
    posterior["rho_star"].values.flatten()
]).T
labels = [r"$t_0$", r"$P$", r"$R_p/R_s$", r"$b$", r"$\rho_\star$"]
fig = corner.corner(samples, labels=labels, show_titles=True, title_fmt=".5f")
fig.savefig(f"corner_plot_perwindow_TIC{tic_id}_s92.pdf", bbox_inches="tight")
plt.close()

# Save model
model_dict = {
    "tic_id": tic_id,
    "time_detrended": time_detrended,
    "flux_detrended": flux_detrended,
    "transit_model": transit_model_full,
    "phase": phase_data,
    "t0_med": t0_med,
    "period_med": period_med,
    "ror_med": ror_med,
    "b_med": b_med,
    "rho_star_med": rho_star_med,
    "u_med": u_med,
    "n_windows": len(transit_windows)
}

cache_dir = "cached_models"
os.makedirs(cache_dir, exist_ok=True)
model_path = os.path.join(cache_dir, f"TIC{tic_id}_perwindow_model_s92.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model_dict, f)

print(f"\nModel saved to {model_path}")
print(f"\nFinal parameters:")
print(f"Period: {period_med:.6f} days")
print(f"t0: {t0_med:.6f} days")
print(f"Rp/Rs: {ror_med:.4f}")
print(f"Impact parameter: {b_med:.3f}")
print(f"Stellar density: {rho_star_med:.3f} g/cm³")
print(f"Number of transit windows: {len(transit_windows)}")