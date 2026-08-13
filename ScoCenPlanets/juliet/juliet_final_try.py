'''
CORRECT WORKING CODE FOR GPR USING Quasi Periodic KERNEL
'''


import os 
import juliet
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import gridspec
import corner

### 29th March 2026 
#### REMINDERS FOR dynesty and nested sampling in general #### 

### dlogz for robust statistical analysis -> 0.01
### number of livepoints = 25-50 * N_dimensions, here our GP 
### dimensions is about N_dimensions = 16. So number of live
### points should be about 400-800. Our default setup is 500.
### let's try and increase that while decreasing dlogz

### Note : the above changes will drastically increase computation
### time. Therefore, it is recommended to use lower live points and
### a relatively high dlogz like 0.5 for exploratory runs!

### actually, since our results seemed fine, let's not change
### number of live points, since we were within the range, albeit
### towards the lower end.

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
number_of_cores = 40
run_number      = 22

# -------------------------------------------------------
# KNOWN STELLAR / ORBITAL PARAMS
# -------------------------------------------------------
p   = 4.64423    # orbital period (days)
t0  = 3803.24126 # transit centre (BTJD)
rot = 1.83     # stellar rotation period from Lomb-Scargle (days)

# -------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------
LCPATH1 = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025099153000-s0091-0000000088297141-0288-s/tess2025099153000-s0091-0000000088297141-0288-s_lc.fits"
LCPATH2 = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025127075000-s0092-0000000088297141-0289-s/tess2025127075000-s0092-0000000088297141-0289-s_lc.fits"


# -------------------------------------------------------
# 1. SIGMA CLIP FUNCTIONS (from WOTAN package)
# -------------------------------------------------------
def clipit(data, low, high, method, center):
    if center == 'mad':
        mid = np.nanmedian(data)
    else:
        mid = np.nanmean(data)
    data = np.nan_to_num(data)
    diff = data - mid
    if method == 'median':
        cutoff = np.nanmedian(np.abs(data - mid))
    else:
        cutoff = np.nanstd(data)
    data[diff >  high * cutoff] = np.nan
    data[diff < -low  * cutoff] = np.nan
    return data

def slide_clip(time, data, window_length, low=3, high=3, method=None, center=None):
    """Sliding time-windowed outlier clipper. From WOTAN package."""
    if method is None: method = 'mad'
    if center is None: center = 'median'
    low_index    = np.min(time)
    hi_index     = np.max(time)
    idx_start    = 0
    idx_end      = 0
    size         = len(time)
    half_window  = window_length / 2
    clipped_data = np.full(size, np.nan)
    for i in range(size - 1):
        if time[i] > low_index and time[i] < hi_index:
            while time[idx_start] < time[i] - half_window:
                idx_start += 1
            while time[idx_end] < time[i] + half_window and idx_end < size - 1:
                idx_end += 1
            clipped_data[idx_start:idx_end] = clipit(
                data[idx_start:idx_end], low, high, method, center)
    return clipped_data


# -------------------------------------------------------
# 2. LOAD + STITCH + NORMALISE + SIGMA CLIP
# -------------------------------------------------------

### we doing SAP for TRICERATOPS NOW! 
def load_tess_lc(lcpath):
    with fits.open(lcpath) as hdul:
        data = hdul[1].data
        time = data['TIME']
        flux = data['PDCSAP_FLUX']
        ferr = data['PDCSAP_FLUX_ERR']
    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(ferr)
    return time[mask], flux[mask], ferr[mask]

t1, flux1, ferr1 = load_tess_lc(LCPATH1)
t2, flux2, ferr2 = load_tess_lc(LCPATH2)

# Normalise each sector by its own median
flux1_norm = flux1 / np.nanmedian(flux1)
flux2_norm = flux2 / np.nanmedian(flux2)
ferr1_norm = ferr1 / np.nanmedian(flux1)
ferr2_norm = ferr2 / np.nanmedian(flux2)

t_full = t2
# # Stitch and sort
# t_full    = np.concatenate([t1, t2])
# flux_full = np.concatenate([flux1_norm, flux2_norm])
# ferr_full = np.concatenate([ferr1_norm, ferr2_norm])
# idx       = np.argsort(t_full)
# t_full    = t_full[idx]
# flux_full = flux_full[idx]
# ferr_full = ferr_full[idx]

# Sigma clip — aggressive upward (flares), disabled downward (protect transits)
# flux_clipped = slide_clip(
#     t_full, flux_full.copy(),
#     window_length=0.5,
#     low=100,
#     high=2.5,
#     method='mad',
#     center='median'
# )
flux_clipped = slide_clip(t2, flux2_norm.copy(), window_length = 0.5, low = 100, high = 2.5, method = 'mad', center = 'median')


mask_finite = np.isfinite(flux_clipped)

t_clean = t2[mask_finite]
#t_clean     = t_full[mask_finite]
flux_clean  = flux_clipped[mask_finite]
# ferr_clean  = ferr_full[mask_finite]
ferr_clean = ferr2_norm[mask_finite]

# print(f"Points before clipping: {len(t_full)}")
# print(f"Points after clipping:  {len(t_clean)}")
# print(f"Points removed:         {len(t_full) - len(t_clean)}")


# -------------------------------------------------------
# 3. FOLD T0 INTO DATA WINDOW + IDENTIFY ALL TRANSITS
# -------------------------------------------------------
N_orbits   = np.round((t_full.mean() - t0) / p)
T0_in_data = t0 + N_orbits * p

N_start = int(np.ceil( (t_full.min() - t0) / p))
N_end   = int(np.floor((t_full.max() - t0) / p))
all_T0s = t0 + np.arange(N_start, N_end + 1) * p

print(f"Data runs from {t_full.min():.2f} to {t_full.max():.2f} BTJD")
print(f"T0 folded into data window: {T0_in_data:.6f} BTJD")
print(f"Number of transits in data: {len(all_T0s)}")




# -------------------------------------------------------
# 4. JULIET PRIORS
#
# Kernel: GP_ExpSineSquared (QP kernel)
#   GP_B_TESS    : amplitude
#   GP_C_TESS    : harmonic complexity (like Gamma in Aigrain et al.)
#   GP_L_TESS    : evolutionary / decay timescale
#   GP_Prot_TESS : rotation period
#
# Transit parameterisation: Espinoza (2018) r1, r2
#   r1, r2 in [0,1] — maps to all physically valid (p, b) combinations
#   p  = r2       if r1 < Ar2/(1+r2) ... (juliet handles conversion internally)
#   Derived: p_p1 (Rp/R*), b_p1 (impact parameter)
#
# Stellar density rho: juliet derives a/R* from rho + period internally
# -------------------------------------------------------
priors = {}

params = [
    'P_p1',          # orbital period
    't0_p1',         # transit centre
    'p_p1',         # Espinoza (2018) radius/impact param 1
    'b_p1',         # Espinoza (2018) radius/impact param 2
    'q1_TESS',       # limb darkening q1 (Kipping 2013)
    'q2_TESS',       # limb darkening q2 (Kipping 2013)
    'ecc_p1',        # eccentricity — fixed circular
    'omega_p1',      # argument of periastron — fixed
    'rho',           # stellar density (kg/m³) — juliet derives a/R* from this
    'mdilution_TESS',# dilution factor — fixed at 1 (no contamination)
    'mflux_TESS',    # flux offset
    'sigma_w_TESS',  # white noise jitter
    'GP_B_TESS',     # QP kernel amplitude
    'GP_C_TESS',     # QP kernel harmonic complexity
    'GP_L_TESS',     # QP kernel decay timescale
    'GP_Prot_TESS',  # QP kernel rotation period
]

dists = [
    'normal',        # P_p1
    'normal',        # t0_p1
    'uniform',       # p_p1
    'uniform',       # b_p1
    'uniform',       # q1_TESS
    'uniform',       # q2_TESS
    'fixed',         # ecc_p1
    'fixed',         # omega_p1
    'loguniform',    # rho
    'fixed',         # mdilution_TESS
    'normal',        # mflux_TESS
    'loguniform',    # sigma_w_TESS
    'loguniform',    # GP_B_TESS
    'uniform',       # GP_C_TESS
    'loguniform',    # GP_L_TESS
    'normal',        # GP_Prot_TESS
]

hyperps = [
    [p, 0.01],           # P_p1 — tight Gaussian on known period
    [T0_in_data, 0.1],   # t0_p1 — tight Gaussian on folded T0
    [0.,1.],            # p_p1
    [0., 1.2],            # b_p1
    [0., 1.],            # q1_TESS
    [0., 1.],            # q2_TESS
    0.0,                 # ecc_p1
    90.,                 # omega_p1
    [100, 50000],        # stellar density rho — loguniform, M dwarf range --> change this to [100,50000] for next run
    1.0,                 # mdilution_TESS
    [0., 0.1],           # mflux_TESS
    [0.1, 1000.],        # sigma_w_TESS — jitter in ppm
    [1e-6, 1.0],         # GP_B_TESS — amplitude
    [0.0,  1.0],         # GP_C_TESS — harmonic complexity
    [0.5,  100.],        # GP_L_TESS — decay timescale
    [rot, 0.5],          # GP_Prot_TESS — Gaussian prior on known rotation period
]

# Populate priors dictionary in juliet format
for param, dist, hyperp in zip(params, dists, hyperps):
    priors[param] = {}
    priors[param]['distribution']   = dist
    priors[param]['hyperparameters'] = hyperp


# -------------------------------------------------------
# 5. LOAD DATASET INTO JULIET
# -------------------------------------------------------
times        = {'TESS': t_clean}
fluxes       = {'TESS': flux_clean}
fluxes_error = {'TESS': ferr_clean}

dataset = juliet.load(
    priors         = priors,
    t_lc           = times,
    y_lc           = fluxes,
    yerr_lc        = fluxes_error,
    GP_regressors_lc = times,
    out_folder     = f'88297141_GP_QP_SAP_v{run_number}',
    verbose        = True
)

print(f"Fitting with dynesty using {number_of_cores} cores...")


# -------------------------------------------------------
# 6. FIT 
# -------------------------------------------------------

### For run number = 4, we are updating the dataset.fit for a more complex analysis.
### we are going to make the dlogz threshold 0.01, instead of the "default" version which runs 
# when add_live = True. For more info, import dynesty and type help(dynesty.NestedSampler.run_nested)

results = dataset.fit(use_dynesty=True, dynesty_nthreads=number_of_cores, dlogz = 0.01)

# -------------------------------------------------------
# 7. RESULTS
# -------------------------------------------------------
posterior_samples = results.posteriors['posterior_samples']

print("\n" + "="*60)
print("BEST-FIT PARAMETERS (Median ± 1σ)")
print("="*60)

params_to_report = {
    'P_p1':        'Period (days)',
    't0_p1':       'T0 (BTJD)',
    'rho':         'Stellar density (kg/m³)',
    'b_p1':       'Impact parameter',
    'p_p1':       'Rp/R*',
    'GP_B_TESS':   'GP amplitude',
    'GP_C_TESS':   'GP harmonic complexity',
    'GP_L_TESS':   'GP decay timescale (days)',
    'GP_Prot_TESS':'GP rotation period (days)',
    'sigma_w_TESS':'Jitter',
}

for param, label in params_to_report.items():
    med  = np.median(posterior_samples[param])
    lo   = np.percentile(posterior_samples[param], 16)
    hi   = np.percentile(posterior_samples[param], 84)
    print(f"  {label:35s}: {med:.6f} +{hi-med:.6f} -{med-lo:.6f}")
'''
# Derive p (Rp/R*) and b (impact parameter) from Espinoza r1, r2
r1 = posterior_samples['r1_p1']
r2 = posterior_samples['r2_p1']

### correcting the impatc parameter and Rp/Rs parameters

b, p_p1 = juliet.utils.reverse_bp(r1, r2, 0., 1.)

p_med, p_hi, p_lo = juliet.utils.get_quantiles(p_p1)
b_med, b_hi, b_lo = juliet.utils.get_quantiles(b)

#p_p1 = np.where(r2 < 0.5, r1, 1 - r1)
#b_p1 = np.where(r2 < 0.5, r2 * (1 + p_p1), (1 - r2) * (1 + p_p1))

posterior_samples['p_p1'] = p_p1
posterior_samples['b_p1'] = b

print("\n--- Derived Parameters ---")
print(f"  Rp/R*              : {p_med:.6f} +{p_hi-p_med:.6f} -{p_med-p_lo:.6f}")
print(f"  Impact parameter b : {b_med:.6f} +{b_hi-b_med:.6f} -{b_med-b_lo:.6f}")
'''
print("="*60)

results_dir = os.path.join('results', f'SAP_run_v{run_number}')
os.makedirs(results_dir, exist_ok=True)

# -------------------------------------------------------
# 8. PLOTS
# -------------------------------------------------------
# Extract model components
transit_plus_gp = results.lc.evaluate('TESS')
transit_model   = results.lc.model['TESS']['deterministic']
gp_model        = results.lc.model['TESS']['GP']
gp_corrected =  dataset.data_lc['TESS'] - gp_model 

### added this to save the gp stuff cuz im not able to access em
np.savez(
    os.path.join(results_dir, f'88297141_GP_QP_joint_v{run_number}_gp_SAP_results.npz'),
    time=dataset.times_lc['TESS'],
    flux=dataset.data_lc['TESS'],
    fluxerr=dataset.errors_lc['TESS'],
    gp=gp_model,
    deterministic=transit_model,
    full_model=transit_plus_gp,
    gp_corrected=gp_corrected
)


# Compute phases using best-fit period and t0
p_best  = np.median(posterior_samples['P_p1'])
t0_best = np.median(posterior_samples['t0_p1'])
phases  = juliet.utils.get_phases(dataset.times_lc['TESS'], p_best, t0_best)


# -------------------------------------------------------
# Phase-fold and bin the GP-corrected data
# -------------------------------------------------------

def bin_in_phase(phases, y, yerr, bin_width, phase_range=(-0.05, 0.05)):
    """Weighted-mean binning in phase space. Returns only non-empty bins."""
    bin_edges = np.arange(phase_range[0], phase_range[1] + bin_width, bin_width)
    centers, means, errs = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (phases >= lo) & (phases < hi)
        if in_bin.sum() > 0:
            w = 1. / yerr[in_bin]**2
            centers.append(0.5 * (lo + hi))
            means.append(np.average(y[in_bin], weights=w))
            errs.append(1. / np.sqrt(np.sum(w)))
    return np.array(centers), np.array(means), np.array(errs)

bin_width_minutes = 20.
phase_span = max(np.abs(phases).max() * 1.1, 0.01)
bin_width  = bin_width_minutes / (p_best * 24. * 60.)

bin_centers, bin_flux, bin_err = bin_in_phase(
    phases,
    gp_corrected,
    dataset.errors_lc['TESS'],
    bin_width,
    phase_range=(-phase_span, phase_span)
)


# --- Plot 1: full light curve + model ---
fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
mask = dataset.data_lc['TESS'] > 0.8

ax1.errorbar(dataset.times_lc['TESS'][mask], dataset.data_lc['TESS'][mask],
             yerr=dataset.errors_lc['TESS'][mask], fmt='.', alpha=0.1, label='Data')
ax1.plot(dataset.times_lc['TESS'][mask], transit_plus_gp[mask],
         color='black', lw=2, zorder=10, label='Transit + GP model')
ax1.set_xlabel('Time (BTJD)', fontsize=12)
ax1.set_ylabel('Relative flux', fontsize=12)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_title('Full light curve')

# --- Plot 2: phase-folded, GP subtracted ---
ax2 = plt.subplot(gs[1])
idx_sort = np.argsort(phases)

ax2.errorbar(phases, dataset.data_lc['TESS'] - gp_model,
             yerr=dataset.errors_lc['TESS'], fmt='.', alpha=0.3,
             label='GP-corrected data')
ax2.plot(phases[idx_sort], transit_model[idx_sort],
         color='black', lw=2, zorder=10, label='Transit model')
if len(bin_centers) > 0:
    ax2.errorbar(bin_centers, bin_flux, yerr=bin_err, fmt='o',
                    color='navy', ms=5, elinewidth=1.5, capsize=2,
                    zorder=4, label=f'{int(bin_width_minutes)}-min bins')

ax2.set_xlabel('Phase', fontsize=12)
ax2.set_ylabel('Relative flux', fontsize=12)
ax2.set_xlim([-0.05, 0.05])
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_title('Phase-folded transit')

instrument = 'TESS'

model = results.lc.evaluate(instrument)

t = dataset.times_lc[instrument]
flux = dataset.data_lc[instrument]
residuals = flux - model

# phase-fold on the fitted period/t0
P = np.median(results.posteriors['posterior_samples']['P_p1'])
t0 = np.median(results.posteriors['posterior_samples']['t0_p1'])
phase_hr = (((t - t0 + 0.5*P) % P) / P - 0.5) * P * 24  # hours from mid-transit
order = np.argsort(phase_hr)

plt.tight_layout()
plt.savefig(f'88297141_GP_QP_fit_SAP_v{run_number}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: 88297141_GP_QP_fit_SAP_v{run_number}.png")

# --- Plot 3: corner plot ---
params_corner = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'rho',
                 'GP_B_TESS', 'GP_Prot_TESS']
labels_corner = ['Period (d)', 'T0 (BTJD)', 'Rp/R*', 'b',
                 'ρ* (kg/m³)', 'GP amp', 'GP Prot (d)']

samples_corner = np.array([posterior_samples[param]
                            for param in params_corner]).T

fig_corner = corner.corner(
    samples_corner,
    labels=labels_corner,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 10},
    label_kwargs={"fontsize": 12}
)
fig_corner.savefig(f'88297141_corner_SAP_v{run_number}.png', dpi=300, bbox_inches='tight')


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})

ax1.scatter(phase_hr, flux, s=4, alpha=0.4, color='gray')
ax1.plot(phase_hr[order], model[order], color='crimson', lw=1.5, zorder=5)
ax1.set_ylabel('Relative flux')

ax2.scatter(phase_hr, residuals, s=4, alpha=0.4, color='gray')
ax2.axhline(0, color='k', ls='--', lw=1)
ax2.set_ylabel('Residuals')
ax2.set_xlabel('Hours from mid-transit')

half_window_hr = 2.5  # ~t14/2 (220.5 min) + a bit of padding
ax1.set_xlim(-half_window_hr, half_window_hr)
ax2.set_xlim(-half_window_hr, half_window_hr)
plt.tight_layout()
plt.show()
plt.savefig(f"88297141_GP_QP_fit_SAP_v{run_number}_residuals.png", dpi = 300, bbox_inches = 'tight')

plt.close()
print(f"Saved: 88297141_corner_SAP_v{run_number}.png")
print("Done.")