'''
13th April 2026
Quick Script for potential TTVs in the 88297141 data

'''


import juliet
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import gridspec
import corner
import emcee 
import batman 
import os 

# -------------------------------------------------------
# orbital params from the previous GP fit,  
# we arenot going to do the fitting again
# -------------------------------------------------------
p   = 4.644695    # orbital period (days)
t0  = 3803.239888 # transit centre (BTJD)
duration = 0.155 # going a little over 3.5 hours to account for 0.1h error


# MCMC settings
nwalkers = 24
nsteps = 2500
nburn = 1000
thin = 10

# Output
plot_dir = "transit_window_plots"
os.makedirs(plot_dir, exist_ok=True)

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

# Stitch and sort
t_full    = np.concatenate([t1, t2])
flux_full = np.concatenate([flux1_norm, flux2_norm])
ferr_full = np.concatenate([ferr1_norm, ferr2_norm])
idx       = np.argsort(t_full)
t_full    = t_full[idx]
flux_full = flux_full[idx]
ferr_full = ferr_full[idx]

# Sigma clip — aggressive upward (flares), disabled downward (protect transits)
flux_clipped = slide_clip(
    t_full, flux_full.copy(),
    window_length=0.5,
    low=100,
    high=2.5,
    method='mad',
    center='median'
)

mask_finite = np.isfinite(flux_clipped)
t_clean     = t_full[mask_finite]
flux_clean  = flux_clipped[mask_finite]
ferr_clean  = ferr_full[mask_finite]

print(f"Points before clipping: {len(t_full)}")
print(f"Points after clipping:  {len(t_clean)}")
print(f"Points removed:         {len(t_full) - len(t_clean)}")



### our time series is saved in the t_clean, flux_clean, ferr_clean variables. the errors are redundant for
#  our work

### now we create a half window variable to allow for off transit points for our polynomial fitter to anchor and use

half_window = 2.0 * duration  # more than enough off transit points for a polynomial imo

## minimum transit number, basically the linear multiple we shd start checking O-C for.
N_min = int(np.floor((t_clean.min() - t0) / p))

## for instance, if our data starts 50 days before t0, we would have 50/4.446 ~ -10.7 which gets floored to -11
## so our linear ephermeris calculations would start from -11 all the way to some positive number which we find from N_max 
## similarly

N_max = int(np.ceil((t_clean.max()-t0)/ p))

transit_numbering = np.arange(N_min, N_max +1)

transit_windows = []
for N in transit_numbering:
    t_predicted = t0 + N * p
    mask = np.abs(t_clean - t_predicted) < half_window

    ### the construct of this mask is imp to understand
    ### For a given transit N, the mask is asking for every point: 
    # "is this data point close enough to the predicted midpoint
    # to be included in this window?" draw a visual, it'll help
    # understand this a lot better. that's what i did [GK] to 
    # think abt it 

    if mask.sum() > 10: ## if less, chances are we are in a downlink gap, so not enough to fit
        transit_windows.append({
            'N': N,
            't_predicted': t_predicted,
            'time': t_clean[mask],   # just this window's times
            'flux': flux_clean[mask],   # just this window's fluxes
            'flux_err': ferr_clean[mask]
        })

print(f"Found {len(transit_windows)} transit windows")


#------------------------------------------------------------#
# Okay now let's do the polynomial + transit fit
##-----------------------------------------------------------#
import numpy as np
import batman
import matplotlib.pyplot as plt
from scipy.optimize import minimize

## fixed params
# --- Orbital ---
t0        = 3803.239888
period    = 4.644695
rp        = 0.092670       # Rp/R*
b         = 0.313832       # impact parameter
rho_star  = 1090.520350    # kg/m³

# --- Limb darkening (Kipping 2013 to batman u1, u2) ---
q1 = 0.2654260321 
q2 = 0.3770894837
u1 = 2 * np.sqrt(q1) * q2
u2 = np.sqrt(q1) * (1 - 2 * q2)

G = 6.674e-11
P_sec = period * 86400
aRs = ((G * rho_star * P_sec**2) / (3 * np.pi))**(1/3)
print(f"Derived a/R* = {aRs:.4f}")

print("found a/R* = 10.7543167448")

# --- Derive inclination from b and a/R* ---
inc = np.degrees(np.arccos(b / aRs))
print(f"Derived inc  = {inc:.4f} deg")

print("Found inc = 88.339")

# -------------------------------------------------------
# BATMAN SETUP (fixed params — only t_mid varies)

def make_batman_params(t_mid):
    params = batman.TransitParams()
    params.t0  = t_mid
    params.per = period
    params.rp  = rp
    params.a   = aRs
    params.inc = inc
    params.ecc = 0.0
    params.w   = 90.0
    params.u   = [u1, u2]
    params.limb_dark = 'quadratic'
    return params

def transit_model(time_arr, t_mid):
    params = make_batman_params(t_mid)
    m = batman.TransitModel(params, time_arr)
    return m.light_curve(params)


# -------------------------------------------------------
# PER-WINDOW FIT: transit * polynomial simultaneously
# poly_order = 2, but change to 3 if needed
# -------------------------------------------------------

poly_order = 2

def full_model(time_arr, t_mid, poly_coeffs):
    """Transit model multiplied by a local polynomial baseline."""
    dt = time_arr - t_mid                          # centre polynomial on midpoint
    poly = np.polyval(poly_coeffs, dt)             # evaluate polynomial
    transit = transit_model(time_arr, t_mid)
    return transit * poly

def chi2(theta, time_arr, flux_arr, err_arr):
    t_mid = theta[0]
    poly_coeffs = theta[1:]                        # length = poly_order + 1
    model = full_model(time_arr, t_mid, poly_coeffs)
    return np.sum(((flux_arr - model) / err_arr)**2)


### vibe coded from here 

def log_prior(theta, t_pred, poly_order_local):
    t_mid = theta[0]
    coeffs = theta[1:]
    # Uniform prior on midpoint near predicted
    if not (t_pred - 0.5 < t_mid < t_pred + 0.5):
        return -np.inf
    # Weak Gaussian priors on poly coeffs to regularize extremes
    # Constant term around 1.0; other coeffs around 0
    if len(coeffs) != poly_order_local + 1:
        return -np.inf
    lp = 0.0
    for i, c in enumerate(coeffs):
        if i == len(coeffs) - 1:
            lp += -0.5 * ((c - 1.0) / 0.1) ** 2
        else:
            lp += -0.5 * (c / 5.0) ** 2
    return lp
def log_likelihood(theta, time_arr, flux_arr, err_arr):
    model = full_model(time_arr, theta[0], theta[1:])
    resid = (flux_arr - model) / err_arr
    return -0.5 * np.sum(resid**2 + np.log(2.0 * np.pi * err_arr**2))
def log_probability(theta, t_pred, time_arr, flux_arr, err_arr, poly_order_local):
    lp = log_prior(theta, t_pred, poly_order_local)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, time_arr, flux_arr, err_arr)
# -------------------------------------------------------
# FIT LOOP: OPTIMIZE + MCMC
# -------------------------------------------------------
t_obs_list = []
t_obs_err_list = []
N_list = []
t_pred_list = []
for win in transit_windows:
    t_win = win["time"]
    f_win = win["flux"]
    e_win = win["flux_err"]
    N = win["N"]
    t_pred = win["t_predicted"]
    # Initial guess for deterministic optimizer
    p0_poly = np.zeros(poly_order + 1)
    p0_poly[-1] = np.median(f_win)
    theta0 = np.concatenate([[t_pred], p0_poly])
    lower = [t_pred - 0.5] + [-np.inf] * (poly_order + 1)
    upper = [t_pred + 0.5] + [ np.inf] * (poly_order + 1)
    bounds = list(zip(lower, upper))
    result = minimize(
        chi2, theta0, args=(t_win, f_win, e_win),
        method="L-BFGS-B", bounds=bounds
    )
    if not result.success:
        print(f"N={N:+04d} optimize failed; skipping")
        continue
    theta_best = result.x
    t_mid_best = theta_best[0]
    # MCMC around optimum
    ndim = 1 + (poly_order + 1)
    pos = theta_best + 1e-4 * np.random.randn(nwalkers, ndim)
    # Keep walkers in prior
    for i in range(nwalkers):
        if not np.isfinite(log_prior(pos[i], t_pred, poly_order)):
            pos[i, 0] = np.clip(pos[i, 0], t_pred - 0.49, t_pred + 0.49)
            pos[i, 1:-1] = 0.0
            pos[i, -1] = 1.0
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(t_pred, t_win, f_win, e_win, poly_order)
    )
    sampler.run_mcmc(pos, nsteps, progress=False)
    flat = sampler.get_chain(discard=nburn, thin=thin, flat=True)
    t_samples = flat[:, 0]
    # Median + 16/84 percentile uncertainty
    t16, t50, t84 = np.percentile(t_samples, [16, 50, 84])
    t_mid = t50
    t_err = 0.5 * (t84 - t16)
    # Store
    N_list.append(N)
    t_pred_list.append(t_pred)
    t_obs_list.append(t_mid)
    t_obs_err_list.append(t_err)
    # Plot window with best model at posterior median params
    med_params = np.median(flat, axis=0)
    model_fit = full_model(t_win, med_params[0], med_params[1:])
    dt = t_win - med_params[0]
    poly_only = np.polyval(med_params[1:], dt)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle(
        f"N={N:+04d} | t_pred={t_pred:.6f} | t_fit={t_mid:.6f} +/- {t_err*24*60:.2f} min"
    )
    axes[0].errorbar(t_win, f_win, yerr=e_win, fmt=".", color="steelblue", alpha=0.7, label="Data")
    axes[0].plot(t_win, model_fit, "k-", lw=2, label="Transit x poly")
    axes[0].plot(t_win, poly_only, "r--", lw=1.5, label="Polynomial")
    axes[0].axvline(t_mid, color="orange", ls="--", lw=1, label="Fitted t_mid")
    axes[0].axvline(t_pred, color="gray", ls=":", lw=1, label="Predicted t_mid")
    axes[0].set_ylabel("Relative flux")
    axes[0].legend(fontsize=8)
    resid = f_win - model_fit
    axes[1].errorbar(t_win, resid, yerr=e_win, fmt=".", color="steelblue", alpha=0.7)
    axes[1].axhline(0.0, color="k", lw=1)
    axes[1].set_ylabel("Residuals")
    axes[1].set_xlabel("Time (BTJD)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"transit_N{N:+04d}.png"), dpi=120)
    plt.close()
    print(f"N={N:+04d} | t_mid={t_mid:.7f} +/- {t_err*24*60:.2f} min")
print(f"\nSuccessfully measured {len(t_obs_list)} transits.")
# -------------------------------------------------------
# O-C ANALYSIS (with ephemeris refit)
# -------------------------------------------------------
N_arr = np.array(N_list, dtype=int)
t_obs = np.array(t_obs_list, dtype=float)
sig_t = np.array(t_obs_err_list, dtype=float)
# Sort by transit number
srt = np.argsort(N_arr)
N_arr = N_arr[srt]
t_obs = t_obs[srt]
sig_t = sig_t[srt]
# Weighted linear ephemeris fit: t(N) = t_ref + P*N
A = np.vstack([np.ones_like(N_arr), N_arr]).T
W = np.diag(1.0 / sig_t**2)
cov_beta = np.linalg.inv(A.T @ W @ A)
beta = cov_beta @ (A.T @ W @ t_obs)
t_ref_fit, P_fit = beta[0], beta[1]
sig_tref = np.sqrt(cov_beta[0, 0])
sig_P = np.sqrt(cov_beta[1, 1])
cov_tref_P = cov_beta[0, 1]
t_calc = t_ref_fit + P_fit * N_arr
oc_days = t_obs - t_calc
# O-C uncertainty including ephemeris uncertainty
oc_var = sig_t**2 + sig_tref**2 + (N_arr**2) * sig_P**2 + 2.0 * N_arr * cov_tref_P
oc_var = np.where(oc_var > 0, oc_var, sig_t**2)
oc_err_days = np.sqrt(oc_var)
oc_min = oc_days * 24.0 * 60.0
oc_err_min = oc_err_days * 24.0 * 60.0
wrms = np.sqrt(np.average(oc_min**2, weights=1.0 / oc_err_min**2))
chi2_zero = np.sum((oc_min / oc_err_min)**2)
dof_zero = len(oc_min)
print("\nRefit ephemeris:")
print(f"t_ref = {t_ref_fit:.8f} +/- {sig_tref:.8e} BTJD")
print(f"P     = {P_fit:.8f} +/- {sig_P:.8e} days")
print(f"O-C WRMS = {wrms:.3f} min")
print(f"chi2(O-C=0) = {chi2_zero:.2f} for dof={dof_zero}")
# Plot O-C vs N
plt.figure(figsize=(8, 4))
plt.axhline(0, color="k", lw=1)
plt.errorbar(N_arr, oc_min, yerr=oc_err_min, fmt="o", capsize=2, color="tab:blue")
plt.xlabel("Transit number N")
plt.ylabel("O - C (minutes)")
plt.title("O-C diagram with MCMC timing errors")
plt.tight_layout()
plt.savefig("OC_vs_N.png", dpi=150)
plt.close()
# Plot O-C vs time
plt.figure(figsize=(8, 4))
plt.axhline(0, color="k", lw=1)
plt.errorbar(t_calc, oc_min, yerr=oc_err_min, fmt="o", capsize=2, color="tab:orange")
plt.xlabel("Calculated mid-transit time (BTJD)")
plt.ylabel("O - C (minutes)")
plt.title("O-C vs time")
plt.tight_layout()
plt.savefig("OC_vs_time.png", dpi=150)
plt.close()
# Save table
out = np.column_stack([
    N_arr,
    t_calc,
    t_obs,
    sig_t,
    oc_days,
    oc_err_days,
    oc_min,
    oc_err_min
])
header = (
    "N,t_calc_btjd,t_obs_btjd,sigma_tobs_days,"
    "oc_days,oc_err_days,oc_min,oc_err_min"
)
np.savetxt("ttv_results.csv", out, delimiter=",", header=header, comments="")
print("Saved: ttv_results.csv, OC_vs_N.png, OC_vs_time.png")