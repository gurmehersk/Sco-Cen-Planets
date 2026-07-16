'''
CORRECT WORKING CODE FOR GPR USING Quasi Periodic KERNEL


Duplicated from joint_fit.py 

Here, we will execute model 3

Recall that:

theta = [planet size, t0, period, impact parameter [0,1.1], ldc [u1,u2], 2xSHO QP Kernel, nuisane params + jitter..]

Model 1: with the theta as above, w/ GP everywhere, jitter: per instrument. Planet size free!! 

Model 2: theta as above, but for ground based, dont use GP: instead, use a second degree polynomial in time

Model 3; Let (Rp/R*) --> (Rp/R*)i, basically no longer a global param, rather localized... localize to each instrument/band pass --> WE DO THIS NOW IN A NEW SCRIPT BECAUSE THE PLOTTING IS GOING TO BE A BIT DIFFERENT

Model 4: further it to (epoch, bandpass) tuple


[GK] 16th July, 2026 --> NOTE TO AUTHORS:

IF YOU WANT TO RUN MODEL 2... PLEASE RUN JOINT_FIT.PY WITH THE RUN NUMBER SET TO 1 IF YOU JUST WANT TO GET NEW PLOTS OR SOMETHING

Now we are trying Model 4, letting Rp/R* float for each instrument epoch. 
'''

#### Run without bandpass specific limb darkening coefficients. Make it a global parameter. --> No need tbf.. [according to Luke]

from astropy.table import Table
import juliet
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import gridspec
import corner
import pandas as pd
import os


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
run_number      = 2    # for file naming — increment for each run with different settings
 
results_dir = os.path.join('results', f'run_v{run_number}')
os.makedirs(results_dir, exist_ok=True)

# -------------------------------------------------------
# KNOWN STELLAR / ORBITAL PARAMS
# -------------------------------------------------------
p   = 4.64423    # orbital period (days)
t0  = 3803.24126 # transit centre (BTJD)
rot = 1.8099     # stellar rotation period from Lomb-Scargle (days)

# -------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------

TESS_S91 = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025099153000-s0091-0000000088297141-0288-s/tess2025099153000-s0091-0000000088297141-0288-s_lc.fits" # straight up tess data
TESS_S92 = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025127075000-s0092-0000000088297141-0289-s/tess2025127075000-s0092-0000000088297141-0289-s_lc.fits" # tess data 
SWOPE_R = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/SWOPE_data/swope_8829.xls.csv"  # Luke's swope night data 
SSO_IP = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/SSO_data/TIC_88297141-22_20260425_LCO-SSO-1.0m_ip_5pix_measurements.tbl" # Khalid's SSO data (Karen SG1)
CTIO_Z = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/CTIO_data/TIC_88297141_LCOGT_CTIO_20260624_LGBrdx_z-band.csv" # Jerome observations 
CTIO_G = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/CTIO_data/TIC_88297141_LCOGT_CTIO_20260624_LGBrdx_g-band_C2_only.csv" # Jerome's June ctio observations 
MCD_IP = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/Mcdonald_data/TIC88297141-22_20260416_LCO-McD-1m0_ip_5px_KC_bjd-flux-err-detrended.dat" # Wilkins (KC SG1)
MCD_GP = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/Mcdonald_data/TIC88297141-22_20260416_LCO-McD-1m0_gp_4px_KC_bjd-flux-err-detrended.dat"  # Wilkins (KC SG1)
CTIO_apr_gp = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/CTIO_april_data/TIC88297141-22_20260416_LCO-CTIO-1m0_gp_4px_KC_bjd-flux-err-detrended.dat" # Wilkins.. (SG1)
CTIO_apr_ip = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/CTIO_april_data/TIC88297141-22_20260416_LCO-CTIO-1m0_ip_4px_KC_bjd-flux-err-detrended.dat" ### Wilkins and Glauk.. This is different than the above CTIO which was taken by Jerome's team in June (SG1)

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

## Add and normalize tess data 

t1, flux1, ferr1 = load_tess_lc(TESS_S91)
t2, flux2, ferr2 = load_tess_lc(TESS_S92)

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

## functions to load ground data 
def load_swope_lc(lcpath):
    df = pd.read_csv(lcpath)
    unconverted_time = df['BJD_TDB'].values 
    time = unconverted_time - 2457000.0  # Convert to BTJD
    flux = df['rel_flux_T1'].values 
    ferr = df['rel_flux_err_T1'].values
    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(ferr)
    ## normalize

    ferr = ferr / np.nanmedian(flux)
    flux = flux / np.nanmedian(flux)
    
    return time[mask], flux[mask], ferr[mask]

def load_SSO_lc(lcpath):
    t = Table.read(lcpath,format="ascii")
    unconverted_time = t['BJD_TDB'].data
    time = unconverted_time - 2457000.0 # convert to BTJD
    flux = t['rel_flux_T1'].data
    ferr = t['rel_flux_err_T1'].data
    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(ferr)

    ferr = ferr / np.nanmedian(flux)
    flux = flux / np.nanmedian(flux)
    

    return time[mask], flux[mask], ferr[mask]

def load_CTIO_lc(lcpath):
    '''

    works for Jerome's CTIO files.. could not just use 
    the SWOPE function because the comma separation 
    for these data, which are converted form xls is 
    a bit sketchy, so to be safe just created another 
    function.

    '''
    df = pd.read_csv(lcpath, sep = ",")
    unconverted_time = df['BJD_TDB'].values
    time = unconverted_time - 2457000.0  # convert to BTJD
    flux = df['rel_flux_T1'].values
    ferr = df['rel_flux_err_T1'].values
    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(ferr)
    ## normalize
    ferr = ferr / np.nanmedian(flux)
    flux = flux / np.nanmedian(flux)
    
    return time[mask], flux[mask], ferr[mask]

def load_MCD_CTIO_apr_lc(lcpath):
    '''
    Loading the MCD & CTIO files from april. They are both
    .dat files so there's some consistency in getting the
    correct formatting done.
    '''
    time, flux, ferr = np.loadtxt(lcpath, unpack = True)
    time -= 2457000.0 # convert to BTJD
    ## normalize

    ferr = ferr / np.nanmedian(flux)
    flux = flux/np.nanmedian(flux)
    

    # mask NaNs

    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(ferr)

    return time[mask], flux[mask], ferr[mask]

### Retrieving all the ground based data sequentially
### Note: these lc are alr normalized inside their respective functions

### Adding SWOPE data
t3, flux3, ferr3 = load_swope_lc(SWOPE_R) 

## Adding SSO data
t4, flux4, ferr4 = load_SSO_lc(SSO_IP)

## Adding CTIO Jerome
t5, ctio_z, ctio_z_err = load_CTIO_lc(CTIO_Z)
t6, ctio_g, ctio_g_err = load_CTIO_lc(CTIO_G)

# Adding MCD 
t7, mcd_g, mcd_g_err = load_MCD_CTIO_apr_lc(MCD_GP)
t8, mcd_i, mcd_i_err = load_MCD_CTIO_apr_lc(MCD_IP)

# Adding CTIO Wilkins
t9, ctio_apr_g, ctio_apr_g_err = load_MCD_CTIO_apr_lc(CTIO_apr_gp)
t10, ctio_apr_i, ctio_apr_i_err = load_MCD_CTIO_apr_lc(CTIO_apr_ip)

### now we dont just stitch this directly, we follow juliet protocol!! 

# -------------------------------------------------------
# 3. FOLD T0 INTO DATA WINDOW + IDENTIFY ALL TRANSITS --> Its okay not to do this for SWOPE

## we arent considered with the phase folded to include SWOPE & SSO, we only care about the 
## recovered transit parameters which will have the SWOPE & SSO data included 
# -------------------------------------------------------
N_orbits   = np.round((t_full.mean() - t0) / p)
T0_in_data = t0 + N_orbits * p

N_start = int(np.ceil( (t_full.min() - t0) / p))
N_end   = int(np.floor((t_full.max() - t0) / p))
all_T0s = t0 + np.arange(N_start, N_end + 1) * p

print(f"Data runs from {t_full.min():.2f} to {t_full.max():.2f} BTJD")
print(f"T0 folded into data window: {T0_in_data:.6f} BTJD")
print(f"Number of transits in data: {len(all_T0s)}")

print("SWOPE time range:", t3.min(), t3.max())
print("SWOPE flux range:", flux3.min(), flux3.max())
print("SWOPE ferr range:", ferr3.min(), ferr3.max())
print("Any NaN in t3:", np.any(np.isnan(t3)))
print("Any NaN in flux3:", np.any(np.isnan(flux3)))
print("Any NaN in ferr3:", np.any(np.isnan(ferr3)))
print("Any zero/negative errors:", np.any(ferr3 <= 0))
print("Number of SWOPE points:", len(t3))

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

###
## [28th APril 2026]
'''
Lets learn some things

We do not need a separate P, t0, r1, r2 for the swope and SSO data, since we are doing a joint fit. 
The transit parameters are shared between the two datasets, and the SWOPE & SSO data will help constrain those 
parameters better. These are therefore treated as "global" parameters in the juliet fit, meaning they are 
the same for both datasets. The only parameters that are dataset-specific (ie, "local") are the dilution factor, 
flux offset, and jitter, since these can differ between TESS and SWOPE due to different instruments and observational 
conditions. These parametres are " instrument-level nuisance parameters" — they describe not the planet, 
but the telescope and observing conditions.

mdilution — how much the transit depth is diluted by contaminating flux in the aperture (nearby stars etc.). 
Fixed at 1.0 for both here since you're assuming no contamination

mflux — a constant additive offset to put the out-of-transit baseline exactly at zero. Every instrument 
will have its own normalisation quirks, so each needs its own offset

sigma_w — extra white noise jitter on top of your formal error bars. Accounts for any underestimated errors or 
correlated noise that isn't captured by ferr.  TESS and SWOPE have completely different detectors and noise floors 
so these are completely independent.

Note, we do not add any GP params for SWOPE since the data is too short and at the same time
noisy to have any major stellar variability. Even if we did, it would just fit as linear, which
is probably something sigma_w can also fit as jitter, and mflux as offset.

WE might add it if the fit doesnt match our expectations.

FOR SSO, the run we are going to embark on rn [1st June], I will not be adding stellar variability to the fit,
cuz again, my theory is it can be fit by the offset since it might just be linear...
'''


### 16th July:
'''
Let's implement model 3

'''
base_params = [
    'P_p1',          # orbital period
    't0_p1',         # transit centre
    'b_p1',         # impact parameter  --> these first three parameters are common for all telescopes.. note: we have now removed the planet size from the global regime 
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
    'p_p1_TESS',     # depth  --> specific to TESS!!! 
]

ground_telescopes = ['SWOPE', 'SSO', 'CTIOz', 'CTIOg', 'mcdg','mcdi', 'ctioaprg', 'ctioapri'] ### instrument names should NOT have an __ in them! Juliet uses underscores as a method of identification!!! Be cautious 

### removing per instrument's q1 and q2, cuz we will try and fit these together... 
per_instrument = [
    'p_p1',         # NEW — local depth per ground instrument 
    'mdilution',
    'mflux',
    'sigma_w',
    'theta0',
    'theta1',
]

params = base_params.copy()

for telescope in ground_telescopes:
    params += [f'{p}_{telescope}' for p in per_instrument]


dists = [
    'normal',        # P_p1
    'normal',        # t0_p1
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
    'uniform',       # p_p1_TESS 
]


### on juliet, cannot separate the b and size priors for swope and tess cuz theyre fitted as global parameters 
### Change the prior stuff to impact parameter and Rp/Rs parameter...
hyperps = [
    [p, 0.01],           # P_p1 — tight Gaussian on known period
    [T0_in_data, 0.1],   # t0_p1 — tight Gaussian on folded T0
    [0., 1.2],            # b_p1
    [0., 1.],            # q1_TESS
    [0., 1.],            # q2_TESS
    0.0,                 # ecc_p1
    90.,                 # omega_p1
    [100, 50000],        # stellar density rho — loguniform, M dwarf range 
    1.0,                 # mdilution_TESS
    [0., 0.1],           # mflux_TESS
    [0.1, 1000.],        # sigma_w_TESS — jitter in ppm
    [1e-6, 1.0],         # GP_B_TESS — amplitude
    [0.0,  1.0],         # GP_C_TESS — harmonic complexity
    [0.5,  100.],        # GP_L_TESS — decay timescale
    [rot, 0.5],          # GP_Prot_TESS — Gaussian prior on known rotation period
    [0, 1],            # p_p1 --> changed this to uniform, instead of normal. Now, we give everything a free shot 
]

instrument_dists = [
    'uniform',
    'fixed',
    'normal',
    'loguniform',
    'uniform',
    'uniform',
]


## re-evaluate the g band contamination

mdilution_values ={
'SWOPE': 1.0 , 
'SSO': 0.985, 
'CTIOz': 1.0, 
'CTIOg': 0.99 , # for now
'mcdg': 0.97, # for now
'mcdi': 0.965, 
'ctioaprg' :0.98, # for now  
'ctioapri': 0.985}

for telescope in ground_telescopes:
    dists.extend(instrument_dists)
    hyperps.extend([0,1.0],
    [mdilution_values[telescope],
    [0., 0.1],
    [0.1, 10000.], ### this is a reasonable jitter to put on all the ground based data... High enough to account for noise in all cases... 
    [-1., 1.],
    [-1., 1.],])



# Limb darkening: prior shared within filter-matched groups,
# individual for telescopes with no filter partner (SWOPE, CTIOz)

# should work.. and if it does, we can use this same ideology and logic for the planet size and depth constraint... 
g_band_key = '_'.join(['CTIOg', 'mcdg', 'ctioaprg'])
i_band_key = '_'.join(['SSO', 'mcdi', 'ctioapri'])

solo_ld_telescopes = ['SWOPE', 'CTIOz']
for telescope in solo_ld_telescopes:
    params  += [f'q1_{telescope}', f'q2_{telescope}']
    dists   += ['uniform', 'uniform']
    hyperps += [[0., 1.], [0., 1.]]

params  += [f'q1_{g_band_key}', f'q2_{g_band_key}', f'q1_{i_band_key}', f'q2_{i_band_key}']
dists   += ['uniform', 'uniform', 'uniform', 'uniform']
hyperps += [[0., 1.], [0., 1.], [0., 1.], [0., 1.]]

# sanity check before building priors — catch length mismatches early
assert len(params) == len(dists) == len(hyperps), \
    f"mismatch: {len(params)} params, {len(dists)} dists, {len(hyperps)} hyperps"

# Populate priors dictionary in juliet format
for param, dist, hyperp in zip(params, dists, hyperps):
    priors[param] = {}
    priors[param]['distribution']   = dist
    priors[param]['hyperparameters'] = hyperp

## adding linear regressors (quadratic polynomial to the juliet fit)
t_bar_SWOPE = np.mean(t3)
t_bar_SSO = np.mean(t4)
t_bar_ctio_z = np.mean(t5)
t_bar_ctio_g = np.mean(t6)
t_bar_mcd_g = np.mean(t7)
t_bar_mcd_i = np.mean(t8)
t_bar_ctio_apr_g = np.mean(t9)
t_bar_ctio_apr_i = np.mean(t10)

lm_regressors = {
    'SWOPE': np.column_stack([t3 - t_bar_SWOPE, (t3 - t_bar_SWOPE)**2]),
    'SSO': np.column_stack([t4 - t_bar_SSO, (t4 - t_bar_SSO)**2]),
    'CTIOz': np.column_stack([t5 - t_bar_ctio_z, (t5 - t_bar_ctio_z)**2]),
    'CTIOg' : np.column_stack([t6 - t_bar_ctio_g, (t6 - t_bar_ctio_g)**2]), 
    'mcdg': np.column_stack([t7 - t_bar_mcd_g, (t7 - t_bar_mcd_g)**2]),
    'mcdi' : np.column_stack([t8 - t_bar_mcd_i, (t8 - t_bar_mcd_i)**2]),
    'ctioaprg': np.column_stack([t9 - t_bar_ctio_apr_g, (t9 - t_bar_ctio_apr_g)**2]),
    'ctioapri': np.column_stack([t10 - t_bar_ctio_apr_i, (t10 - t_bar_ctio_apr_i)**2]),
}

# -------------------------------------------------------
# 5. LOAD DATASET INTO JULIET
# -------------------------------------------------------
times        = {'TESS': t_clean, 'SWOPE': t3, 'SSO': t4, 'CTIOz': t5, 'CTIOg': t6, 'mcdg': t7,'mcdi': t8, 'ctioaprg' : t9 , 'ctioapri': t10}
fluxes       = {'TESS': flux_clean, 'SWOPE': flux3, 'SSO': flux4, 'CTIOz': ctio_z, 'CTIOg': ctio_g, 'mcdg': mcd_g,'mcdi': mcd_i, 'ctioaprg' : ctio_apr_g , 'ctioapri': ctio_apr_i }
fluxes_error = {'TESS': ferr_clean, 'SWOPE': ferr3, 'SSO': ferr4, 'CTIOz': ctio_z_err, 'CTIOg': ctio_g_err, 'mcdg': mcd_g_err, 'mcdi': mcd_i_err, 'ctioaprg' : ctio_apr_g_err , 'ctioapri': ctio_apr_i_err}

dataset = juliet.load(
    priors         = priors,
    t_lc           = times,
    y_lc           = fluxes,
    yerr_lc        = fluxes_error,
    GP_regressors_lc = {'TESS': t_clean},
    linear_regressors_lc=lm_regressors,
    out_folder     = os.path.join(results_dir, f'88297141_GP_QP_joint_SSO_SWOPE_v{run_number}'),
    verbose        = True
)

print(f"Fitting with dynesty using {number_of_cores} cores...")


# -------------------------------------------------------
# 6. FIT 
# -------------------------------------------------------

### For run number = 4, we are updating the dataset.fit for a more complex analysis.
### we are going to make the dlogz threshold 0.01, instead of the "default" version which runs 
# when add_live = True. For more info, import dynesty and type help(dynesty.NestedSampler.run_nested)

results = dataset.fit(use_dynesty=True, n_live_points = 750, dynesty_nthreads=number_of_cores, dlogz = 0.5, dynesty_sample='rslice') ## changing dlogz to 0.5 for exploratory run.. for actual investigation make it 0.01 
# -------------------------------------------------------
# 7. RESULTS
# -------------------------------------------------------
posterior_samples = results.posteriors['posterior_samples']

# -------------------------------------------------------
# Model 3: Rp/R* is localized per instrument/bandpass instead
# of being one global parameter. There is no single 'p_p1' key
# in posterior_samples anymore — instead there are 9 of them:
# 'p_p1_TESS' plus 'p_p1_<telescope>' for every ground telescope.
# depth_params collects all of them so the reporting, saving, and
# plotting steps below can loop over them instead of assuming a
# single global depth.
# -------------------------------------------------------
depth_params = ['p_p1_TESS'] + [f'p_p1_{telescope}' for telescope in ground_telescopes]

print("\n" + "="*60)
print("BEST-FIT PARAMETERS (Median ± 1σ)")
print("="*60)

# Save posterior samples + run metadata so results (or anything else)
# can be regenerated later WITHOUT re-running the fit.
import pickle

with open(os.path.join(results_dir, 'posterior_samples.pkl'), 'wb') as f:
    pickle.dump(posterior_samples, f)

run_metadata = {
    'run_number': run_number,
    'priors': priors,
    'ground_telescopes': ground_telescopes,
    'solo_ld_telescopes': solo_ld_telescopes,
    'g_band_key': g_band_key,
    'i_band_key': i_band_key,
    'depth_params': depth_params,
}
with open(os.path.join(results_dir, 'run_metadata.pkl'), 'wb') as f:
    pickle.dump(run_metadata, f)

params_to_report = {
    'P_p1':        'Period (days)',
    't0_p1':       'T0 (BTJD)',
    'b_p1':        'Impact parameter',
    'rho':         'Stellar density (kg/m³)',
    'GP_B_TESS':   'GP amplitude',
    'GP_C_TESS':   'GP harmonic complexity',
    'GP_L_TESS':   'GP decay timescale (days)',
    'GP_Prot_TESS':'GP rotation period (days)',
    'sigma_w_TESS':'Jitter',
}

# Localized Rp/R* — one entry per instrument/bandpass (model 3)
for dp in depth_params:
    band = dp.replace('p_p1_', '')
    params_to_report[dp] = f'Rp/R* [{band}]'

for param, label in params_to_report.items():
    med  = np.median(posterior_samples[param])
    lo   = np.percentile(posterior_samples[param], 16)
    hi   = np.percentile(posterior_samples[param], 84)
    print(f"  {label:35s}: {med:.6f} +{hi-med:.6f} -{med-lo:.6f}")


'''
# Derive p (Rp/R*) and b (impact parameter) from Espinoza r1, r2
r1 = posterior_samples['r1_p1']
r2 = posterior_samples['r2_p1']

### correcting the impact parameter and Rp/Rs parameters

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


# -------------------------------------------------------
# 8. PLOTS
# -------------------------------------------------------
# Extract model components
transit_plus_gp = results.lc.evaluate('TESS')
transit_model   = results.lc.model['TESS']['deterministic']
gp_model        = results.lc.model['TESS']['GP']
gp_corrected    = dataset.data_lc['TESS'] - gp_model

# Compute phases using best-fit period and t0
p_best  = np.median(posterior_samples['P_p1'])
t0_best = np.median(posterior_samples['t0_p1'])
phases  = juliet.utils.get_phases(dataset.times_lc['TESS'], p_best, t0_best)


### added this to save the gp stuff cuz im not able to access em
# Also stash the full posterior for each of the 9 localized depths
# (p_p1_TESS + one per ground telescope) so they can be reloaded and
# re-plotted / re-tabulated later without re-running the fit.
depth_samples = {dp: posterior_samples[dp] for dp in depth_params}

np.savez(
    os.path.join(results_dir, f'88297141_GP_QP_joint_v{run_number}_gp_results.npz'),
    time=dataset.times_lc['TESS'],
    flux=dataset.data_lc['TESS'],
    fluxerr=dataset.errors_lc['TESS'],
    gp=gp_model,
    deterministic=transit_model,
    full_model=transit_plus_gp,
    gp_corrected=gp_corrected,
    depth_param_names=np.array(depth_params),
    **depth_samples
)

### adding phase bins 
# Bin GP-corrected data in phase — 20 minute bins
bin_width   = 20. / (p_best * 24. * 60.)
bin_edges   = np.arange(-0.05, 0.05 + bin_width, bin_width)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

bin_flux = np.zeros(len(bin_centers))
bin_err  = np.zeros(len(bin_centers))
bin_mask = np.zeros(len(bin_centers), dtype=bool)

for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    in_bin = (phases >= lo) & (phases < hi)
    if in_bin.sum() > 0:
        w            = 1. / dataset.errors_lc['TESS'][in_bin]**2
        bin_flux[i]  = np.average(gp_corrected[in_bin], weights=w)
        bin_err[i]   = 1. / np.sqrt(np.sum(w))
        bin_mask[i]  = True


# --- Plot 1: full light curve + model ---

fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
ax1 = plt.subplot(gs[0])


### Let's also add binned data points! 

from scipy.stats import binned_statistic

binned_time_plot = dataset.times_lc['TESS']
binned_data_plot = dataset.data_lc['TESS']
binned_err_plot = dataset.errors_lc['TESS']

# 30 minutes = 30/(24*60) days
bin_width = 30.0 / (24.0 * 60.0)

bins = np.arange(binned_time_plot.min(), binned_time_plot.max() + bin_width, bin_width)

# Mean time and flux in each bin
bin_time_plot, _, _ = binned_statistic(binned_time_plot, binned_time_plot,
                                  statistic='mean', bins=bins)
bin_flux_plot, _, _ = binned_statistic(binned_time_plot, binned_data_plot,
                                  statistic='mean', bins=bins)

# Number of points per bin
counts, _, _ = binned_statistic(binned_time_plot, binned_data_plot,
                                statistic='count', bins=bins)

# Error of weighted mean (assuming independent measurements)
weights = 1/binned_err_plot**2
sumw, _, _ = binned_statistic(binned_time_plot, weights,
                              statistic='sum', bins=bins)
bin_err_plot = np.sqrt(1/sumw)

mask = (
    np.isfinite(bin_time_plot) &
    np.isfinite(bin_flux_plot) &
    np.isfinite(bin_err_plot)
)

bin_time_plot = bin_time_plot[mask]
bin_flux_plot = bin_flux_plot[mask]
bin_err_plot  = bin_err_plot[mask]

print(len(bin_time_plot), len(bin_flux_plot), len(bin_err_plot))

ax1.errorbar(dataset.times_lc['TESS'], dataset.data_lc['TESS'],
             yerr=dataset.errors_lc['TESS'], zorder = 1, color = '0.7', alpha = 0.15, fmt='.', label='TESS')

## putting error bars on the binned data wont look aesthetic 
ax1.scatter(bin_time_plot, bin_flux_plot, s = 2, marker = "o", color='dodgerblue',
    zorder=4,
    label='30 minute bins')

### Let's try and reduce the stray line drawn through the data gaps in the model... 
time_model_plot = dataset.times_lc['TESS']
model_plot = transit_plus_gp

gap = np.where(np.diff(time_model_plot) > 0.5)[0] + 1
segments = np.split(np.arange(len(time_model_plot)), gap)

for i, seg in enumerate(segments):
    ax1.plot(time_model_plot[seg], model_plot[seg],
             color='black', lw=1.5, zorder = 3,
             label='Transit + GP model' if i == 0 else None)

'''
ax1.plot(dataset.times_lc['TESS'], transit_plus_gp,
         color='black', lw=2, zorder=1, label='Transit + GP model') '''

ax1.set_xlabel('Time (BTJD)', fontsize=12)
ax1.set_ylabel('Relative flux', fontsize=12)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_title('Full light curve')

# --- Plot 2: phase-folded w/binned points ---
ax2 = plt.subplot(gs[1])
idx_sort = np.argsort(phases)
# Unbinned — background
ax2.errorbar(phases, gp_corrected,
             yerr=dataset.errors_lc['TESS'],
             fmt='.', color='steelblue', alpha=0.15,
             ms=2, elinewidth=0.5, zorder=2,
             label='GP-corrected data')

# Model — behind binned and regular data points
ax2.plot(phases[idx_sort], transit_model[idx_sort],
         color='black', lw=2, zorder=1, label='Transit model')


# Binned points — front
ax2.errorbar(bin_centers[bin_mask], bin_flux[bin_mask],
             yerr=bin_err[bin_mask],
             fmt='o', color='navy', ms=5,
             elinewidth=1.5, capsize=2, zorder=3,
             label='20-min bins')

ax2.set_xlabel('Phase', fontsize=12)
ax2.set_ylabel('Relative flux', fontsize=12)
ax2.set_xlim([-0.05, 0.05])
ax2.set_ylim([0.975, 1.015])
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_title('Phase-folded transit')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, f'88297141_GP_QP_fit_joint_v{run_number}.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: 88297141_GP_QP_fit_joint_v{run_number}.png")

'''

ax2 = plt.subplot(gs[1])
idx_sort = np.argsort(phases)

ax2.errorbar(phases, dataset.data_lc['TESS'] - gp_model,
             yerr=dataset.errors_lc['TESS'], fmt='.', alpha=0.3,
             label='GP-corrected data')
ax2.plot(phases[idx_sort], transit_model[idx_sort],
         color='black', lw=2, zorder=10, label='Transit model')
ax2.set_xlabel('Phase', fontsize=12)
ax2.set_ylabel('Relative flux', fontsize=12)
ax2.set_xlim([-0.05, 0.05])
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_title('Phase-folded transit')

plt.tight_layout()
plt.savefig(f'88297141_GP_QP_fit_joint_v{run_number}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: 88297141_GP_QP_fit_joint_v{run_number}.png")
'''


print("----------------------------------------------")
print("Check to see if everything ran the way we wanted it to")
print("----------------------------------------------")

print(results.posteriors['posterior_samples'].keys())



# --- Plot 3: corner plot (shared / global parameters only) ---
# Rp/R* (p_p1) is NOT included here — in model 3 it's no longer a
# single global parameter, it floats independently per instrument
# (see depth_params). Mixing a per-instrument depth into this corner
# plot isn't meaningful since there isn't one value to put on the axis.
# The dedicated depth corner plot below (Plot 3b) covers those.
params_corner = ['P_p1', 't0_p1', 'b_p1', 'rho',
                 'GP_B_TESS', 'GP_Prot_TESS']
labels_corner = ['Period (d)', 'T0 (BTJD)', 'b',
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
fig_corner.savefig(os.path.join(results_dir, f'88297141_corner_joint_v{run_number}.png'), dpi=300, bbox_inches='tight') 
plt.close()
print(f"Saved: 88297141_corner_joint_v{run_number}.png")

# --- Plot 3b: corner plot of the 9 localized Rp/R* values ---
# One panel per instrument/bandpass (TESS + the 8 ground telescopes),
# so you can see how the individually-fit depths compare/correlate.
depth_labels  = [dp.replace('p_p1_', '') for dp in depth_params]
samples_depth = np.array([posterior_samples[dp] for dp in depth_params]).T

fig_depth_corner = corner.corner(
    samples_depth,
    labels=[f'Rp/R*\n[{lbl}]' for lbl in depth_labels],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 8},
    label_kwargs={"fontsize": 9}
)
fig_depth_corner.savefig(os.path.join(results_dir, f'88297141_corner_depths_v{run_number}.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: 88297141_corner_depths_v{run_number}.png")

### Plotting ground based data with transit model:


### Bin in phase that is done above's helper function... this is kinda a redundant function but we can keep it to ease
### the work of the function below 


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

def get_phase_folded_data(instrument, phase_span, bin_width_minutes=20.):

    t_inst    = dataset.times_lc[instrument]
    data_inst = dataset.data_lc[instrument]
    err_inst  = dataset.errors_lc[instrument]

    full_model, up68, low68, components = results.lc.evaluate(
        instrument,
        return_err=True,
        return_components=True,
        all_samples=True
    )

    lm_component   = components['lm']
    pure_transit   = full_model - lm_component
    data_corrected = data_inst - lm_component

    phases = juliet.utils.get_phases(t_inst, p_best, t0_best)
    idx_sort = np.argsort(phases)

    bin_width = bin_width_minutes / (p_best * 24. * 60.)

    bin_centers, bin_flux, bin_err = bin_in_phase(
        phases,
        data_corrected,
        err_inst,
        bin_width,
        phase_range=(-phase_span, phase_span)
    )

    return {
        "phases": phases,
        "idx_sort": idx_sort,
        "model": pure_transit,
        "data": data_corrected,
        "err": err_inst,
        "bin_centers": bin_centers,
        "bin_flux": bin_flux,
        "bin_err": bin_err,
    }

all_phases = []

for instrument in ground_telescopes:
    phases = juliet.utils.get_phases(
        dataset.times_lc[instrument],
        p_best,
        t0_best
    )
    all_phases.append(phases)

all_phases = np.concatenate(all_phases)

phase_span = np.max(np.abs(all_phases)) * 1.05

import aesthetic.plot
aesthetic.plot.set_style("science")


fig, axes = plt.subplots(
    2,
    4,
    figsize=(18,8),
    sharex=False,
    sharey=False
)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

axes = axes.flatten()

for i, (ax, instrument) in enumerate(zip(axes, ground_telescopes)):

    d = get_phase_folded_data(
        instrument,
        phase_span,
        bin_width_minutes=20
    )

    # Transit model
    ax.plot(
        d["phases"][d["idx_sort"]],
        d["model"][d["idx_sort"]],
        color="black",
        lw=2.5,
        zorder=2
    )

    # Unbinned data
    ax.errorbar(
        d["phases"],
        d["data"],
        yerr=d["err"],
        fmt=".",
        color="steelblue",
        alpha=0.45,
        ms=3,
        elinewidth=0.7,
        zorder=3
    )

    # Binned data
    if len(d["bin_centers"]) > 0:
        ax.errorbar(
            d["bin_centers"],
            d["bin_flux"],
            yerr=d["bin_err"],
            fmt="o",
            color="navy",
            ms=6,
            capsize=2,
            elinewidth=1.5,
            zorder=4
        )

    if i % 4 == 0:
        ax.set_ylabel("Relative Flux")

    if i >= 4:
        ax.set_xlabel("Orbital Phase")

    pad = 0.002

    xmin = np.min(d["phases"]) - pad
    xmax = np.max(d["phases"]) + pad

    ax.set_xlim(xmin, xmax)
    ax.grid(alpha=0.3)
    ax.set_title(instrument, fontsize=11)


# Bottom row
for ax in axes[4:]:
    ax.set_xlabel("Orbital Phase")

plt.tight_layout()

plots_dir = os.path.join(results_dir, 'ground_based_plots')
os.makedirs(plots_dir, exist_ok=True)

outfile = os.path.join(
    plots_dir,
    "ground_phase_folded_mosaic.png"
)

plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {outfile}")

# Left column
for ax in axes[::4]:
    ax.set_ylabel("Relative Flux")


aesthetic.plot.set_style("science")
def plot_ground_instrument(instrument, save_dir, bin_width_minutes=20.):
    t_inst    = dataset.times_lc[instrument]
    data_inst = dataset.data_lc[instrument]
    err_inst  = dataset.errors_lc[instrument]
 
    # Documented juliet approach for linear-model (non-GP) instruments:
    # components['lm'] is the polynomial systematics term.
    full_model, up68, low68, components = results.lc.evaluate(
        instrument, return_err=True, return_components=True, all_samples=True)
 
    lm_component   = components['lm']
    pure_transit   = full_model - lm_component     # transit + dilution + oot flux, no trend
    data_corrected = data_inst - lm_component       # data with poly trend removed
 
    phases   = juliet.utils.get_phases(t_inst, p_best, t0_best)
    idx_sort = np.argsort(phases)
 
    # Phase window sized to this instrument's own transit coverage
    phase_span = max(np.abs(phases).max() * 1.1, 0.01)
    bin_width  = bin_width_minutes / (p_best * 24. * 60.)
    bin_centers, bin_flux, bin_err = bin_in_phase(
        phases, data_corrected, err_inst, bin_width,
        phase_range=(-phase_span, phase_span))
 
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
 
    # Panel 1: full light curve (transit + polynomial trend) -- model behind, data in front
    ax1 = plt.subplot(gs[0])
    ax1.plot(t_inst, full_model, color='black', lw=2, zorder=2,
              label='Transit + polynomial model')
    ax1.errorbar(t_inst, data_inst, yerr=err_inst, fmt='.', alpha=0.4,
                 color='steelblue', zorder=3, label='Data')
    ax1.set_xlabel('Time (BTJD)', fontsize=12)
    ax1.set_ylabel('Relative flux', fontsize=12)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_title(f'{instrument} -- full light curve')
 
    # Panel 2: phase-folded, detrended data -- model behind, data + bins in front
    ax2 = plt.subplot(gs[1])
    ax2.plot(phases[idx_sort], pure_transit[idx_sort], color='black', lw=2,
              zorder=2, label='Transit model')
    ax2.errorbar(phases, data_corrected, yerr=err_inst, fmt='.',
                 color='steelblue', alpha=0.3, ms=4, elinewidth=0.7,
                 zorder=3, label='Detrended data')
    if len(bin_centers) > 0:
        ax2.errorbar(bin_centers, bin_flux, yerr=bin_err, fmt='o',
                     color='navy', ms=5, elinewidth=1.5, capsize=2,
                     zorder=4, label=f'{int(bin_width_minutes)}-min bins')
    ax2.set_xlabel('Phase', fontsize=12)
    ax2.set_ylabel('Relative flux', fontsize=12)
    ax2.set_xlim([-phase_span, phase_span])
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax2.set_title(f'{instrument} -- phase-folded')
 
    plt.tight_layout()
    outpath = os.path.join(save_dir, f'{instrument}_fit_fixed_zorder.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")
 
# -------------------------------------------------------
# RUN
# -------------------------------------------------------
for telescope in ground_telescopes:
    plot_ground_instrument(telescope, plots_dir)


print("Done.")