import numpy as np
import celerite2
from celerite2 import terms
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import lightkurve as lk
from astropy.io import fits
import emcee
import corner
import batman

### Just a few notes: we will be sigma clipping flares before GP fitting,
### recall that flares are not Guassian!

### sigma clipping light curve. This is a function from the WOTAN package, which is a package for detrending light curves.
#  We will use this function to clip flares from our light curve before fitting the GP.



def clipit(data, low, high, method, center):
    """Clips data in the current segment"""

    # For the center point, take the median (if center_code==0), else the mean
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

    # Clip values high (low) times the threshold (in MAD or STDEV)
    data[diff > high * cutoff] = np.nan
    data[diff < -low * cutoff] = np.nan
    return data

def slide_clip(time, data, window_length, low=3, high=3, method=None, center=None): # FUNCTION IS FROM WOTAN PACKAGE 

    """Sliding time-windowed outlier clipper.

    Parameters
    ----------
    time : array-like
        Time values
    flux : array-like
        Flux values for every time point
    window_length : float
        The length of the filter window in units of ``time`` (usually days)
    low : float or int
        Lower bound factor of clipping. Default is 3.
    high : float or int
        Lower bound factor of clipping. Default is 3.
    method : ``mad`` (median absolute deviation; default) or ``std`` (standard deviation)
        Outliers more than ``low`` and ``high`` times the ``mad`` (or the ``std``) from
        the middle point are clipped
    center : ``median`` (default) or ``mean``
        Method to determine the middle point

    Returns
    -------

    clipped : array-like
        Input array with clipped elements replaced by ``NaN`` values.
    """

    if method is None:
        method = 'mad'
    if center is None:
        center = 'median'

    low_index = np.min(time)
    hi_index = np.max(time)
    idx_start = 0
    idx_end = 0
    size = len(time)
    half_window = window_length / 2
    clipped_data = np.full(size, np.nan)
    for i in range(size-1):
        if time[i] > low_index and time[i] < hi_index:
            # Nice style would be:
            #   idx_start = numpy.argmax(time > time[i] - window_length/2)
            #   idx_end = numpy.argmax(time > time[i] + window_length/2)
            # But that's too slow (factor 10). Instead, we write:
            while time[idx_start] < time[i] - half_window:
                idx_start += 1
            while time[idx_end] < time[i] + half_window and idx_end < size-1:
                idx_end += 1
            # Clip the current sliding segment
            clipped_data[idx_start:idx_end] = clipit(
                data[idx_start:idx_end], low, high, method, center)
    return clipped_data

def load_tess_lc(lcpath):
    with fits.open(lcpath) as hdul:
        time = hdul[1].data['TIME']
        flux = hdul[1].data['PDCSAP_FLUX']
        ferr = hdul[1].data['PDCSAP_FLUX_ERR']
    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(ferr)
    return time[mask], flux[mask], ferr[mask]

# sector91
lcpath1 = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025099153000-s0091-0000000088297141-0288-s/tess2025099153000-s0091-0000000088297141-0288-s_lc.fits"

# sector92
lcpath2 = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025127075000-s0092-0000000088297141-0289-s/tess2025127075000-s0092-0000000088297141-0289-s_lc.fits"


t1, flux1, ferr1 = load_tess_lc(lcpath1)
t2, flux2, ferr2 = load_tess_lc(lcpath2)

## lowkey no need to do all this for error, but let's do it anyways

# first we normalize
flux1_norm = flux1 / np.nanmedian(flux1)
flux2_norm = flux2 / np.nanmedian(flux2)
ferr1_norm = ferr1 / np.nanmedian(flux1)
ferr2_norm = ferr2 / np.nanmedian(flux2)

# --- Stitch ---
t_full    = np.concatenate([t1, t2])
flux_full = np.concatenate([flux1_norm, flux2_norm])
ferr_full = np.concatenate([ferr1_norm, ferr2_norm])

idx       = np.argsort(t_full)
t_full    = t_full[idx]
flux_full = flux_full[idx]
ferr_full = ferr_full[idx]

# --- Sigma clip (aggressive upward for flares, conservative downward for transits) ---
flux_clipped = slide_clip(
    t_full, flux_full.copy(),   # .copy() because slide_clip modifies in place
    window_length=0.5,
    low=100, ## we do not want to even accidentally clip anything at the bottom here, so playing it safe
    high=2.5, ## rather aggressive, but need this for active stars 
    method='mad',
    center='median'
)

# --- Remove NaNs left by clipping ---
mask      = np.isfinite(flux_clipped)
t_clean   = t_full[mask]
flux_clean = flux_clipped[mask]
ferr_clean = ferr_full[mask]

print(f"Points before clipping: {len(t_full)}")
print(f"Points after clipping:  {len(t_clean)}")
print(f"Points removed:         {len(t_full) - len(t_clean)}")

## 

T0_KNOWN = 3803.23  # T0 for TESS
PER_KNOWN = 4.6370  # days
ROT_KNOWN = 1.8099  # days

plt.figure(figsize=(14, 4))
plt.plot(t_full, flux_full, 'k.', ms=1, alpha=0.4, label='raw')
plt.plot(t_clean, flux_clean, 'r.', ms=1, alpha=0.5, label='clipped')
plt.xlabel("Time (BTJD)")
plt.ylabel("Normalised Flux")
plt.legend()
plt.tight_layout()
plt.savefig("clipping_check.png", dpi=300)
plt.close()
print("Saved: clipping_check.png")

print(f"Sector 91 amplitude: {flux1_norm.max() - flux1_norm.min():.4f}")
print(f"Sector 92 amplitude: {flux2_norm.max() - flux2_norm.min():.4f}")
print(f"Sector 91 median: {np.nanmedian(flux1_norm):.6f}")
print(f"Sector 92 median: {np.nanmedian(flux2_norm):.6f}")


### Transit model time --> we will use BATMAN 


### TDL: we might want to model/fit the impact parameter and stuff, this might be something to look into mate.
def transit_model(t, t0, per, rp, a, inc, u1, u2):
    p            = batman.TransitParams()
    p.t0         = t0
    p.per        = per
    p.rp         = rp           # Rp/R*
    p.a          = a            # a/R*
    p.inc        = inc          # degrees
    p.ecc        = 0.0
    p.w          = 90.0
    p.u          = [u1, u2]     # quadratic limb darkening
    p.limb_dark  = "quadratic"
    return batman.TransitModel(p, t).light_curve(p)


### Now here comes, hopefully, the magic. We are going to use a QuasiPeriodic Kernel to model the stellar variability.

# -------------------------------------------------------
# 4. GP KERNEL — Quasi-Periodic Cosine (RotationTerm)
#
# RotationTerm is the QPC kernel from Aigrain et al.:
#   k(tau) = A^2 * exp(-|tau|/rho) * cos(2*pi*tau/P)
# plus its first harmonic (for spot + faculae structure).
#
# Parameters:
#   sigma  : overall amplitude
#   period : stellar rotation period
#   Q0     : quality factor (damping — higher = more coherent oscillation)
#   dQ     : secondary peak quality offset
#   f      : fractional amplitude of harmonic peak
# -------------------------------------------------------

## recall that in the tutorial, we were using 2 independent SHOTerm terms to model the variability.
## while this is okay for general use case, for stellar variability, we know that variability is
## quasi periodic, so we can use a more specific kernel that celerite already has implemented, the RotationTerm, 
# which is a quasi-periodic cosine kernel. This kernel is designed to model the kind of variability we expect 
# from rotating stars with spots.


def build_gp(gp_params, t, ferr):
    mean, log_sigma, log_period, log_Q0, log_dQ, log_f, log_jit = gp_params
    kernel = terms.RotationTerm(
        sigma=np.exp(log_sigma),
        period=np.exp(log_period),
        Q0=np.exp(log_Q0),
        dQ=np.exp(log_dQ),
        f=np.exp(log_f),
    )
    ### So, just to summarize what just happened above
    '''
    We have 7 GP Parameters being fit for:

    mean: baseline level flux
    log_sigma: amplitude of variability
    log_period: stellar rotation period [we have a good grasp of this]
    log_Q0: quality factor (coherence of oscillation)
    log_dQ: harmonic peak shape
    log_f : harmonic fraction 
    log_jit: additional white noise floor (jitter)

    5 of them, exclusing log_jit and mean are actually inherently part of
    the actual RotationTerm kernel parameters. Then mean and log_jit are 
    extras that aren't part of the kernel itself but are part of the GP setup. 
    The mean is just the baseline and jitter adds a small extra diagonal to the 
    covariance matrix to soak up any white noise the kernel doesn't account for.

    Also note, that everything is being fit in logspace, because these parameters
    are all positive and can vary over orders of magnitude, so fitting in log space
    is more stable and efficient.

    '''
    gp = celerite2.GaussianProcess(kernel, mean=mean)
    gp.compute(t, diag=ferr**2 + np.exp(log_jit), quiet=True)
    return gp

# log probability function for MCMC (likelihood+priors)

def log_prob(params, t, flux, ferr):

    ### we are splitting the matrix vector into transit params and Gp params
    t0, per, rp, a, inc, u1, u2 = params[:7]
    gp_params = params[7:]

    # --- Transit priors ---
    # Tight on t0 and period since we know them

    '''
    TDL: fold the T0_known value here 
    '''
    if not (T0_KNOWN  - 0.1  < t0  < T0_KNOWN  + 0.1):  return -np.inf 
    ### okay i need to modify T0 somehow, fold it or something because the sampler might not 
    # catch the t0  we got in this  stitched setup. 

    if not (PER_KNOWN - 0.1 < per < PER_KNOWN + 0.1):  return -np.inf
    if not (0.01 < rp  < 0.5):                            return -np.inf
    if not (1.0  < a   < 100.0):                          return -np.inf
    if not (60.0 < inc < 90.0):                           return -np.inf
    if not (0.0  < u1  < 1.0):                            return -np.inf
    if not (0.0  < u2  < 1.0):                            return -np.inf
    if u1 + u2 > 1.0:                                     return -np.inf

    # --- Wide Gaussian prior on GP hyperparams ---
    gp_prior = -0.5 * np.sum((gp_params / 2.0) ** 2)

    try:
        flux_transit = transit_model(t, t0, per, rp, a, inc, u1, u2)
        residuals    = flux - flux_transit      # GP models the residuals, not the raw flux
        gp           = build_gp(gp_params, t, ferr)
        return gp.log_likelihood(residuals) + gp_prior
    except Exception:
        return -np.inf
    


