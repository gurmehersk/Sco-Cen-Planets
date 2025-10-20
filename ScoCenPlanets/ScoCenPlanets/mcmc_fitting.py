### This script is to create the mcmc for the transits of our 
### planet candidates


### transit depths are measured in percentages. We care about the size of the transit depth. but the parameter we 
### can measure, just the depth goes as the square of the size. log spreads it out, so it is numerically easier
### for samplers to sample in the logs. So for R/Rp, make it a log()

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.timeseries import LombScargle
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os
from wotan import flatten
from glob import glob
from transitleastsquares import transitleastsquares 
from numpy import array as nparr, all as npall, isfinite as npisfinite
import numpy.lib.recfunctions as rfn
from collections import OrderedDict
from copy import deepcopy
from astropy.timeseries import BoxLeastSquares 
import lightkurve as lk
from os.path import join
import pickle

### For exoplanet package transit fitting & mcmc
import pymc as pm
import exoplanet as xo
import pytensor.tensor as pt
import arviz as az

def clipit(data, low, high, method, center):
    """Clips data in the current segment"""

    # For the center point, take the median (if center_code==0), else the mean
    if center == 'median': # earlier, this was center == 'mad', changed this on 25th sept
        mid = np.nanmedian(data)
    else:
        mid = np.nanmean(data)
    data = np.nan_to_num(data)
    diff = data - mid

    if method == 'mad': # earlier, this was method == 'median', clipping doesn't seem to be working. changed this on 25th sept
        cutoff = np.nanmedian(np.abs(data - mid))
    else:
        cutoff = np.nanstd(data)

    # Clip values high (low) times the threshold (in MAD or STDEV)
    data[diff > high * cutoff] = np.nan
    data[diff < -low * cutoff] = np.nan
    return data

def slide_clip(time, data, window_length, low=3, high=3, method=None, center=None):
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

def clip_masks(sap_flux):
    '''MASK CREATION FOR CLIPPING'''
    q1 = np.nanpercentile(sap_flux, 25)
    q3 = np.nanpercentile(sap_flux, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    mask = sap_flux < upper_bound
    return q1, q3, iqr, mask

def Lombscargle(time, flux):
    frequency_PDCSAP, power_PDCSAP = LombScargle(time, flux).autopower()
    mask2 = frequency_PDCSAP < 20
    frequency_PDCSAP = frequency_PDCSAP[mask2]
    power_PDCSAP = power_PDCSAP[mask2]
    best_frequency_PDCSAP = frequency_PDCSAP[np.argmax(power_PDCSAP)]
    best_period_PDCSAP = 1 / best_frequency_PDCSAP
    return best_period_PDCSAP

def bin_lightcurve(time, flux, bin_minutes=30):
    '''BINNING TO 30 MINUTES'''
    bin_size = bin_minutes / (24 * 60)  # minutes to days
    bins = np.arange(np.nanmin(time), np.nanmax(time) + bin_size, bin_size)
    digitized = np.digitize(time, bins)

    binned_time = []
    binned_flux = []

    for i in range(1, len(bins)):
        bin_time = time[digitized == i]
        bin_flux = flux[digitized == i]
        if len(bin_time) > 0:
            binned_time.append(np.nanmean(bin_time))
            binned_flux.append(np.nanmean(bin_flux))

    return np.array(binned_time), np.array(binned_flux)

def clean_arrays(time, flux):
    mask = (~np.isnan(time)) & (~np.isnan(flux))
    return time[mask], flux[mask]
    
tic_id = 88297141
path = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025099153000-s0091-0000000088297141-0288-s/tess2025099153000-s0091-0000000088297141-0288-s_lc.fits"

hdu_list = fits.open(path)
hdr = hdu_list[0].header
data = hdu_list[1].data
time1 = data['TIME']
tessmag = hdr.get('TESSMAG', 'N/A')
tempeff = hdr.get('TEFF', 'N/A')

path2 = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025127075000-s0092-0000000088297141-0289-s/tess2025127075000-s0092-0000000088297141-0289-s_lc.fits"
hdu_list2 = fits.open(path2)
data2 = hdu_list2[1].data
time2 = data2['TIME']

pdcsap_flux1 = data['PDCSAP_FLUX']
pdcsap_flux2 = data2['PDCSAP_FLUX']

normalized = data['PDCSAP_FLUX']/np.nanmedian(data['PDCSAP_FLUX'])
normalized2 = data2['PDCSAP_FLUX']/np.nanmedian(data2['PDCSAP_FLUX'])

#normalized3 = data3['PDCSAP_FLUX']/np.nanmedian(data3['PDCSAP_FLUX'])
time = np.concatenate((time1, time2))# time3
pdcsap_flux = np.concatenate((normalized, normalized2))#, data3['PDCSAP_FLUX']))

time, pdcsap_flux = clean_arrays(time, pdcsap_flux)

rotational_period = Lombscargle(time, pdcsap_flux)

wdwl = 0.15 * rotational_period

clipped_flux = slide_clip(
            time, pdcsap_flux, window_length=1,
            low=100, high=2, method='mad', center='median') 

time, pdcsap_flux = clean_arrays(time, 1.*clipped_flux)

assert len(time) == len(pdcsap_flux)

time_bin, pdcsap_flux_bin = bin_lightcurve(time, pdcsap_flux)
assert len(time_bin)  == len(pdcsap_flux_bin)

tessmag = hdr.get('TESSMAG', 'N/A')
tempeff = hdr.get('TEFF', 'N/A')
qual = data['QUALITY']
bkgd = data['SAP_BKG']

print(f"TIC {tic_id}")
print("Shape of time1:", time1.shape)
print("Shape of time2:", time2.shape)
print("Shape of time:", time.shape)
print("Shape of flux:" , pdcsap_flux.shape)

flatten_lc, trend_lc = flatten(time, pdcsap_flux, window_length = wdwl, return_trend = True, method = 'biweight')

time_clean, flatten_clean = clean_arrays(time,flatten_lc)


### And now we have a clean time array and flattened light curve array. We can now proceed to do mcmc sampling and 
### fit transits!


### recovered parameters 
t0 = 3803.23
p = 4.6370567442246955

import pymc as pm
import pytensor.tensor as pt
import exoplanet as xo
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# --- Inputs (fill in your data) ---
# time_clean: time array
# flatten_clean: normalized flux array
# t0, p: initial guesses for transit time and period



### standard practice is to 

### for period --> +/- 1 or few days , you want an informative prior. A uniform prior +/- 10% of the period
### periodogram might not be a 100% accurate --> allow for such stuff. We want the uncertainties on the period and 
with pm.Model() as transit_model:

    ### Priors

    # Time of transit center
    ## for the ephermerises, in the future, here's where the transit will happen.
    ### order of magnitude of sigma should be 10-^-1
    t0_var = pm.Normal("t0", mu=t0, sigma=0.1) ### 0.1

    # Orbital period
    # .01 days --> don't want a prior that biases the uncertainties that we get.
    ### include the errors and the uncertainties, etc.
    period_var = pm.Normal("period", mu=p, sigma=0.01)

    # Radius ratio (Rp/Rs)
    log_ror_var = pm.Uniform("log_ror", lower=np.log(0.01), upper=np.log(0.3))
    ror_var = pm.Deterministic("ror", pt.exp(log_ror_var)) # this is done to ensure that the impact parameter is correctly 
    # sampled within the limits we create. Also, this was recommende by copilot [lol]

    # Impact parameter
    #### from 0 to 1 + R
    b_var = pm.Uniform("b", lower=0.0, upper=1.0+(pt.exp(log_ror_var)))

    ### currently, if I simply try b_var = pm.Uniform("b", lower=0.0, upper=1.0+(ror_var)), it gives an error
    ### can't have upper limit depend on another variable [like it isnt a number, its a pymc tensor]. So we do the following:
    # b_var = pm.Uniform("b", lower=0.0, upper= 1.0 + pt.exp(log_ror_var))

    # Stellar density (ρ*, g/cm³)
    rho_star_var = pm.Normal("rho_star", mu=1.0, sigma=0.5)

    # Limb darkening coefficients (quadratic)
    u_var = xo.distributions.QuadLimbDark("u")

    # Orbit
    orbit_var = xo.orbits.KeplerianOrbit(
        period=period_var,
        t0=t0_var,
        b=b_var,
        rho_star=rho_star_var
    )

    # Light curve model
    lc_model = xo.LimbDarkLightCurve(u_var).get_light_curve(
        orbit=orbit_var, r=ror_var, t=time_clean
    )
    lc_model = pt.sum(lc_model, axis=-1)

    # Flux offset
    mean_offset_var = pm.Normal("mean_offset", mu=1.0, sigma=0.1)
    model_flux_var = lc_model + mean_offset_var

    # White noise model
    flux_sigma_var = pm.HalfNormal("flux_sigma", sigma=1e-2)

    # Likelihood
    pm.Normal("obs", mu=model_flux_var, sigma=flux_sigma_var, observed=flatten_clean)

    # --- Sampling ---
    trace_transit = pm.sample(
        tune=2000,
        draws=2000,
        target_accept=0.9,
        return_inferencedata=True,
        cores=2
    )

# --- Post-processing ---
az.plot_trace(trace_transit)
az.summary(trace_transit)

# --- Extract posterior medians ---
posterior = trace_transit.posterior

period_med = posterior["period"].median().item()
t0_med = posterior["t0"].median().item()
ror_med = posterior["ror"].median().item()
b_med = posterior["b"].median().item()
rho_star_med = posterior["rho_star"].median().item()
u_med = [posterior["u"][:, :, i].median().item() for i in range(2)]
mean_offset_med = posterior["mean_offset"].median().item()

# --- Build best-fit model ---
orbit_best = xo.orbits.KeplerianOrbit(
    period=period_med,
    t0=t0_med,
    b=b_med,
    rho_star=rho_star_med
)
ld_model = xo.LimbDarkLightCurve(u_med)
model_lc = ld_model.get_light_curve(
    orbit=orbit_best, r=ror_med, t=time_clean
).eval().flatten() + mean_offset_med


time_clean_bin, flatten_clean_bin = bin_lightcurve(time_clean, flatten_clean)
# --- Plot ---
plt.figure(figsize=(10, 6))
plt.scatter(time_clean_bin, flatten_clean_bin, s=5, color="k", alpha=0.3, label="Data")
plt.plot(time_clean, model_lc, color="C1", label="Median Model")
plt.xlabel("Time")
plt.ylabel("Normalized Flux")
plt.legend()
plt.savefig("mcmc_10_20_TIC88297141.pdf", bbox_inches="tight")
plt.close()

import corner, numpy as np, matplotlib.pyplot as plt

posterior = trace_transit.posterior
samples = np.vstack([
    posterior["t0"].values.flatten(),
    posterior["period"].values.flatten(),
    posterior["ror"].values.flatten(),
    posterior["b"].values.flatten(),
    posterior["rho_star"].values.flatten()
]).T

labels = [r"$t_0$", r"$P$", r"$R_p/R_s$", r"$b$", r"$\rho_\star$"]

fig = corner.corner(samples, labels=labels, show_titles=True, title_fmt=".5f")
fig.savefig("corner_plot_20_11_TIC88297141.pdf", bbox_inches="tight")

plt.show()
plt.close()

