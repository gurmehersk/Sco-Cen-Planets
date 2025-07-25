import matplotlib
matplotlib.use("AGG")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.timeseries import LombScargle
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os
from glob import glob
from transitleastsquares import transitleastsquares 
from numpy import array as nparr, all as npall, isfinite as npisfinite
import numpy.lib.recfunctions as rfn
from collections import OrderedDict
from copy import deepcopy

# 24th July
# remove flares

# TLS not giving correct period ---> fixed by fixing window length
# thing is, here, I can't even try to run the notch filter on the unbinned data cuz that will take too much time. So the detrending
# which is essentially happening on the binned data, is the only thing we can apply the flasttening to, so there is loss of SNR!

# I don't get why we are trying to do this method now frankly

# also, it doesn't seem to be too robust, I don't get the hype 

# ADD LS periodogram to be consistent about window length!

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

def Lombscargle(time, flux):
    frequency_PDCSAP, power_PDCSAP = LombScargle(time, flux).autopower()
    mask2 = frequency_PDCSAP < 20
    frequency_PDCSAP = frequency_PDCSAP[mask2]
    power_PDCSAP = power_PDCSAP[mask2]
    best_frequency_PDCSAP = frequency_PDCSAP[np.argmax(power_PDCSAP)]
    best_period_PDCSAP = 1 / best_frequency_PDCSAP
    return best_period_PDCSAP

def bin_lightcurve(time, flux, bin_minutes=30):
    bin_size = bin_minutes / (24 * 60)  # minutes to days
    bins = np.arange(time.min(), time.max() + bin_size, bin_size)
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

def _run_notch(TIME, FLUX, dtr_dict, verbose=False):

    from notch_and_locor.core import sliding_window

    #
    # HARD-CODE notch options (nb. could also let them be options via
    # dtr_dict).
    #
    # Set to True to do a full fit over time and arclength (default False).
    use_arclength = False
    # Internally, keep as "False" to use wdat.fcor as the flux.
    use_raw = False
    # BIC difference between transit and no-transit model required to
    # select the transit model
    min_deltabic = -1.0
    # By default (resolvabletrans == False), a grid of transit durations is
    # searched. [0.75, 1.0, 2.0, 4.0] hours.  If this is set to be True,
    # the 45 minute one is dropped.
    resolvable_trans = False
    # show_progress: if True, puts out a TQDM bar
    show_progress = verbose

    # Format "data" into recarray format needed for notch.
    N_points = len(TIME)
    data = np.recarray(
        (N_points,),
        dtype=[('t',float), ('fraw',float), ('fcor',float), ('s',float),
               ('qual',int), ('divisions',float)]
    )
    data.t = TIME
    data.fcor = FLUX
    data.fraw[:] = 0
    data.s[:] = 0
    data.qual[:] = 0

    # Run notch
    if verbose:
        LOGINFO('Beginning notch run...')
    
    fittimes, depth, detrend, polyshape, badflag = (
        sliding_window(
            data, windowsize=dtr_dict['window_length'],
            use_arclength=use_arclength, use_raw=use_raw,
            deltabic=min_deltabic, resolvable_trans=resolvable_trans,
            show_progress=show_progress
        )
    )
    print("depth[1] (deltabic) summary:")
    print("min:", np.nanmin(depth[1]))
    print("max:", np.nanmax(depth[1]))
    print("unique values:", np.unique(depth[1]))
    if verbose:
        LOGINFO('Completed notch run.')

    assert len(fittimes) == len(TIME)

    # Store everything in a common format recarray
    N_points = len(detrend)
    notch = np.recarray(
        (N_points, ), dtype=[
            ('t', float), ('detrend', float), ('polyshape', float),
            ('notch_depth', float), ('deltabic', float), ('bicstat', float),
            ('badflag', int)
        ]
    )

    notch.t = data.t
    notch.notch_depth = depth[0].copy()
    notch.deltabic    = depth[1].copy()
    notch.detrend     = detrend.copy()
    notch.badflag     = badflag.copy()
    notch.polyshape   = polyshape.copy()

    bicstat = notch.deltabic-np.median(notch.deltabic)
    notch.bicstat = 1- bicstat/np.max(bicstat)

    #
    # Convert to my naming scheme.
    #
    flat_flux = notch.detrend
    trend_flux = notch.polyshape

    return flat_flux, trend_flux, notch

tic_id = 441420236
# 166527623 , the hip star isnt working --> wrong period everytime
# 441420236, AU Mic b worked with this pipeline
# 460205581, TOI 837b, Luke's planet worked.
# 146520535 , not working

# Major point to note, the flattened flux method is much better than the deltabic thing which sometimes just doesnt work
path = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2020186164531-s0027-0000000441420236-0189-s/tess2020186164531-s0027-0000000441420236-0189-s_lc.fits"
pdfpath = f"/home/gurmeher/gurmeher/Notch_and_LOCoR/results/TIC_{tic_id}.pdf"

hdu_list = fits.open(path)
hdr = hdu_list[0].header
data = hdu_list[1].data
time = data['TIME']
tessmag = hdr.get('TESSMAG', 'N/A')
tempeff = hdr.get('TEFF', 'N/A')
pdcsap_flux = data['PDCSAP_FLUX']
qual = data['QUALITY']
bkgd = data['SAP_BKG'] # TODO : PLOT ME!
mask = np.isfinite(time) & np.isfinite(pdcsap_flux)
time_clean = time[mask]
flux_clean = pdcsap_flux[mask] / np.nanmedian(pdcsap_flux[mask])
pdc_time_binned, pdc_flux_binned = bin_lightcurve(time_clean, flux_clean)
#pdc_time_binned, pdc_flux_binned = bin_lightcurve(time, pdcsap_flux/np.nanmedian(pdcsap_flux))
#pdc_time_binned, pdc_flux_binned = bin_lightcurve(time, pdcsap_flux)
fig, axs = plt.subplots(nrows = 5, figsize = (6,10), sharex = False)
axs[0].scatter(pdc_time_binned, pdc_flux_binned, s = 0.5, zorder = 2)

prot = Lombscargle(pdc_time_binned, pdc_flux_binned)
cadence = np.nanmedian(np.diff(pdc_time_binned))
#dictionary = {"window_length" : 0.3*prot}
dictionary = {"window_length" : np.maximum(prot/20, cadence * 10)}
'''if prot > 2:
    dictionary = {"window_length" : 1}

else :
    dictionary = {"window_length" : 0.5} # window size 1 day'''

clipped_flux = slide_clip(
            pdc_time_binned, pdc_flux_binned, window_length=dictionary['window_length'],
            low=100, high=2, method='mad', center='median'
        )

sel = np.isfinite(pdc_time_binned) & np.isfinite(clipped_flux)
pdc_time_binned = pdc_time_binned[sel]
pdc_flux_binned = 1.* clipped_flux[sel]
assert len(pdc_time_binned) == len(pdc_flux_binned)
#print(f"Time length: {len(time)}, Flux length: {len(pdcsap_flux)}")
#print(f"Valid points: {np.sum(np.isfinite(pdcsap_flux))}")


#flat_flux, trend_flux, notch = _run_notch(pdc_time_binned, pdc_flux_binned/np.nanmedian(pdc_flux_binned), dictionary)
flat_flux, trend_flux, notch = _run_notch(pdc_time_binned, pdc_flux_binned/np.nanmedian(pdc_flux_binned), dictionary)

axs[1].scatter(pdc_time_binned , flat_flux, s = 0.5)
axs[0].plot(pdc_time_binned, trend_flux, color = 'red', linewidth = 1.5, zorder = 1)

delbic = notch.deltabic * -1
delbic = delbic/np.nanmedian(delbic)
axs[2].scatter(pdc_time_binned, delbic, color = 'pink', s = 0.5)
print(notch.deltabic)

print(f"pdc_time_binned: {pdc_time_binned}")
print(f"delbic: {delbic}")
print(f"len(pdc_time_binned): {len(pdc_time_binned)}")
print(f"Any NaNs in time? {np.any(np.isnan(pdc_time_binned))}")



if len(pdc_time_binned) == 0:
    raise ValueError("pdc_time_binned is empty!")
#model1 = transitleastsquares(pdc_time_binned, delbic)
model2 = transitleastsquares(pdc_time_binned, flat_flux)

min_period = 0.5  # days, or a bit more than your cadence
max_period = (pdc_time_binned.max() - pdc_time_binned.min()) / 2

#results1 = model1.power(period_min = min_period, period_max = max_period) # now inputting minimum and maximum period to try and fix valueError of empty TLS
results2 = model2.power(period_min = min_period, period_max = max_period)

#period1 = results1.period
period2 = results2.period

#sde1 = results1.SDE
sde2 = results2.SDE

#axs[3].scatter(results1.folded_phase, results1.folded_y, marker = 'o', s = 0.25, color = 'black', label = f'SAP phase-folded\nTLS Period = {period1:.4f} d\nSDE = {sde1:.2f}')
#axs[3].plot(results1.model_folded_phase, results1.model_folded_model, color = 'red', label = 'TLS MODEL for SAP Flux')
axs[4].scatter(results2.folded_phase, results2.folded_y, marker = 'o', s = 0.25, color = 'black', label = f'PDCSAP phase-folded\nTLS Period = {period2:.4f} d\nSDE = {sde2:.2f}')
axs[4].plot(results2.model_folded_phase, results2.model_folded_model, color = 'red', label = 'TLS MODEL for PDCSAP Flux')



for ax in axs:
    ax.legend()
print(prot)
with PdfPages(pdfpath) as pdf:
    pdf.savefig(fig, bbox_inches = 'tight')
    plt.close(fig)
