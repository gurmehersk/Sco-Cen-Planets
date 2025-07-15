import os
import numpy as np
from glob import glob
from astropy.io import fits
from lightkurve import search_lightcurve
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from matplotlib.backends.backend_pdf import PdfPages
from wotan import flatten
from transitleastsquares import transitleastsquares
import pickle 

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


sector_number = 89 # sector number can be changed 
lcpaths = glob(f"/ar1/TESS/SPOC/s00{sector_number}/*.fits") # accesses all the fits files in whatever path you've stored them in

failed_tics_path = f"/home/gurmeher/gurmeher/TLSperiodogram/sector_{sector_number}/failed_tics.txt"

# Load failed TICs into a set for quick lookup
if os.path.exists(failed_tics_path):
    with open(failed_tics_path, 'r') as f:
        failed_tics = set(line.strip() for line in f)
else:
    failed_tics = set()

for lcpath in lcpaths:
    #print(lcpath)
    hdu_list = fits.open(lcpath)
    hdr = hdu_list[0].header
    data = hdu_list[1].data
    hdu_list.close()
    tic_id = hdr.get('TICID', 'unknown') # automated tic id since we don't know what the tic would be, different form pipelinev0

    ''' CACHING TO IGNORE ANY ITERATIONS THAT HAVE ALREADY OCCURED ''' 

    pickle_path = f"/home/gurmeher/gurmeher/TLSperiodogram/sector_{sector_number}/TIC_{tic_id}.pkl" # change this to a new path 
    if os.path.exists(pickle_path) or str(tic_id) in failed_tics: # trying to cache the failed ones as well
        print(f"Skipping TIC {tic_id} — already cached.")
        continue

    ''' IF NOT ALREADY DONE, CONTINUE WITH THIS '''

    time = data['TIME']
    tessmag = hdr.get('TESSMAG', 'N/A')
    tempeff = hdr.get('TEFF', 'N/A')
    sap_flux = data['SAP_FLUX']
    pdcsap_flux = data['PDCSAP_FLUX']
    qual = data['QUALITY']
    bkgd = data['SAP_BKG'] # TODO : PLOT ME!


    sel = (qual == 0)
    time = time[sel]
    sap_flux = sap_flux[sel]
    pdcsap_flux = pdcsap_flux[sel]
    bkgd = bkgd[sel]


    q1 = np.nanpercentile(sap_flux, 25)
    q3 = np.nanpercentile(sap_flux, 75)
    iqr = q3 - q1

    # Define upper limit to clip flares
    upper_bound = q3 + 1.5 * iqr
    mask = sap_flux < upper_bound

    sap_time_binned, sap_flux_binned = bin_lightcurve(time[mask], sap_flux[mask]/np.nanmedian(sap_flux[mask]))
    pdc_time_binned, pdc_flux_binned = bin_lightcurve(time[mask], pdcsap_flux[mask]/np.nanmedian(pdcsap_flux[mask]))
    bkg_time_binned, bkg_flux_binned = bin_lightcurve(time[mask], bkgd[mask]/np.nanmedian(bkgd[mask]))

    ''' LOMBSCARGLE '''

    frequency_SAP, power_SAP = LombScargle(sap_time_binned, sap_flux_binned).autopower()
    frequency_PDCSAP, power_PDCSAP = LombScargle(pdc_time_binned, pdc_flux_binned).autopower()
    mask = frequency_PDCSAP < 20
    frequency_PDCSAP = frequency_PDCSAP[mask]
    power_PDCSAP = power_PDCSAP[mask]
    frequency_SAP = frequency_SAP[mask]
    power_SAP = power_SAP[mask]
    best_frequency_SAP = frequency_SAP[np.argmax(power_SAP)]
    best_period_SAP = 1 / best_frequency_SAP 
    best_frequency_PDCSAP = frequency_PDCSAP[np.argmax(power_PDCSAP)]
    best_period_PDCSAP = 1 / best_frequency_PDCSAP


    # phase = (time % period) / period
 
    sap_phase = (sap_time_binned % best_period_SAP)/best_period_SAP
    pdcsap_phase = (pdc_time_binned % best_period_PDCSAP)/best_period_PDCSAP

    wdwl = 0.3 * best_period_PDCSAP

    flatten_lc1, trend_lc1 = flatten(sap_time_binned, sap_flux_binned, window_length = wdwl, return_trend = True, method = 'biweight')
    flatten_lc2, trend_lc2 = flatten(pdc_time_binned, pdc_flux_binned, window_length = wdwl, return_trend = True, method = 'biweight')

    def clean_arrays(time, flux):
        mask = (~np.isnan(time)) & (~np.isnan(flux))
        return time[mask], flux[mask]

    sap_time_clean, flatten_lc1_clean = clean_arrays(sap_time_binned, flatten_lc1)
    pdc_time_clean, flatten_lc2_clean = clean_arrays(pdc_time_binned, flatten_lc2)


    if len(sap_time_clean) == 0 or len(flatten_lc1_clean) == 0:
        with open(f"/home/gurmeher/gurmeher/TLSperiodogram/sector_{sector_number}/failed_tics.txt", "a") as failfile:
            failfile.write(f"{tic_id}\n")
        #print(f"TIC {tic_id}: Empty SAP arrays for TLS, skipping.")
        continue

    if len(pdc_time_clean) == 0 or len(flatten_lc2_clean) == 0:
        with open(f"/home/gurmeher/gurmeher/TLSperiodogram/sector_{sector_number}/failed_tics.txt", "a") as failfile:
            failfile.write(f"{tic_id}\n")
        #print(f"TIC {tic_id}: Empty PDCSAP arrays for TLS, skipping.")
        continue

    if len(sap_time_clean) < 50 or (sap_time_clean.max() - sap_time_clean.min()) < 5:
        with open(f"/home/gurmeher/gurmeher/TLSperiodogram/sector_{sector_number}/failed_tics.txt", "a") as failfile:
            failfile.write(f"{tic_id}\n")
        #print(f"TIC {tic_id}: Not enough data points or too short time span for TLS. Skipping.")

        continue
    # I have commented the above check out because I'm actually now open to getting the ValueError. I have also now successfully cached.
    # I can now save the ones where i get a ValueError

    model1 = transitleastsquares(sap_time_clean, flatten_lc1_clean)
    model2 = transitleastsquares(pdc_time_clean, flatten_lc2_clean)

    min_period = 0.5  # days, or a bit more than your cadence
    max_period = (sap_time_clean.max() - sap_time_clean.min()) / 2  # maximum orbtial period is half baseline
    try:
        results1 = model1.power(period_min=min_period, period_max=max_period)
        results2 = model2.power(period_min=min_period, period_max=max_period)
        with open(pickle_path, 'wb') as f:
            pickle.dump(results2, f) # Only doing PDCSAP for this 
    except ValueError as e:
        print(f"TIC {tic_id}: TLS failed with error: {e}")
        with open(f"/home/gurmeher/gurmeher/TLSperiodogram/sector_{sector_number}/failed_tics.txt", "a") as failfile:
            failfile.write(f"{tic_id}\n")
        continue



    
