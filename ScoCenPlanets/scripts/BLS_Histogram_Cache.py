# This is code to create a histogram from BLS, and to save all the data in pickle files
# If there are repeats, now save sector number + the tic_id

# Okay, let's try and save/cache as much as we can.

import pickle 
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.timeseries import LombScargle
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from transitleastsquares import transitleastsquares 
# Do not actually need to import TransitLeastSquares
from numpy import array as nparr, all as npall, isfinite as npisfinite
import numpy.lib.recfunctions as rfn
from collections import OrderedDict
from copy import deepcopy
from astropy.timeseries import BoxLeastSquares 
from glob import glob
import pickle 
import os
import re # regular expressions, regex101.com
from os.path import join

def sde_calc(power_spectrum, periods, window_width=0.1, harmonics=3):
    """
    We want to calculate the SDE while excluding peak period and its harmonics.
    
    The reason we want to calculate the SDE is because when i was running the SNR and likelihood implementation for
    88297141, the SNR, and likelihood were higher, but pretty close to the "no transit" star condition. This had me
    worried for quite a bit because it indicates that SNR and likelihood are not good indicators of transit detection.
    So, we want to calculate the SDE, which then showed me a stark difference in the values found on both. 

    Arguments:
        power_spectrum: The power spectrum which is either SNR or likelihood.
        frequencies : The PERIODS of the power spectrum. It is named frequencies due to the LS analogy.
        window_width: The width of the window to suppress peaks and harmonics. The window width is actually 
        the half width, since we are going +\- the window_width around the peak frequencies.
        harmonics: Number of harmonics to suppress for each peak, we always keep this as 3 because we want to surpress
        0.5,1 and 2 harmonics of the peak frequency. --> we aren't using the harmonics variable anywhere, but it's just 
        kept there to remind ourself that the multipliers column should have harmonics number always
        I realize that keeping the 2 harmonic is a bit redundant for high orbital period planets, but for lower ones 
        it makes sense.
    Returns:
        sde: The calculated SDE, to 2 significant figures


    TLS documentation mentions that the SDE is defined as:

        I want to take a moment to also discuss how i am calcualting this sde analog. I went through the TLS documentation, 
        where they use something called Spectral Ratios. TLS doesn't really build a power spectrum similar to BLS, it tries to
        minimize chi squared values of different transit fits at each trial period. I.E., at each trial period, it computes a chi2
        value, and then they define a "Spectral Ratio" as the ratio of the minimum chi2 value to the chi2 value at the trial period P.
        They then do SDE = ( 1 - np.mean(Spectral Ratio) ) / (np.nanstd(Spectral Ratio)). They are using this philosophy of the peak
        being 1 because of the way they defined the spectral ratios, and are saying: SDE = peak [here 1] - mean (without removing 
        the 1 peak though) / standard deviation of the spectral ratios without removing the peaks.

    Our Implementation:
    
        This seems a little less efficint because they are essentially doing what we are also trying, but we are trying to do
        peak of the power spectrum - median of power spectrum without the peaks / the MAD of the power spectrum without the peaks.
        This analog of the SDE seems to me to be more robust as it accounts the fact that the peaks would influence the "scatter." 
        which should NOT be the case. 

    """
    # Copy the power spectrum to avoid modifying the original
    modified_power_spectrum = power_spectrum.copy()

    # Find the index of the maximum peak
    peak_index = np.argmax(modified_power_spectrum)
    peak_power = modified_power_spectrum[peak_index]
    peak_period = periods[peak_index] # Get the period corresponding to the peak power

    multipliers = [0.5,1,2]
    # Just a check to ensure I never accidentally change the number of harmonics
    if len(multipliers) == harmonics:
        pass
    else:
        raise ValueError(f"Number of harmonics ({harmonics}) does not match the length of multipliers ({len(multipliers)}).")
    # Suppress the peak and its harmonics
    for harmonic in multipliers:
        harmonic_period = peak_period * harmonic
        # i am aware that the 2nd harmonic, i.e. multipliers = 2 is redundant for a lot of the higher orbital 
        # periods, especially in the true positives cases that I tested it with. However, considering how on average I was finding low 
        # orbital period planets, I think it is better to keep it in the code, so that we can suppress the 2nd harmonic.

        # Find indices within the window around the harmonic frequency
        # this is where I was stating that the window_width is actually the half width, since we are going +/-
        window_indices = np.where(
            (periods >= harmonic_period - window_width) &
            (periods <= harmonic_period + window_width)
        )[0]
        # Suppress the power in the window
        modified_power_spectrum[window_indices] = np.nan  # Exclude from calculations by making it a nan value 

    # Calculate the median power without the peaks, not including those that have now are nan
    # Additionally, wwe are making these changes to the modified_power_spectrum to avoid any reference pass that changes the 
    # original array 
    median_power = np.nanmedian(modified_power_spectrum)

    # Calculate the MAD (Median Absolute Deviation)
    mad_power = np.nanmedian(np.abs(modified_power_spectrum - median_power))

    # Calculate the SDE
    # I believe this analog calculation works? I think that's the formula usually used
    # It is peak - the median excluding the peak(s) / deviation of the power
    sde = (peak_power - median_power) / mad_power

    return sde

def count_points_per_window(time, window_length): # to do the data downlink cleaning 
    half_window = window_length / 2
    counts = []

    for t in time:
        in_window = (time >= t - half_window) & (time <= t + half_window)
        counts.append(np.sum(in_window))

    return np.array(counts)

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

def _run_notch(TIME, FLUX, dtr_dict, verbose=False): # KEEP VERBOSE FALSE because we dont have logging imported

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
    verbose and print("depth[1] (deltabic) summary:")
    verbose and print("min:", np.nanmin(depth[1]))
    verbose and print("max:", np.nanmax(depth[1]))
    verbose and print("unique values:", np.unique(depth[1]))
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

def clean_arrays(time, flux):
    mask = (~np.isnan(time)) & (~np.isnan(flux))
    return time[mask], flux[mask]
    
def phase_folder(sap_time_binned, best_period_SAP, pdc_time_binned, best_period_PDCSAP):
    sap_phase = (sap_time_binned % best_period_SAP)/best_period_SAP
    pdcsap_phase = (pdc_time_binned % best_period_PDCSAP)/best_period_PDCSAP
    return sap_phase, pdcsap_phase

def clip_masks(sap_flux):
    '''MASK CREATION FOR CLIPPING'''
    q1 = np.nanpercentile(sap_flux, 25)
    q3 = np.nanpercentile(sap_flux, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    mask = sap_flux < upper_bound
    return q1, q3, iqr, mask


def clean_data(qual, time, sap_flux, pdcsap_flux, bkgd):
    ''' CLEANING THE DATA '''
    sel = (qual == 0)
    time = time[sel]
    sap_flux = sap_flux[sel]
    pdcsap_flux = pdcsap_flux[sel]
    bkgd = bkgd[sel]
    return time, sap_flux, pdcsap_flux, bkgd


def get_data(data, hdr):
    ''' GETTING DATA FROM THE FITS FILES '''
    time = data['TIME']
    tessmag = hdr.get('TESSMAG', 'N/A')
    tempeff = hdr.get('TEFF', 'N/A')
    sap_flux = data['SAP_FLUX']
    pdcsap_flux = data['PDCSAP_FLUX']
    qual = data['QUALITY']
    bkgd = data['SAP_BKG'] # TODO : PLOT ME!
    return time, tessmag, tempeff, sap_flux, pdcsap_flux, qual, bkgd

def get_this_sectors_ticids(make_plots, sector_number, detrend, wdwstr):
    ''' EITHER CREATES PICKLE FILES OR DOES THE DATAVISUALIZATION PROCESS '''
    # NEED TO CHANGE THIS TO ACCOUNT FOR NOTCH IMPLEMENTATION 
    ticids = []
    if make_plots:
            highsdetic_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/notch_v2_Aug_2025/mainlc/sector{sector_number}/highsdetic.txt"
            with open(highsdetic_path, 'r') as f: # opening it in read only mode
                ticids = set(line.strip() for line in f if line.strip())
    else:
        # add if condition for wotan, else for notch --> done 
        lcpaths = glob(f"/ar1/TESS/SPOC/s00{sector_number}/*.fits") # accesses all the fits files in whatever path you've stored them in
        # TODO:
        # Get full list of ticids for this sector
        for lcpath in lcpaths:
            match = re.search(r'-0{6,}(\d+)-', lcpath)
            if match:
                tic_id = match.group(1)
                #print(tic_id)
            else:
                print("TIC ID not found in expected format.")
            ticids.append(tic_id)# process lcpaths to extract ticids
    return ticids

def bin_lightcurve(time, flux, bin_minutes=30):
    '''BINNING TO 30 MINUTES'''
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


def pipeline(detrender = "notch", sect_no = 90, wdwle = 15, make_plots = False):
    sector_number = sect_no
    sector_str = str(sect_no)
    sdethreshold_wotan = 10
    sdethreshold_notch_bic = 19
    wdwstr = str(wdwle)
    detrend = detrender # defining the detrending method, this is not relevant rn, will get relevant when we have notch as alt.
    lcpaths = glob(f"/ar1/TESS/SPOC/s00{sector_number}/*.fits") # wherever your fits lightcurves are saved

    failed_tics_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/notch_v2_Aug_2025/mainlc/sector{sector_number}/failed_tics.txt"
    rapid_rotators_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/notch_v2_Aug_2025/mainlc/sector{sector_number}/rapidrotators.txt"
 
    #if make_plots == True:
        # Input possibility of make_plots being true 

    valid_ticids = get_this_sectors_ticids(make_plots, sector_number, detrend, wdwstr)
    DEBUG = True

    for lcpath in lcpaths:

        hdu_list = fits.open(lcpath)
        hdr = hdu_list[0].header
        data = hdu_list[1].data
        hdu_list.close()
        tic_id = hdr.get('TICID', 'unknown')
        ra = hdr.get('RA') or hdr.get('RA_OBJ') or hdr.get('TARGRA') or 'Not found'
        dec = hdr.get('DEC') or hdr.get('DEC_OBJ') or hdr.get('TARGDEC') or 'Not found'

        if str(tic_id).lstrip('0') not in valid_ticids:
            DEBUG and print(f"{tic_id} is not a high sde tic... skipping") 
            continue # scanning through every tic

        outpath = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/notch_v2_Aug_2025/mainlc/sector{sector_number}/TIC_{tic_id}.pdf"
        if os.path.exists(outpath):
            DEBUG and print(f"Skipping TIC {tic_id} - already cached")
            continue

        pickle_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/notch_v2_Aug_2025/pickle/sector{sector_number}/TIC_{tic_id}.pkl"

        if not make_plots:
            if os.path.exists(pickle_path) or str(tic_id) in failed_tics_path: # trying to cache the failed ones as well
                DEBUG and print(f"Skipping TIC {tic_id} — already in failed_tics_path.")
                continue

        time, tessmag, tempeff, sap_flux, pdcsap_flux, qual, bkgd = get_data(data, hdr)
        time, sap_flux, pdcsap_flux, bkgd = clean_data(qual, time, sap_flux, pdcsap_flux, bkgd)

        q1,q3,iqr,mask = clip_masks(sap_flux)

        time = time[mask]
        pdcsap_flux = pdcsap_flux[mask]
        sap_flux = sap_flux[mask]
        bkgd = bkgd[mask]

        sap_time_binned, sap_flux_binned = bin_lightcurve(time, sap_flux/np.nanmedian(sap_flux))
        pdc_time_binned, pdc_flux_binned = bin_lightcurve(time, pdcsap_flux/np.nanmedian(pdcsap_flux))
        bkg_time_binned, bkg_flux_binned = bin_lightcurve(time, bkgd/np.nanmedian(bkgd))

        '''LOMBSCARGLE PERIODOGRAM '''

        frequency_SAP, power_SAP = LombScargle(sap_time_binned, sap_flux_binned).autopower()
        frequency_PDCSAP, power_PDCSAP = LombScargle(pdc_time_binned, pdc_flux_binned).autopower()
        mask2 = frequency_PDCSAP < 20
        frequency_PDCSAP = frequency_PDCSAP[mask2]
        power_PDCSAP = power_PDCSAP[mask2]
        frequency_SAP = frequency_SAP[mask2]
        power_SAP = power_SAP[mask2]
        best_frequency_SAP = frequency_SAP[np.argmax(power_SAP)]
        best_period_SAP = 1 / best_frequency_SAP 
        best_frequency_PDCSAP = frequency_PDCSAP[np.argmax(power_PDCSAP)]
        best_period_PDCSAP = 1 / best_frequency_PDCSAP

        DEBUG and print(f'Best period : {best_period_PDCSAP}')
        DEBUG and print(f'Best frequency: {best_frequency_PDCSAP}')

        if best_period_PDCSAP < 1:
                DEBUG and print("Rapid rotating, skipping for now....")
                exist_rotator = set()
                if os.path.exists(rapid_rotators_path):
                    with open(rapid_rotators_path, "r") as f:
                        exist_rotator = set(line.strip() for line in f)
                        if str(tic_id) not in exist_rotator:
                            with open(rapid_rotators_path, "a") as f:
                                f.write(f"{tic_id}\n")
                continue
        
        '''PHASE FOLDING TO ONE TIME PERIOD '''
        sap_phase, pdcsap_phase = phase_folder(sap_time_binned, best_period_SAP, pdc_time_binned, best_period_PDCSAP)


        ''' NOW WE DO THE WOTAN FLATTENING OF THE LIGHT CURVE '''
        if not make_plots:
            if best_period_PDCSAP > 2:
                dictionary = {"window_length" : 1}
            else:
                dictionary = {"window_length" : 0.5}

            clipped_flux = slide_clip(
            time, pdcsap_flux, window_length=dictionary['window_length'],
            low=100, high=2, method='mad', center='median') # extra clipping for NOTCH, from the wotan code

            sel = np.isfinite(time) & np.isfinite(clipped_flux)
            time = time[sel]
            pdcsap_flux = 1.* clipped_flux[sel]
            assert len(time) == len(pdcsap_flux)

            flatten_lc, trend_lc, notch = _run_notch(time, pdcsap_flux/np.nanmedian(pdcsap_flux), dictionary)

            ''' WE DO NOT NEED TO NORMALIZE THE DELTABIC ''' 
            delbic = notch.deltabic * -1 
            time_clean, delbic_clean = clean_arrays(time, delbic)
            time_clean_pdc, pdc_clean = clean_arrays(time, flatten_lc)

            DEBUG and print(f"TIC {tic_id}: DELBIC clean time length = {len(time_clean)}, flatten length = {len(delbic_clean)}")
            DEBUG and print(f"TIC {tic_id}: PDCSAP clean time length = {len(time_clean_pdc)}, flatten length = {len(pdc_clean)}")

            DEBUG and print(f"TIC {tic_id}: SAP/DELBIC clean time NaNs = {np.isnan(time_clean).sum()}, flatten NaNs = {np.isnan(delbic_clean).sum()}")
            DEBUG and print(f"TIC {tic_id}: PDCSAP clean time NaNs = {np.isnan(time_clean_pdc).sum()}, flatten NaNs = {np.isnan(pdc_clean).sum()}")

            objective = "likelihood"
            try:
                model1 = BoxLeastSquares(time_clean, delbic_clean) # setting dy = None
                #model2 = BoxLeastSquares(time_clean_pdc, pdc_clean)

                min_period = 0.5  # days, or a bit more than your cadence
                max_period = (time_clean.max() - time_clean.min()) / 2

                durations = np.linspace(0.01, 1, 75)
                ### if objective unspecified, bls assumes objective = 'likelihood'
                results1 = model1.autopower(durations, objective='likelihood')
                #results2 = model2.autopower(durations, objective='likelihood')

                best_idx = np.argmax(results1.power)
                best_period = results1.period[best_idx]
                best_depth = results1.depth[best_idx]
                best_transit_time = results1.transit_time[best_idx]
                best_snr = results1.depth_snr[best_idx]
                best_power = results1.power[best_idx]

            except:
                DEBUG and print(f"TIC {tic_id}: BLS model creation failed with error")
                with open(failed_tics_path, "a") as failfile:
                    failfile.write(f"{tic_id}\n")
                continue
            
            sde = sde_calc(results1.power, results1.period, window_width= 0.1)

            DEBUG and print(f"TIC {tic_id}: SDE = {sde}")

            row = {"tic_id" : tic_id, "time" : time_clean, "flux" : pdcsap_flux, "flatten_flux" : pdc_clean, "trend_lc" : trend_lc,
                 "deltabic" : delbic_clean, "Detection_Statistic" : sde, "Orbital_P" : best_period, "Peak_Power": best_power,
                 "Best_depth" : best_depth, "Best_SNR" : best_snr, "Best_Transit_Time" : best_transit_time,
                 "Powers" : results1.power, "Periods" : results1.period, "Depths" : results1.depth, "Depth_SNR" : results1.depth_snr,
                  "Transit_Time" : results1.transit_time, "Results": results1}

            with open(pickle_path, 'wb') as f:
                pickle.dump(row, f)


            
pipeline()

