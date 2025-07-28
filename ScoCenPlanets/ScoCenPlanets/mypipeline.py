from os.path import join
from ScoCenPlanets.paths import RESULTSDIR
import re # regular expressions, regex101.com
import numpy as np
from glob import glob
import pickle 
import os
from astropy.io import fits
from lightkurve import search_lightcurve
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from wotan import flatten
from transitleastsquares import transitleastsquares
from matplotlib.backends.backend_pdf import PdfPages
# import logging

#9th July
# creating a rapid rotators text file right now to just ensure whether all stars are actually undergoing processing.


#11th July
# create a subdirectory for everything for the windowlength as well. --> add this 
# create a subdirectory for the method wotan vs notch as well.

#22nd July
# Need to try to be able to only get unique new tics from new slider method.
# create a new checker where if the tic id is not in any of the other different windows same sector highsdetic.txt file
# only then add this tic and then analyze it. 
# did not do this for 30, but gonna do it for 20, and 10 

# 22nd July
# TO DO: Once the pickle files are generated, run the mainlc runs, ie makeplots = True for windowlengths 10 and 20! 
'''***IMPORTANT*** REMEMBER! --> WHENEVER YOU ARE USING A NEW SLIDER, 
OR THE NOTCH METHOD, JUST CREATE THE SUBDIRECTORY BEFORE YOU RUN THE CODE!'''

# 28th July
# Making the NOTCH part of this program since we now have the notch implementation semi complete.
# FOR NOTCH, THE INPUT OF THE SLIDER DOESNT MATTER SINCE THE SLIDER LENGTH IS DETERMINED BY THE ROTATIONAL PERIOD!

'''FUNCTIONS *ONLY* FOR NOTCH IMPLEMENTATION BEGINNING'''
def create_downlink_mask(
    time: np.ndarray,
    gap_threshold: float = 1.0,
    pre_gap_window: float = 0.5,
    post_gap_window: float = 0.5): # returns an np.ndarray
    """
    Creates a boolean mask to find & ignore data points immediately
    preceding and proceeding a "significant" data gap, the significance 
    is mentioned in the arguments section.

    I did this mainly because as mentioned in the main NOTCHBINNED.py code, 
    TLS isnt detecting the planet transit, but artificial transits
    created due to data downlinks --> big problem --> similar to what wotan 
    did in many cases, maybe jumped over possible other transits
    in a similar fashion/way --> i remember trying to find a way to remove 
    these earlier, just to remove any dips near data downtime regions,
    but i couldnt figure out a way to do this, maybe talk to luke about this.
    I found a way to do it but it is a little iffy right now

    I also want to point out that I most likely will not be using this method in
    the final implementation as I would rather have extra graphs to manually vet 
    than my process skipping over potential transits due to the removal of points.
    Additionally, This process is *NOT* for *WOTAN* since I am not using wotan on 
    binned data. This process is mainly for *NOTCH* because it works on binned data
    which can be sensitive to artifact sampling. Therefore, you will only see this 
    function being used in the notch detrending method of the algorithm. 

    Formal Parameters:
        time (np.ndarray): The array of time values for the light curve.
        gap_threshold (float): The minimum time difference (in days)
            to be considered a data gap. TESS gaps can
            be > 1 day. Defaults to 0.5.
        pre_gap_window (float): The duration of the window (in days)
            BEFORE a gap to be masked. Messing around with this, i think 0.5 was ifne
        post_gap_window (float, optional): Same but post the data gap

    Returns:
        np.ndarray: A boolean array of the same length as 'time'.
                    'True' indicates a good data point to keep --> basically not before 
                    or after a data gap. 'False' means it should be ignored
                   
    """
    # assume all data is good initially, so set -> True
    mask = np.ones_like(time, dtype=bool)

    
    # np.diff() returns an array one element shorter, so we prepend 0
    # so the resulting array has the same # of elements as the `time` 
    # array and in the same order
    dt = np.diff(time, prepend=time[0])

    # Now trying to find the indices where a data gap begins
    # A large dt at index i means time[i] is the first point AFTER the gap
    # The point BEFORE the gap is at index i-1
    gap_index = np.where(dt > gap_threshold)[0] # accesing first element of the tuple, which just contains array for indices

    if gap_index.size == 0: # basically no elements in the array
        print("No significant data gaps found.")
        return mask

    print(f"Found {len(gap_index)} data gap(s). Masking pre-/post-gap windows")

    # For each gap found, we can mask the data in the window preceding and proceeding it
    for indx in gap_index:
        '''Masking the window BEFORE the gap'''

        last_point_before_gap_time = time[indx - 1]
        pre_window_start = last_point_before_gap_time - pre_gap_window # this pre gap window is susceptible 
        # to change acc to what we set it
        pre_window_end = last_point_before_gap_time
        
        points_to_mask_pre = (time >= pre_window_start) & (time <= pre_window_end)
        mask[points_to_mask_pre] = False
        print(f"Masking PRE-gap data between T={pre_window_start:.3f} and T={pre_window_end:.3f}")

        '''Masking the window AFTER the gap'''
        # same thing but now we add, kinda like the window slider code in many ways
        first_point_after_gap_time = time[indx]
        post_window_start = first_point_after_gap_time
        post_window_end = first_point_after_gap_time + post_gap_window

        points_to_mask_post = (time >= post_window_start) & (time <= post_window_end)
        mask[points_to_mask_post] = False
        print(f"Masking POST-gap data between T={post_window_start:.3f} and T={post_window_end:.3f}")
        print("dt values >", gap_threshold, ":", dt[dt > gap_threshold])
        print("Corresponding indices:", np.where(dt > gap_threshold)[0])
    return mask

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

'''FUNCTIONS *ONLY* FOR NOTCH IMPLEMENTATION END'''



def unvetted_tic(tic_id, detrend, sector_number, wdwstr):
    tic_id = str(tic_id)

    if detrend.lower() == "wotan":
        window_length = ["10","15","20","30"]
        other_windows = [w for w in window_length if w != wdwstr]
        for i in other_windows:
            path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+detrend+"/window"+ i +f"/mainlc/sector{sector_number}/highsdetic10.txt"
            existing_tics = set()
            with open(path, "r") as f:
                existing_tics = set(line.strip() for line in f)
            if  tic_id in existing_tics:
                return False
            else:
                continue
    elif detrend.lower() == "notch":
        # the path changes for notch since no windowed slider lengths
    return True


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
        # add if condition for wotan, else for notch
        highsdetic_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+detrend+"/window"+ wdwstr +f"/mainlc/sector{sector_number}/highsdetic10.txt" # changed the file to highsdetic10.txt now 
        with open(highsdetic_path, 'r') as f: # opening it in read only mode
            ticids = set(line.strip() for line in f if line.strip())
    else:
        # add if condition for wotan, else for notch
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
        # ...
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

def pipeline(detrender, sect_no, wdwle, make_plots = False):
    "IMPORTED EVERYTHING OUTSIDE NOW THAT I'M CHUNKING EVERYTHING"
    sector_number = sect_no
    sector_str = str(sect_no)
    sdethreshold = 10
    wdwstr = str(wdwle)
    detrend = detrender # defining the detrending method, this is not relevant rn, will get relevant when we have notch as alt.
    lcpaths = glob(f"/ar1/TESS/SPOC/s00{sector_number}/*.fits") # wherever your fits lightcurves are saved 
    #lcpaths = [ lcpaths[1] ]
    # IMPORTANT!! wotan separate, notch separate, notch doesnt have a rapid_rotators path though since it will do it on that as well.
    failed_tics_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+ detrend +"/window"+ wdwstr +f"/mainlc/sector{sector_number}/failed_tics.txt"
    rapid_rotators_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+ detrend + "/window"+ wdwstr +f"/mainlc/sector{sector_number}/rapidrotators.txt"
    # creating validticids
    valid_ticids = get_this_sectors_ticids(make_plots, sector_number, detrend, wdwstr) # now this list will either have all ticids (if makeplots is false)
    # or it will have only the high sde tic ids
    DEBUG = True

    for lcpath in lcpaths:

        hdu_list = fits.open(lcpath)
        hdr = hdu_list[0].header
        data = hdu_list[1].data
        hdu_list.close()
        tic_id = hdr.get('TICID', 'unknown')

        if str(tic_id).lstrip('0') not in valid_ticids:
            DEBUG and print(f"{tic_id} is not a high sde tic... skipping") 
            continue # scanning through every tic
        
        #multipage_pdf_path = f"/home/gurmeher/gurmeher/detrending/sde10lightcurves/edited/sector{sector_number}/TIC_{tic_id}.pdf" # comment this 
        outpath = join(RESULTSDIR, detrend, 'window'+wdwstr, 'mainlc', 'sector'+sector_str, f"TIC_{tic_id}.pdf")
        if os.path.exists(outpath):
            DEBUG and print(f"Skipping TIC {tic_id} — already cached.")
            continue # caching. If tic has already been produced, don't run algorithm again
        

        # NEW PICKLE PATH FOR NOTCH??? --> LOT OF WORK, CONSULT LUKE 
        pickle_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+ detrend + "/window"+ wdwstr + f"/pickle/sector{sector_number}/TIC_{tic_id}.pkl" # change this to a new path 
        
        if not make_plots:
            if os.path.exists(pickle_path) or str(tic_id) in failed_tics_path: # trying to cache the failed ones as well
                DEBUG and print(f"Skipping TIC {tic_id} — already cached.")
                continue

        time, tessmag, tempeff, sap_flux, pdcsap_flux, qual, bkgd = get_data(data, hdr)
        #savpath = f"/home/gurmeher/gurmeher/detrending/TIC_{tic_id}.pdf"
        time, sap_flux, pdcsap_flux, bkgd = clean_data(qual, time, sap_flux, pdcsap_flux, bkgd)
        #sel = (qual == 0)
        #time = time[sel]
        #sap_flux = sap_flux[sel]
        #pdcsap_flux = pdcsap_flux[sel]
        #bkgd = bkgd[sel]

        #q1 = np.nanpercentile(sap_flux, 25)
        #q3 = np.nanpercentile(sap_flux, 75)
        #iqr = q3 - q1

        # Define upper limit to clip flares
        #upper_bound = q3 + 1.5 * iqr
        #mask = sap_flux < upper_bound
       
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


        '''Neglecting rapid rotators for now ''' 
        if best_period_PDCSAP < 1:
            DEBUG and print("Rapid rotating, skipping for now....")
            exist_rotator = set()
            if os.path.exists(rapid_rotators_path):
                with open(rapid_rotators_path, "r") as f:
                    exist_rotator = set(line.strip() for line in f)
                    if tic_id not in exist_rotator:
                        with open(rapid_rotators_path, "a") as f:
                            f.write(f"{tic_id}\n")
            continue

        '''PHASE FOLDING TO ONE TIME PERIOD '''
        sap_phase, pdcsap_phase = phase_folder(sap_time_binned, best_period_SAP, pdc_time_binned, best_period_PDCSAP)


        ''' NOW WE DO THE WOTAN FLATTENING OF THE LIGHT CURVE '''
        # add the wotan question wotan vs notch, and add the percentage slider as something the user can input 
        # also, add wotan flattening method if you need [not necessary]
        if detrend.lower() == "wotan":
            wdwl = (wdwle/100.0) * best_period_PDCSAP
            # changing the wotan flattening to the entire data, not just binned to preserve SNR!
            flatten_lc1, trend_lc1 = flatten(time, sap_flux/np.nanmedian(sap_flux), window_length = wdwl, return_trend = True, method = 'biweight')
            flatten_lc2, trend_lc2 = flatten(time, pdcsap_flux/np.nanmedian(pdcsap_flux), window_length = wdwl, return_trend = True, method = 'biweight')

        elif detrend.lower() == "notch":
            '''
            Inputting Notch Implementation, which works a little differently
            It isn't blackboxed as a package on wotan. The implemenetation aspect has
            been done manually using Rizzuto+2017's method along with the process being
            forked as a package by Luke Bouma on git.

            Important to note that notch doesn't have varying window length, i.e., we aren't 
            testing different windows, so the file structure in results will be different and something
            we will have to change particularly for notch
            
            Also, in the NOTCH implementation, we will only be dealing with PDCSAP, no SAP!
            '''
            if best_period_PDCSAP > 2:
                dictionary = {"window_length" : 1}
            else:
                dictionary = {"window_length" : 0.5}

            clipped_flux = slide_clip(
            pdc_time_binned, pdc_flux_binned, window_length=dictionary['window_length'],
            low=100, high=2, method='mad', center='median') # extra clipping for NOTCH, from the wotan code

            sel = np.isfinite(pdc_time_binned) & np.isfinite(clipped_flux)
            pdc_time_binned = pdc_time_binned[sel]
            pdc_flux_binned = 1.* clipped_flux[sel]
            assert len(pdc_time_binned) == len(pdc_flux_binned)

            downtimemask = create_downlink_mask(pdc_time_binned)
            pdc_time_binned = pdc_time_binned[downtimemask]
            pdc_flux_binned = pdc_flux_binned[downtimemask]

            flatten_lc2, trend_lc2, notch = _run_notch(pdc_time_binned, pdc_flux_binned/np.nanmedian(pdc_flux_binned), dictionary)

            delbic = notch.deltabic * -1
            delbic = delbic/np.nanmedian(delbic)

            flatten_lc1 = delbic # for naming consistency for plotting

            # continue
        

        ''' NOW WE DO THE TLS PHASE FOLDING AND PLOTTING ON A NEW GRAPH '''
        sap_time_clean, flatten_lc1_clean = clean_arrays(time, flatten_lc1)
        pdc_time_clean, flatten_lc2_clean = clean_arrays(time, flatten_lc2)

        DEBUG and print(f"TIC {tic_id}: SAP/DELBIC clean time length = {len(sap_time_clean)}, flatten length = {len(flatten_lc1_clean)}")
        DEBUG and print(f"TIC {tic_id}: PDCSAP clean time length = {len(pdc_time_clean)}, flatten length = {len(flatten_lc2_clean)}")

        DEBUG and print(f"TIC {tic_id}: SAP/DELBIC clean time NaNs = {np.isnan(sap_time_clean).sum()}, flatten NaNs = {np.isnan(flatten_lc1_clean).sum()}")
        DEBUG and print(f"TIC {tic_id}: PDCSAP clean time NaNs = {np.isnan(pdc_time_clean).sum()}, flatten NaNs = {np.isnan(flatten_lc2_clean).sum()}")

        if len(sap_time_clean) == 0 or len(flatten_lc1_clean) == 0:
            DEBUG and print(f"TIC {tic_id}: Empty SAP/DELBIC arrays for TLS.")
        

        if len(pdc_time_clean) == 0 or len(flatten_lc2_clean) == 0:
            DEBUG and print(f"TIC {tic_id}: Empty PDCSAP arrays for TLS.")


        if len(sap_time_clean) < 50 or (sap_time_clean.max() - sap_time_clean.min()) < 5:
            DEBUG and print(f"TIC {tic_id}: Not enough data points or too short time span for TLS.")

            # not skipping them now though, i will let them go into the exception error and be saved in the failedtics.txt list
        
        
            
        
        model1 = transitleastsquares(sap_time_clean, flatten_lc1_clean) 
        '''*ATTENTION* !!!! 
        sap_time_clean will be ** if NOTCH is chosen.
        NOT CHANGING THIS FOR NAMING CONSISTENCY IN CODE '''
        model2 = transitleastsquares(pdc_time_clean, flatten_lc2_clean)

        min_period = 0.5  # days, or a bit more than your cadence
        max_period = (pdc_time_clean.max() - pdc_time_clean.min()) / 2  # maximum orbtial period is half baseline
        DEBUG and print(max_period)

        # TO ADD PLOTS ACCORDING TO THE DETRENDER CHOSEN!

            #import IPython; IPython.embed() # --> to mess around and investigate inside the code
        if detrend.lower() == "wotan":
            # getting an error on the results module here --> NOT ANYMORE I THINK (28TH JULY) 
            if not make_plots:
                try:
                    results1 = model1.power(period_min = min_period, period_max = max_period) # now inputting minimum and maximum period to try and fix valueError of empty TLS
                    results2 = model2.power(period_min = min_period, period_max = max_period)
                    ''' CACHING TO IGNORE ANY ITERATIONS THAT HAVE ALREADY OCCURED ''' 

                    '''pickle_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/pickle/sector{sector_number}/TIC_{tic_id}.pkl" # change this to a new path 
                    if os.path.exists(pickle_path) or str(tic_id) in failed_tics_path: # trying to cache the failed ones as well
                        DEBUG and print(f"Skipping TIC {tic_id} — already cached.")
                        continue'''
                    # the above line of code of the pickle path and the if statement was moved up to the top to avoid running tls on these
                    # as that wasted computing space.
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(results2, f) # Only doing PDCSAP for this
                    
                    # Here, the entire High_Detections.py code goes in!!
                    highsdetic_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+detrend+ "/window"+ wdwstr +f"/mainlc/sector{sector_number}/highsdetic10.txt"
                    objects = []
                    high_detection = []
                    path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+detrend+ "/window"+ wdwstr +f"/pickle/sector{sector_number}/"

                    # a bit of caching 
                    existing_tics = set()
                    if os.path.exists(highsdetic_path):
                        with open(highsdetic_path, "r") as f:
                            existing_tics = set(line.strip() for line in f)

                    for file in os.listdir(path): # traversing through all files inside the path
                        if file.endswith(".pkl"): # if the file is a pickle file 
                            filename_no_ext = os.path.splitext(file)[0]  # This is now a string like "TIC_123456789"
                            tic_id = filename_no_ext.split("_")[1]       # 2 step process to extract the TIC ID before unpacking it and adding it to a txt file
                            filepath = os.path.join(path, file)
                            with open(filepath, "rb" ) as f:
                                obj = pickle.load(f)
                                sde = obj.get('SDE', None)
                            if sde > sdethreshold and tic_id not in existing_tics:
                                with open(highsdetic_path, "a") as f:
                                    f.write(f"{tic_id}\n")

                except ValueError as e:
                    DEBUG and print(f"TIC {tic_id}: TLS failed with error: {e}")
                    with open(failed_tics_path, "a") as failfile:
                        failfile.write(f"{tic_id}\n")
                    continue


            if make_plots:
                
                if not unvetted_tic(tic_id, detrend, sector_number, wdwstr): # we are only trying to plot files which are not found in 
                #other highsdetic files, i.e., haven't already been run and found/detected/vetted
                    continue

                results1 = model1.power(period_min = min_period, period_max = max_period) # now inputting minimum and maximum period to try and fix valueError of empty TLS
                results2 = model2.power(period_min = min_period, period_max = max_period)

                # Predict transit times using TLS results
                tls_period = results2.period
                tls_t0 = results2.T0

                # Compute all expected transit times within observed time span
                epochs = np.arange(-1000, 1000)
                transit_times = tls_t0 + (tls_period * epochs)
                # Compute a y-position slightly below the light curve's minimum flux
                y_marker = np.nanmin(flatten_lc2) - 0.005  # or adjust the offset

                # Only keep transits that fall within your light curve time span
                in_transit = (transit_times > pdc_time_binned.min()) & (transit_times < pdc_time_binned.max())
                visible_transits = transit_times[in_transit]
                ticker = True
                harmonic = 0
                possible_harmonics = [0.25, 0.5, 1, 2, 3, 4]
                for h in possible_harmonics:
                    harmonic_checker = ((np.abs(best_period_PDCSAP-(h*tls_period)))/best_period_PDCSAP)*100
                    if harmonic_checker <= 1:
                        ticker = False
                        harmonic = h
                        break

                # no need for a try and except here because only working tls algorithms will fall down this path since they are in high
                # sde tic ids

                ''' PLOTTING RAW DATA FIRST '''

                fig3, axs3 = plt.subplots(nrows =3 , figsize = (6,10), sharex = True)
                plt.subplots_adjust(hspace = 0.3)
                axs3[0].scatter(time, sap_flux, c='k', s= 0.8, label = 'SAP')
                axs3[1].scatter(time, pdcsap_flux, c = 'k', s= 0.8, label = 'PDCSAP')
                axs3[2].scatter(time, bkgd, c = 'k', s = 0.8, label = 'BKGD')
                axs3[0].set_title(f" RAW DATA FOR TIC {tic_id}")


                ''' PLOTTING BINNED DATA '''

                fig, axs = plt.subplots(nrows=6, figsize=(12,18))
                plt.subplots_adjust(hspace=0.3)
                axs[0].scatter(sap_time_binned, sap_flux_binned, c='k', zorder = 2, s=0.8, label = 'SAP')
                axs[0].set_ylabel("SAP", fontsize = 8 )
                axs[0].set_title(f"TIC {tic_id} — TESS mag = {tessmag} at Temp = {tempeff} Binned to 30 minutes", fontsize=8)

                axs[1].scatter(pdc_time_binned, pdc_flux_binned, c='k', zorder = 2, s=0.8, label = 'PDCSAP')
                axs[1].set_ylabel("PDCSAP", fontsize = 8)


                ''' PLOTTING LOMBSCARGLE PERIODOGRAM '''
                
                axs[2].plot(frequency_SAP, power_SAP, label = 'SAP LS')
                axs[2].set_ylabel("Power", fontsize = 8)
                axs[2].set_xlabel('Frequency', fontsize = 8)

                axs[3].plot(frequency_PDCSAP, power_PDCSAP, label = 'PDC LS')
                axs[3].set_ylabel("Power", fontsize = 8)
                axs[3].set_xlabel('Frequency', fontsize = 8)

                axs[2].set_xscale('log')  # For SAP
                axs[3].set_xscale('log')  # For PDCSAP


                # keeping the sap cutoff for the pdc_sap cutoff since they should be about the same when considering flares, wouldn't change much i believe
                for i in range(len(axs)):
                    if i < 2 or i > 3:
                    # initially had put the q1, q3 and everything here, caused such problems
                        axs[i].set_xlim(time.min()-2, time.max())
                    else:
                        axs[i].set_xlim(0,20)
                        # I set the axis limit at 20 because it was catching another rotation period at about 0.02 days, which was a little crazy for the 
                        # that 0.02 thing I saw was the binning cadence signal which is okay to mask
                        # lightcurve
                        continue

                    #ax.set_ylim([0, upper_bound])
            
                '''PLOTTING PHASE FOLDED TO ONE PERIOD '''

                fig_phase, axs_phase = plt.subplots(2, figsize=(10, 8), sharex=True)
                plt.subplots_adjust(hspace=0.3)
                axs_phase[0].scatter(sap_phase, sap_flux_binned, s=0.5, c='black', label='SAP Phase Folded')
                axs_phase[0].set_ylabel("Flux")
                axs_phase[0].set_title(f"TIC {tic_id} — SAP Phase Folded at {best_period_SAP:.4f} d") # four decimal places rounded 
                axs_phase[0].legend()

                axs_phase[1].scatter(pdcsap_phase, pdc_flux_binned, s=0.5, c='black', label='PDCSAP Phase Folded')
                axs_phase[1].set_xlabel("Phase")
                axs_phase[1].set_ylabel("Flux")
                axs_phase[1].set_title(f"TIC {tic_id} — PDCSAP Phase Folded at {best_period_PDCSAP:.4f} d") # 4 decimal places round
                axs_phase[1].legend()

                plt.close(fig_phase)


                ''' PLOTTING WOTAN FLATTENING CURVE '''

                axs[0].plot(time, trend_lc1, linewidth = 1.5, zorder = 1, color = 'red')
                axs[1].plot(time, trend_lc2, linewidth = 1.5, zorder = 1, color = 'red')

                axs[4].scatter(time, flatten_lc1, s=1, color='black', label = 'Flattened SAP')
                axs[5].scatter(time, flatten_lc2, s = 1, color = 'black', label = 'Flattened PDCSAP')

                for ax in axs:
                    ax.legend()

                # Plot blue triangles at each expected transit time
                for t in visible_transits:
                    axs[5].scatter(t, y_marker, marker='^', color='blue', s=20, zorder=3, label='Transit time' if t==visible_transits[0] else "")
                    axs[1].scatter(t, y_marker, marker = '^', color = 'blue', s=20, zorder=3, label='Transit time' if t == visible_transits[0] else "")


                if not ticker:
                    axs[1].set_title(f"Prot is {harmonic:.2f}x of Porb ", color = "red")
                else:
                    axs[1].set_title(f"Prot is not a harmonic of Porb, Potential Planetary Signal")
    
                #fig.savefig(savpath, bbox_inches='tight', format = "pdf")
                plt.close(fig)

                ''' PLOTTING TLS '''

                figure2, axs2 = plt.subplots(nrows = 2, figsize = (10,12))
                plt.subplots_adjust(hspace=0.3)

                # whats the TLS found orbiital period 
                    
                period1 = results1.period
                sde1 = results1.SDE

                period2 = results2.period
                sde2 = results2.SDE

                periods = results2.periods
                power = results2.power

                f = plt.figure(figsize=(10, 5))
                plt.plot(periods, power, color='black')
                plt.xlabel("Trial Period (days)")
                plt.ylabel("TLS Power (SDE)")
                plt.title("TLS Detection Spectrum")
                plt.grid(True)
                plt.close()
                    
                axs2[0].scatter(results1.folded_phase, results1.folded_y, marker = 'o', s = 0.25, color = 'black', label = f'DELTABIC phase-folded\nTLS Period = {period1:.4f} d\nSDE = {sde1:.2f}')
                axs2[0].plot(results1.model_folded_phase, results1.model_folded_model, color = 'red', label = 'TLS MODEL for DELTABIC')
                axs2[0].set_title(f" TLS result algorithm on TIC {tic_id}")
                axs2[1].scatter(results2.folded_phase, results2.folded_y, marker = 'o', s = 0.25, color = 'black', label = f'Flattened Flux phase-folded\nTLS Period = {period2:.4f} d\nSDE = {sde2:.2f}')
                axs2[1].plot(results2.model_folded_phase, results2.model_folded_model, color = 'red', label = 'TLS MODEL for Flattened Flux')

                #savpath2 = f"/home/gurmeher/gurmeher/detrending/TLS_TIC_{tic_id}.pdf"
                for ax in axs2:
                    ax.legend()

                #figure2.savefig(savpath2, bbox_inches ='tight', format = 'pdf')
                plt.close(figure2)




                with PdfPages(outpath) as pdf:

                    # Save first figure as page 1
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                
                    # Saving figure as page 2
                    pdf.savefig(fig_phase, bbox_inches = 'tight')
                    plt.close(fig_phase)

                    # Save figure as page 3
                    pdf.savefig(fig3, bbox_inches = 'tight')
                    plt.close(fig3)

                    # Save figure as page 4
                    pdf.savefig(figure2, bbox_inches='tight')
                    plt.close(figure2)

                    pdf.savefig(f, bbox_inches='tight')
                    plt.close(f)
        
                



            

