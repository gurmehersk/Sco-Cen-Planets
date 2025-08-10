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
import pandas as pd
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
# FIX THE NOTCH PART FOR LABEL = TRUE
# Work on rerunning wotan in the right order, 10 first, then 15, then 20, then 30. Make sure this order is maintained esp for vetting
# We have the pickle files and the highsdetic already so thats good, half the work is done.
# Create a new sort of subdirectory/directory for v0, v1 (the one ill make rn)
# version 1 should have all the lightcurves for the windows and sectors and then a subdirectory called "to vet" which has *NO REPEATS*

# 4th AUGUST 
### SO! Notch does NOT work with rapid rotators, so add that condition even for NOTCH!!! --> dont reserve to wotan --> for future sectors
### 4th August 10.53pm ^^^^ Working on the rapid rotators issue, adding a rapidrotator.txt in every sector and removing the if block
### Fix this tic ids string issue. For some reason the tic ids arent being saved properly and are being repeated/overwritten. This is bad
### 4th august 10.53 pm ^^^ I think I fixed this above issue

# 5th AUGUST
#### HAVE REALIZED A MISTAKE WHICH IS VERY VERY VERY VERY CRUCIAL. With notch, please make sure the make_plots is in the correct setting
#### rerunning under false erases notch pkl file 


'''FUNCTIONS *ONLY* FOR NOTCH IMPLEMENTATION BEGINNING'''
'''def create_downlink_mask(
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
        #Masking the window BEFORE the gap

        last_point_before_gap_time = time[indx - 1]
        pre_window_start = last_point_before_gap_time - pre_gap_window # this pre gap window is susceptible 
        # to change acc to what we set it
        pre_window_end = last_point_before_gap_time
        
        points_to_mask_pre = (time >= pre_window_start) & (time <= pre_window_end)
        mask[points_to_mask_pre] = False
        print(f"Masking PRE-gap data between T={pre_window_start:.3f} and T={pre_window_end:.3f}")

        #Masking the window AFTER the gap
        # same thing but now we add, kinda like the window slider code in many ways
        first_point_after_gap_time = time[indx]
        post_window_start = first_point_after_gap_time
        post_window_end = first_point_after_gap_time + post_gap_window

        points_to_mask_post = (time >= post_window_start) & (time <= post_window_end)
        mask[points_to_mask_post] = False
        print(f"Masking POST-gap data between T={post_window_start:.3f} and T={post_window_end:.3f}")
        print("dt values >", gap_threshold, ":", dt[dt > gap_threshold])
        print("Corresponding indices:", np.where(dt > gap_threshold)[0])
    return mask'''

### 3rd August --> definitely do this tomorrow, just save the deltabic and time as a pickle file as well, somewhere in there, and use that to generate plots
## w.r.t. the notch methodology, since we are running it on unbinned data, we know it takes a long time for the code to run
## Now, since for notch we aren't really concerned with flux time series, to save time, if makeplots = True
## lets just apply tls on the deltabic and show the mosaic plotter of that!
## This skips the entire notch process again when makeplots is now true [before we ran it anyways when makepltos = false to generate pickle file]
## a workaround to this would be, to save the trends that notch generates also in the pickle files.
## we didnt think to do this earlier with wotan since the wotan detrending took mere seconds, and didnt take 8 minutes like notch does
## talk to Luke about this tomorrow!

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

'''FUNCTIONS *ONLY* FOR NOTCH IMPLEMENTATION END'''

def phase_bin(phase, flux, bins=100):
    phase = phase % 1  # wrap phase to [0, 1]
    bin_edges = np.linspace(0, 1, bins + 1) # creates bin number of bins between 0 and 1, since it is folded between 0 and 1
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1]) # calcualtes the center point of each bin 
    digitized = np.digitize(phase, bin_edges) - 1 # to assign the right bin for each object in the array list 

    binned_flux = [np.nanmedian(flux[digitized == i]) if np.any(digitized == i) else np.nan for i in range(bins)] # checks if bin is 
    #empty, returns nan if true

    return bin_centers, np.array(binned_flux)
# 29th July unvetted_tic currently commented out due to the re-running of pipeline 
'''def unvetted_tic(tic_id, detrend, sector_number, wdwstr):
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
        path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+detrend+f"/mainlc/sector{sector_number}/highsdetic10.txt"
        existing_tics = set()
        with open(path, "r") as f:
            existing_tics = set(line.strip() for line in f)
        if tic_id in existing_tics:
            return False
        else:
            return True
        # the path changes for notch since no windowed slider lengths -> COMPLETE
    return True'''


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
        # add if condition for wotan, else for notch -> done, onlu highsdeticpath changes
        if detrend.lower() == "wotan":
            highsdetic_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+detrend+"/window"+ wdwstr +f"/mainlc/sector{sector_number}/highsdetic10.txt" # changed the file to highsdetic10.txt now 
        elif detrend.lower() == "notch":
            highsdetic_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+detrend+ f"/mainlc/sector{sector_number}/highsdetic10.txt"
        
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

def pipeline(detrender, sect_no, wdwle, make_plots = True): # PLEASE CHECK make_plots, IF YOU RUN IT AGAIN WHEN FALSE, NOTCH DATAFRAME
    # GETS ERASED!!!!!!!! ##### BE EXTRA CERTAIN THAT THE FILE HAS SAVED WHEN YOU MAKE THAT CHANGE TO TRUE !!!!
    "IMPORTED EVERYTHING OUTSIDE NOW THAT I'M CHUNKING EVERYTHING"
    sector_number = sect_no
    sector_str = str(sect_no)
    sdethreshold_wotan = 10
    sdethreshold_notch_bic = 19
    wdwstr = str(wdwle)
    detrend = detrender # defining the detrending method, this is not relevant rn, will get relevant when we have notch as alt.
    lcpaths = glob(f"/ar1/TESS/SPOC/s00{sector_number}/*.fits") # wherever your fits lightcurves are saved 
    #lcpaths = [ lcpaths[1] ]
    # IMPORTANT!! wotan separate, notch separate, notch doesnt have a rapid_rotators path though since it will do it on that as well.

    if detrend.lower() == "wotan":
        failed_tics_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+ detrend +"/window"+ wdwstr +f"/mainlc/sector{sector_number}/failed_tics.txt"
        rapid_rotators_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+ detrend + "/window"+ wdwstr +f"/mainlc/sector{sector_number}/rapidrotators.txt"
    
    elif detrend.lower() == "notch": # create this subdirectory noon, 28th July instruction --> DONE 
        failed_tics_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+ detrend + f"/mainlc/sector{sector_number}/failed_tics.txt"
        rapid_rotators_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/notch/mainlc/sector{sector_number}/rapidrotators.txt"

    # creating path to the csv file which holds important information to avoid rerunning notch
    flux_pkls_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/notch/flux_pkls/"
    # This will store all the data before it is turned into a pkl file 
    flux_data = [] 

    # load pickle file only if make_plots is true, otherwise this won't exist!
    if make_plots:
        pkl_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/notch/flux_pkls/{sector_number}_all_flux.pkl"
        df = pd.read_pickle(pkl_path)
        # Check the path to confirm it's what you think it is
        print(f"Attempting to read from: {pkl_path}")

        # Check the size of the DataFrame
        print(f"DataFrame loaded with {len(df)} rows.")

        # Check the columns of the loaded DataFrame
        print(f"DataFrame columns immediately after load: {df.columns}")

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
        ra = hdr.get('RA') or hdr.get('RA_OBJ') or hdr.get('TARGRA') or 'Not found'
        dec = hdr.get('DEC') or hdr.get('DEC_OBJ') or hdr.get('TARGDEC') or 'Not found'

        if str(tic_id).lstrip('0') not in valid_ticids:
            DEBUG and print(f"{tic_id} is not a high sde tic... skipping") 
            continue # scanning through every tic
        
        #multipage_pdf_path = f"/home/gurmeher/gurmeher/detrending/sde10lightcurves/edited/sector{sector_number}/TIC_{tic_id}.pdf" # comment this 
        if detrend.lower() == "wotan":
            # outpath changed for mosaics, added mosaics and removed mainlc
            outpath = join(RESULTSDIR, detrend, 'mosaic', 'window'+wdwstr, 'sector'+sector_str, f"TIC_{tic_id}.pdf")
            if os.path.exists(outpath):
                DEBUG and print(f"Skipping TIC {tic_id} — already cached.")
                continue # caching. If tic has already been produced, don't run algorithm again
        elif detrend.lower() == "notch":
            outpath = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/notch/mainlc/sector{sector_number}/TIC_{tic_id}.pdf"
            if os.path.exists(outpath):
                DEBUG and print(f"Skipping TIC {tic_id} - already cached")
                continue

        # NEW PICKLE PATH FOR NOTCH
        if detrend.lower() == "wotan":
            pickle_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+ detrend + "/window"+ wdwstr + f"/pickle/sector{sector_number}/TIC_{tic_id}.pkl" # change this to a new path 
        elif detrend.lower() == "notch":
            pickle_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/notch/pickle/sector{sector_number}/TIC_{tic_id}.pkl"
            #### extra pickle path for individual flux arrays etc to save time and cache
            #### Mainly need this for sector91 and sector90 if it fails.
            flux_pickle_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/notch/pickle_fluxes/sector{sector_number}/TIC_{tic_id}.pkl"
        
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


        '''Neglecting rapid rotators now FOR WOTAN *AND* NOTCH ''' 
        if detrend.lower() == "wotan" or detrend.lower() == "notch":
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
        else:
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

        elif detrend.lower() == "notch" and not make_plots: # for reasoning behind not make_plots, read final documentation line in comment below
            '''
            Inputting Notch Implementation, which works a little differently
            It isn't blackboxed as a package on wotan. The implemenetation aspect has
            been done manually using Rizzuto+2017's method along with the process being
            forked as a package by Luke Bouma on git.

            Important to note that notch doesn't have varying window length, i.e., we aren't 
            testing different windows, so the file structure in results will be different and something
            we will have to change particularly for notch
            
            Also, in the NOTCH implementation, we will only be dealing with PDCSAP

            Additionally, the time series upon which TLS will be run, is not a flux time series, rather
            a deltabic time series. This is a little difficult to intuitively grasp initially but it helps
            detect transits which much greater efficiency. For this reason deltabic doesnt have a "detrender" pattern 
            per se. We check if within a particular window, the notch + transit model is preferred, and we do a bic comparison 
            in each window. This bic comparison [bayesian information criterion] is done with the following calculation

            BIC = chi square + numebr of parameters * natlog(numberofpoints)
            This chi square is calculated multiple times and in different iterative ways to remove any possible outliers. 

            Also, We should most likely ideally use unbinned data since notch is very sensitive, losing signal to binning 
            can actually be incredibly penalizing in this algorithm. This is from real experience testing with true positives
            planet discoveries from TESS data. 

            4th August edit: the "and not make_plots" is done intentionally to ensure that notch implementation only runs if we are 
            generating this data for the first time. If make_plots is true, we already know that the data generated by notch, i.e., 
            flattened_flux, trend_lc2, deltabic has been saved in a pkl file which can be found in the flux_pkls_path 
            '''
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

            '''downtimemask = create_downlink_mask(pdc_time_binned)
            pdc_time_binned = pdc_time_binned[downtimemask]
            pdc_flux_binned = pdc_flux_binned[downtimemask]'''

            flatten_lc2, trend_lc2, notch = _run_notch(time, pdcsap_flux/np.nanmedian(pdcsap_flux), dictionary)



            ''' NORMALIZING DELTA BIC to make it a TLS runnable TIME SERIES '''
            delbic = notch.deltabic * -1
            median_val = np.nanmedian(delbic)
            min_val = np.min(delbic)
            shift = delbic - min_val # this makes the minimum value 0
            scale_factor = median_val - min_val # the scale factor isnt range, but rather median - min
            delbic = shift/scale_factor 


            flatten_lc1 = delbic # for naming consistency for plotting
            # commenting this out rn cuz technically deltabic doesnt have a trend. 
            #trend_lc1 = trend_lc2  
            # since delbic doesnt have a trend, just make both of them delbic  (?)
            # continue
        

        ''' NOW WE DO THE TLS PHASE FOLDING AND PLOTTING ON A NEW GRAPH '''
        if detrend.lower() == "wotan":
            sap_time_clean, flatten_lc1_clean = clean_arrays(time, flatten_lc1)
            pdc_time_clean, flatten_lc2_clean = clean_arrays(time, flatten_lc2)

        elif detrend.lower() == "notch" and not make_plots:
            sap_time_clean, flatten_lc1_clean = clean_arrays(time, flatten_lc1)
            pdc_time_clean, flatten_lc2_clean = clean_arrays(time, flatten_lc2)
            # REMEMBER HERE --> SAP_TIME_CLEAN IS THE DELBIC 

        if detrend.lower() == "wotan" or (detrend.lower() == "notch" and not make_plots) : # same reason for the make_plots here, need to compound that logical 
            '''
            Same reason for the logical expression check here. If we don't add this if condition above, this block of code 
            will show an error for the notch implementation since pdc_time_clean, flatten_lc2_clean and all the above 
            variables declared would not exist if make_plots is false

            '''
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
            
            
            ### 7TH AUGUST, ADDED THIS FAILSAFE IN CASE DELTABIC CONDITION DOESNT WORK FOR SOME REASON
            ### THIS CAUSED A RIFT IN MY SECTOR91 CODE, SO I HAD TO RERUN IT.
            try:
                model1 = transitleastsquares(sap_time_clean, flatten_lc1_clean)  # This is for deltabic 
                '''*ATTENTION* !!!! 
                sap_time_clean will be *deltabic* if NOTCH is chosen.
                NOT CHANGING THIS FOR NAMING CONSISTENCY IN CODE '''
                model2 = transitleastsquares(pdc_time_clean, flatten_lc2_clean)

                min_period = 0.5  # days, or a bit more than your cadence
                max_period = (pdc_time_clean.max() - pdc_time_clean.min()) / 2  # maximum orbtial period is half baseline
                DEBUG and print(max_period)
            except:
                DEBUG and print(f"TIC {tic_id}: TLS model creation failed with error")
                with open(failed_tics_path, "a") as failfile:
                    failfile.write(f"{tic_id}\n")
                continue

        # TO ADD PLOTS ACCORDING TO THE DETRENDER CHOSEN!

            #import IPython; IPython.embed() # --> to mess around and investigate inside the code
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
                if detrend.lower() == "wotan":
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(results2, f) # Only doing PDCSAP for this
                elif detrend.lower() == "notch":
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(results1, f) # IF NOTCH, WE WANT TO USE DELTABIC!!    
                
                # Here, the entire High_Detections.py code goes in!!
                if detrend.lower() == "wotan":
                    highsdetic_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+detrend+"/window"+ wdwstr +f"/mainlc/sector{sector_number}/highsdetic10.txt" # changed the file to highsdetic10.txt now 
                elif detrend.lower() == "notch":
                    highsdetic_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+detrend+ f"/mainlc/sector{sector_number}/highsdetic10.txt"
                objects = []
                high_detection = []
                
                if detrend.lower() == "wotan":
                    path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+ detrend + "/window"+ wdwstr + f"/pickle/sector{sector_number}/" # change this to a new path 
                elif detrend.lower() == "notch":
                    path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/notch/pickle/sector{sector_number}/"

                # a bit of caching 
                existing_tics = set()
                if os.path.exists(highsdetic_path):
                    with open(highsdetic_path, "r") as f:
                        existing_tics = set(line.strip() for line in f)

                for file in os.listdir(path): # traversing through all files inside the path
                    if file.endswith(".pkl"): # if the file is a pickle file 
                        filename_no_ext = os.path.splitext(file)[0]  # This is now a string like "TIC_123456789"
                        pkl_tic_id = filename_no_ext.split("_")[1]       # 2 step process to extract the TIC ID before unpacking it and adding it to a txt file
                        filepath = os.path.join(path, file)
                        with open(filepath, "rb" ) as f:
                            obj = pickle.load(f)
                            sde = obj.get('SDE', None)
                        if detrend.lower() == "wotan":
                            if sde > sdethreshold_wotan and pkl_tic_id not in existing_tics:
                                with open(highsdetic_path, "a") as f:
                                    f.write(f"{tic_id}\n")
                        elif detrend.lower() == "notch":
                            if sde > sdethreshold_notch_bic and pkl_tic_id not in existing_tics:
                                with open(highsdetic_path, "a") as f:
                                    f.write(f"{tic_id}\n")
                # caching the important data points in a pkl file so I do not need to detrend with Notch when make_plots == True
                # flatten_lc1_clean is just the cleaned deltabic values as per convention!! Sorry for any confusion being caused by this! 
                row = {"tic_id" : tic_id, "time" : pdc_time_clean, "flux" : pdcsap_flux, "flatten_flux" : flatten_lc2_clean, "trend_lc" : trend_lc2,
                 "normalized_deltabic" : flatten_lc1_clean, "DelBIC_SDE" : results1.SDE, "Flux_SDE" : results2.SDE}
                
                # save this row in flux_data 
                flux_data.append(row)
                if detrend.lower() == "notch":
                    with open(flux_pickle_path, 'wb') as f:
                        pickle.dump(row, f)

            except ValueError as e:
                DEBUG and print(f"TIC {tic_id}: TLS failed with error: {e}")
                with open(failed_tics_path, "a") as failfile:
                    failfile.write(f"{tic_id}\n")
                continue


        elif make_plots:
            
            # 4th AUGUST
            '''
            NEED TO CHANGE EVERYTHING FOR NOTCH. THE WAY THE NOTCH IMPLEMENTATION WORKS HERE TO MAKE PLOTS IS NOT THE SAME AS WOTAN AS WE ARE NOT
            GOING TO DETREND AGAIN. INITIALLY IF YOU LOOK UP, THE ENTIRE TLS PROCESS ONLY RUNS IF MAKEPLOTS IS FALSE FOR NOTCH, WHICH IS SET TO
            TRUE NOW! 
            '''
            # currently condition commented out since we want to get all of the files together. The sorting out will happen in post-processing
            '''if not unvetted_tic(tic_id, detrend, sector_number, wdwstr): # we are only trying to plot files which are not found in 
            #other highsdetic files, i.e., haven't already been run and found/detected/vetted
                continue'''
            if detrend.lower() == "wotan":
                results1 = model1.power(period_min = min_period, period_max = max_period) # now inputting minimum and maximum period to try and fix valueError of empty TLS
                results2 = model2.power(period_min = min_period, period_max = max_period)

            elif detrend.lower() == "notch":
                ticstr = str(tic_id)
                speech = False
                # The failing line is below this.
                speech and print(f"Checking DataFrame state just before failure...")
                speech and print(f"DataFrame has {len(df)} rows.")
                speech and print(f"DataFrame columns: {df.columns}")
                if 'tic_id' not in df.columns:
                    print("WARNING: 'tic_id' column is missing!")
                if tic_id in df["tic_id"].values: # the .values is required for changing it from pandas to a numpy array so that we can use the "in" function
                    r = df[df["tic_id"] == tic_id].iloc[0] # without the iloc, we would get a dataframe of that tic_id, we want to extract the row
                    time = r["time"]
                    flux = r["flux"]
                    flatten_flux = r["flatten_flux"]
                    trend = r["trend_lc"]
                    dBIC = r["normalized_deltabic"]
                    assert len(time) == len(flux) == len(flatten_flux) == len(trend) == len(dBIC)
                
                    model1 = transitleastsquares(time, dBIC)
                    model2 = transitleastsquares(time, flatten_flux)
                    min_period = 0.5  # days, or a bit more than your cadence
                    max_period = (time.max() - time.min()) / 2  # maximum orbtial period is half baseline
                    results1 = model1.power(period_min = min_period, period_max = max_period)
                    results2 = model2.power(period_min = min_period, period_max = max_period)
                else:
                    print(f"{tic_id} is not in the dataframe, so we are SKIPPING TIC ....")
                    print(type(ticstr))  # is it int or str?
                    print(df["tic_id"].apply(type).unique())  # types in the column
                    continue


            # Predict transit times using TLS results

            if detrend.lower() == "wotan":
                tls_period = results2.period
                tls_t0 = results2.T0
            elif detrend.lower() == "notch":
                tls_period = results1.period
                tls_t0 = results1.T0
            # this is because deltabic time series is more accurate than the flux time series in notch --> 

            # Compute all expected transit times within observed time span
            epochs = np.arange(-1000, 1000)
            transit_times = tls_t0 + (tls_period * epochs)
            # Compute a y-position slightly below the light curve's minimum flux
            if detrend.lower() == "wotan":
                y_marker = np.nanmin(flatten_lc2) - 0.005  # or adjust the offset
            else:
                y_marker = np.nanmin(flatten_flux) - 0.005 

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
            mosaic = (
            """
            AAAAAAAAAAAAAAAAA
            AAAAAAAAAAAAAAAAA
            AAAAAAAAAAAAAAAAA
            BBBBBBBBBBBBBBBBB
            BBBBBBBBBBBBBBBBB
            BBBBBBBBBBBBBBBBB
            CCCCCCCCCCCCCCCCC
            CCCCCCCCCCCCCCCCC
            CCCCCCCCCCCCCCCCC
            DDDDDDDDDDDDDDDDD
            DDDDDDDDDDDDDDDDD
            DDDDDDDDDDDDDDDDD
            EEEEFFFFGGGHHHIII
            EEEEFFFFGGGHHHIII
            """)

            #CHANGING PLOTTING STYLE TO SUBPLOT MOSAICS!

            subplotter, plaxes = plt.subplot_mosaic(mosaic, figsize=(19, 14 ))
            plt.subplots_adjust(hspace=0.3)

            ''' PLOTTING BINNED DATA '''

            plaxes['A'].scatter(sap_time_binned, sap_flux_binned, c='k', zorder = 2, s=5, label = 'SAP')
            plaxes['A'].set_title(f"DETRENDING TIC {tic_id}")
            plaxes['A'].set_xticks([])
            plaxes['B'].scatter(pdc_time_binned, pdc_flux_binned, zorder =2,  c='k', s=5, label = 'PDCSAP')
            plaxes['B'].set_xticks([])

            ''' PLOTTING LOMBSCARGLE PERIODOGRAM '''
            plaxes['G'].plot(frequency_SAP, power_SAP)
            plaxes['G'].set_yticks([])
            plaxes['H'].plot(frequency_PDCSAP, power_PDCSAP)
            plaxes['H'].set_yticks([])
            plaxes['G'].set_xscale('log')  # For PDCSAP
            plaxes['H'].set_xscale('log')

            '''# keeping the sap cutoff for the pdc_sap cutoff since they should be about the same when considering flares, wouldn't change much i believe
            for i in range(len(axs)):
                if i < 2 or i > 3:
                # initially had put the q1, q3 and everything here, caused such problems
                    axs[i].set_xlim(time.min()-2, time.max())
                else:
                    axs[i].set_xlim(0,20)
                    # I set the axis limit at 20 because it was catching another rotation period at about 0.02 days, which was a little crazy for the 
                    # that 0.02 thing I saw was the binning cadence signal which is okay to mask
                    # lightcurve
                    continue'''

                #ax.set_ylim([0, upper_bound])
    
            ''' PLOTTING WOTAN/NOTCH FLATTENING CURVE '''
            if detrend.lower() == "wotan":
                plaxes['A'].plot(time, trend_lc1, linewidth = 2, zorder = 1, color = 'red')
                plaxes['B'].plot(time, trend_lc2, linewidth = 2, zorder = 1, color = 'red')

                time_flatten1_binned, flatten_lc1_binned = bin_lightcurve(time, flatten_lc1)
                time_flatten2_binned, flatten_lc2_binned = bin_lightcurve(time, flatten_lc2)
                # for visualization purposes, flattened lightcurves have also been binned
                plaxes['C'].scatter(time_flatten1_binned, flatten_lc1_binned, s=1.5, color='black')
                plaxes['C'].set_xticks([])
                plaxes['D'].scatter(time_flatten2_binned, flatten_lc2_binned, s = 1.5, color = 'black')

                ylimitsap = plaxes['C'].get_ylim()
                sapstd = np.nanstd(flatten_lc1)
                yupsap = 1 + 2.5*sapstd if np.isfinite(sapstd) else 1.02
                ylowsap = ylimitsap[0] 
                if np.isfinite(ylowsap) and np.isfinite(yupsap):
                    plaxes['C'].set_ylim(ylowsap, yupsap)
                print(f"STD = {sapstd}")

                ylimitpdc = plaxes['D'].get_ylim()
                pdcstd = np.nanstd(flatten_lc2)
                yuppdc = 1 + 2.5*pdcstd if np.isfinite(pdcstd) else 1.02
                ylowpdc = ylimitpdc[0]
                print(f" STD pdc = {pdcstd}")
                print(ylowpdc)
                print(f"upper {ylimitpdc[1]}")
                print(f"{tic_id} upper = {yuppdc}")
                if np.isfinite(ylowsap) and np.isfinite(yupsap):
                    plaxes['D'].set_ylim(ylowpdc, yuppdc)
                print(f"STD = {pdcstd}")
            
            elif detrend.lower() == "notch":
                plaxes['B'].plot(time, trend, linewidth = 1.5, zorder = 1, color = 'red')

                plaxes['C'].scatter(time, dBIC, s=1.5, color='black', label = 'DeltaBic Time Series')

                time_bin, flatten_bin = bin_lightcurve(time, flatten_flux)
                plaxes['D'].scatter(time_bin, flatten_bin, s = 1.5, color = 'black', label = 'Flattened PDCSAP')

                ylimitBIC = plaxes['C'].get_ylim()
                BICstd = np.nanstd(dBIC)
                yupsap = 1 + 2.5*BICstd if np.isfinite(BICstd) else 1.02
                ylowsap = ylimitBIC[0]
                if np.isfinite(ylowsap) and np.isfinite(yupsap):
                    plaxes['C'].set_ylim(ylowsap, yupsap)
                
                ylimitFLAT = plaxes['D'].get_ylim()
                pdcstd = np.nanstd(flatten_flux)
                yuppdc = 1 + 2.5*pdcstd if np.isfinite(pdcstd) else 1.02
                ylowpdc = ylimitFLAT[0]
                if np.isfinite(ylowpdc) and np.isfinite(yuppdc):
                    plaxes['D'].set_ylim(ylowpdc, yuppdc)

            '''for ax in axs:
                ax.legend()''' # redundant right now

            # Plot blue triangles at each expected transit time
            for t in visible_transits:
                if detrend.lower() == "wotan":
                    plaxes['D'].scatter(t, y_marker, marker='^', color='blue', s=20, zorder=3, label='Transit time' if t==visible_transits[0] else "")
                elif detrend.lower() == "notch":
                    plaxes['C'].scatter(t, y_marker, marker='^', color='blue', s=20, zorder=3, label='Transit time' if t==visible_transits[0] else "")
                plaxes['B'].scatter(t, y_marker, marker = '^', color = 'blue', s=20, zorder=3, label='Transit time' if t == visible_transits[0] else "")


            if not ticker:
                plaxes['B'].set_title(f"Prot is {harmonic:.2f}x of Porb ", color = "red")
            else:
                plaxes['B'].set_title(f"Prot is not a harmonic of Porb within 1%, Potential Planetary Signal")

            #fig.savefig(savpath, bbox_inches='tight', format = "pdf")
            #plt.close(fig)

            ''' PLOTTING TLS '''

            #figure2, axs2 = plt.subplots(nrows = 2, figsize = (10,12))
            #plt.subplots_adjust(hspace=0.3)

            # whats the TLS found orbiital period 
                
            period1 = results1.period
            sde1 = results1.SDE

            period2 = results2.period
            sde2 = results2.SDE

            if detrend.lower() == "wotan":
                periods = results2.periods
                power = results2.power
            elif detrend.lower() == "notch":
                periods = results1.periods
                power = results1.power 

                
            plaxes['E'].scatter(results1.folded_phase, results1.folded_y, marker = 'o', zorder=1, s = 0.25, color = 'black')
            plaxes['E'].plot(results1.model_folded_phase, results1.model_folded_model, zorder=3, linewidth = 1, color = 'red')
            plaxes['F'].scatter(results2.folded_phase, results2.folded_y, marker = 'o', zorder =1, s = 0.25, color = 'black')
            plaxes['F'].plot(results2.model_folded_phase, results2.model_folded_model, zorder =3, linewidth = 1, color = 'red')

            if np.isfinite(ylowsap) and np.isfinite(yupsap):
                plaxes['E'].set_ylim(ylowsap, yupsap)
            else:
                print(f"Ylim a nan for TIC {tic_id}")
            if np.isfinite(ylowpdc) and np.isfinite(yuppdc):
                plaxes['F'].set_ylim(ylowpdc, yuppdc)
            else:
                print(f"Ylim a nan for TIC {tic_id}")
            #savpath2 = f"/home/gurmeher/gurmeher/detrending/TLS_TIC_{tic_id}.pdf"
            #for ax in axs2:
            #   ax.legend()
            # PDCSAP phase-binned
            sap_bin_phase, sap_bin_flux = phase_bin(results1.folded_phase, results1.folded_y, bins = 100)
            plaxes['E'].scatter(sap_bin_phase, sap_bin_flux, s = 40, color = 'dodgerblue', edgecolors = 'black', linewidths = 0.5, zorder = 2)
            #pdc_bin_flux = phase_bin_magseries(results2.folded_phase, results2.folded_y)
            pdc_bin_phase, pdc_bin_flux = phase_bin(results2.folded_phase, results2.folded_y, bins=100)
            plaxes['F'].scatter(
            pdc_bin_phase, pdc_bin_flux,
            s=40, color='dodgerblue', edgecolors='black', linewidths=0.5, zorder=2)
            #figure2.savefig(savpath2, bbox_inches ='tight', format = 'pdf')
            #plt.close(figure2)
            mult = best_period_PDCSAP/period2
            try:
                plaxes['I'].axis('off')
                if isinstance(ra, float) and isinstance(dec, float):
                    plaxes['I'].text(1.0,0.1, f"T-mag = {tessmag:.1f} \n Temp = {tempeff:.1f}K \n RA: {ra:.1f}º \n DEC: {dec:.1f}º\n Prot = {best_period_PDCSAP:.3f}d \n Porb = {period2:.3f}d \n Prot/Porb = {mult:.3f} \n PDCSAP SDE = {sde2:.2f} \n SAP SDE ={sde1:.2f}", ha='right', va='bottom', transform=plaxes['I'].transAxes,
                fontsize=10)
                else:
                    plaxes['I'].text(1.0,0.1, f"T-mag = {tessmag:.1f} \n Temp = {tempeff:.1f}K \n Prot = {best_period_PDCSAP:.3f}d \n Porb = {period2:.3f}d \n Prot/Porb = {mult:.3f} \n PDCSAP SDE = {sde2:.2f} \n SAP/DELBIC SDE ={sde1:.2f}", ha='right', va='bottom', transform=plaxes['I'].transAxes,
                fontsize=10)
            except TypeError:
                print(f"Annotation has a problem in one of the values, check in detail manually for TIC {tic_id}")
            

            subplotter.tight_layout()
            subplotter.savefig(outpath, bbox_inches='tight', format='pdf')
            plt.close(subplotter)
            #with PdfPages(outpath) as pdf:

                # Save first figure as page 1
             #   pdf.savefig(subplotter, bbox_inches='tight')
              #  plt.close(subplotter)
    if not make_plots:
        df_all = pd.DataFrame(flux_data)
        os.makedirs(flux_pkls_path, exist_ok = True) # create the directory. If it exists, dont need to raise error, just move on
        fin_pkl_path = os.path.join(flux_pkls_path, f"{sector_number}_all_flux.pkl")
        if df_all.empty and os.path.exists(fin_pkl_path): # MAKES SURE A PREVIOUS DATA FRAME ISNT REWRITTEN UNLESS ABSOLUTELY NECESSARY
            pass
        else:
            df_all.to_pickle(fin_pkl_path)
            print(f"Saved to {fin_pkl_path}")

               



        

