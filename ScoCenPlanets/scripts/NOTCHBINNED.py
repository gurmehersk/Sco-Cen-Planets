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
from astropy.timeseries import BoxLeastSquares 

### 15th August --> run with likelihood, and not just snr 
# 28th July post meeting notes 
# normalize so median is 1, and transits are below 1, but above 0. normalize it using the range
# and also flip 
# mask negative bics, where polynomial fits are preferred, all we care about is the periodicity of the spikes.
# mask all deltabic less than 0 to be 1 --> in our normalized case, 

# log files directory, KEEP THOSE LOG FILES MATE !!!!

# BIC COMPARISON, NOTCH HAS 3 EXTRA PARAMETERS IN ITS FIT
# BIC = k ln(n)− 2 ln(L) where k = number of parameters, n is number of data points, L is maximum likelihood of model given data


# 24th July
# remove flares

# TLS not giving correct period ---> fixed by fixing window length
# thing is, here, I can't even try to run the notch filter on the unbinned data cuz that will take too much time. So the detrending
# which is essentially happening on the binned data, is the only thing we can apply the flasttening to, so there is loss of SNR!

# I don't get why we are trying to do this method now frankly

# also, it doesn't seem to be too robust, I don't get the hype 

# ADD LS periodogram to be consistent about window length!



# 18th Aug
### My current concern is to jsut be able to find a more robust way to look at this window_width factor. I don't think
### it should be a constant value, but I think it should work at the same time?
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


    # NOTICE/WARNING/ATTENTION, DELTABIC > 0 MEANS NOTCH IS FAVORED!!!
    # Noticed this while going through the sliding window function in more detail.
    min_deltabic = -1

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

tic_id = 166527623
# 166527623 , the hip star isnt working --> wrong period everytime --> it isnt detecting the planet transit, but artificial transits --> 
# /home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2023096110322-s0064-0000000166527623-0257-s/tess2023096110322-s0064-0000000166527623-0257-s_lc.fits
# reason behind hip not working could very well be due to the binning process which removes the transit 
# [makes it shallower, refer to the window20 issue] --> ISSUE FIXED

# 152479118 --> /home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025071122000-s0090-0000000152479118-0287-s/tess2025071122000-s0090-0000000152479118-0287-s_lc.fits
# THE ABOVE IS A NO DETECTION SIGNAL STAR, it is a reference star that has no planet

# created due to data downlinks --> big problem --> similar to what wotan did in many cases, maybe jumped over possible other transits
# in a similar fashion/way --> i remember trying to find a way to remove these earlier, just to remove any dips near data downtime regions,
# but i couldnt figure out a way to do this, maybe talk to luke about this. 

# 441420236 --> tried bls on this, AU Mic b worked with this pipeline --> /home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2020186164531-s0027-0000000441420236-0189-s/tess2020186164531-s0027-0000000441420236-0189-s_lc.fits
# 460205581, TOI 837b, Luke's planet worked. --> /home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025071122000-s0090-0000000460205581-0287-s/tess2025071122000-s0090-0000000460205581-0287-s_lc.fits
# 88297141 --> /home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2025127075000-s0092-0000000088297141-0289-s/tess2025127075000-s0092-0000000088297141-0289-s_lc.fits
# 146520535 , TOI 942 not working --> orbital period is wrong --> /home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2020324010417-s0032-0000000146520535-0200-s/tess2020324010417-s0032-0000000146520535-0200-s_lc.fits
objective = "likelihood"
#objective = "snr"
path = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2023096110322-s0064-0000000166527623-0257-s/tess2023096110322-s0064-0000000166527623-0257-s_lc.fits"
pdfpath = f"/home/gurmeher/gurmeher/Notch_and_LOCoR/results/TIC_{tic_id}_{objective}.pdf"

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


pdc_time_binned, pdc_flux_binned = time_clean, flux_clean
# TRYING UNBINNED, UNCOMMENT BELOW TO USE BINNED ^ IT MENTIONS BINNED BUT ITS ACTUALLY UNBINNED
#pdc_time_binned, pdc_flux_binned = bin_lightcurve(time_clean, flux_clean)



#pdc_time_binned, pdc_flux_binned = bin_lightcurve(time, pdcsap_flux/np.nanmedian(pdcsap_flux))
#pdc_time_binned, pdc_flux_binned = bin_lightcurve(time, pdcsap_flux)

prot = Lombscargle(pdc_time_binned, pdc_flux_binned)
cadence = np.nanmedian(np.diff(pdc_time_binned))
#dictionary = {"window_length" : 0.3*prot}
#dictionary = {"window_length" : np.maximum(prot/20, cadence * 10)}
if prot > 2:
    dictionary = {"window_length" : 1}

else :
    dictionary = {"window_length" : 0.5} # window size 1 day

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

#downtimemask = create_downlink_mask(pdc_time_binned)
#pdc_time_binned = pdc_time_binned[downtimemask]
#pdc_flux_binned = pdc_flux_binned[downtimemask]

fig, axs = plt.subplots(nrows = 6, figsize = (6,10), sharex = False)
axs[0].scatter(pdc_time_binned, pdc_flux_binned, s = 0.5, zorder = 2, color = 'black')
#flat_flux, trend_flux, notch = _run_notch(pdc_time_binned, pdc_flux_binned/np.nanmedian(pdc_flux_binned), dictionary)
flat_flux, trend_flux, notch = _run_notch(pdc_time_binned, pdc_flux_binned, dictionary)

axs[1].scatter(pdc_time_binned , flat_flux, s = 0.5)
axs[0].plot(pdc_time_binned, trend_flux, color = 'red', linewidth = 1.5, zorder = 1)

dbic = notch.deltabic * -1
delbic = notch.deltabic * -1
median_val = np.nanmedian(delbic)
min_val = np.min(delbic)
shift = delbic - min_val # this makes the minimum value 0
scale_factor = median_val - min_val # the scale factor isnt range, but rather median - min
delbic = shift/scale_factor 
transit_model = "bls"
if transit_model == "tls":
    axs[2].scatter(pdc_time_binned, delbic, color = 'pink', s = 0.5)
else:
    axs[2].scatter(pdc_time_binned, dbic, color = 'pink', s =0.5)
print(f"DELTABIC VALUES BEFORE ANY PROCESSING: {notch.deltabic}")

#print(f"pdc_time_binned: {pdc_time_binned}")
#print(f"delbic: {delbic}")
#print(f"len(pdc_time_binned): {len(pdc_time_binned)}")
#print(f"Any NaNs in time? {np.any(np.isnan(pdc_time_binned))}")



if len(pdc_time_binned) == 0:
    raise ValueError("pdc_time_binned is empty!")



    
### 14th AUGUST --> WITHOUT ANY RENORMALIZATION OF THE DELTABIC, WE ARE NOW TRYING TO FIND THE DETECTED DIP 
### USING BOXLEASTSQUARES! 
if transit_model == "bls":
    model1 = BoxLeastSquares(pdc_time_binned, dbic) # setting dy = None
    model2 = BoxLeastSquares(pdc_time_binned, flat_flux)

elif transit_model == "tls":
    model1 = transitleastsquares(pdc_time_binned, delbic)
    model2 = transitleastsquares(pdc_time_binned, flat_flux)

min_period = 0.5  # days, or a bit more than your cadence
max_period = (pdc_time_binned.max() - pdc_time_binned.min()) / 2

if transit_model == "bls":
    durations = np.linspace(0.01, 1, 75)
    ### if objective unspecified, bls assumes objective = 'likelihood'
    results1 = model1.autopower(durations, objective='likelihood')
    results2 = model2.autopower(durations, objective='likelihood')

    best_idx = np.argmax(results1.power)
    best_period = results1.period[best_idx]
    best_depth = results1.depth[best_idx]
    best_transit_time = results1.transit_time[best_idx]
    #best_snr = results1.depth_snr[best_idx]
    best_power = results1.power[best_idx]

    # testing window size
    left = best_period - 0.1
    right = best_period + 0.1

    harmonics = [0.5, 2]  # Example harmonic multipliers
    harmonic_lines = [(best_period * h - 0.1, best_period * h + 0.1) for h in harmonics]


    axs[3].plot(results1.period, results1.power)
    axs[3].axvline(left, color = 'red', linestyle ='--')
    axs[3].axvline(right, color = 'red', linestyle ='--')

    for i, (h_left, h_right) in enumerate(harmonic_lines):
        axs[3].axvline(h_left, color='green', linestyle='--', label=f"Harmonic {i+1} - 0.1")
        axs[3].axvline(h_right, color='purple', linestyle='--', label=f"Harmonic {i+1} + 0.1")

    axs[4].plot(results2.period, results2.power)

    print(f"Period = {best_period}")
    print(f"Depth = {best_depth}")
    print(f"Transit Time = {best_transit_time}")
    #print(f"SNR = {best_snr}")
    ### peak snr only needed if we are using the snr method 
    print(f"Power = {best_power}")
else:
    results1 = model1.power(period_min = min_period, period_max = max_period) # now inputting minimum and maximum period to try and fix valueError of empty TLS
    results2 = model2.power(period_min = min_period, period_max = max_period)

    period1 = results1.period
    period2 = results2.period

    sde1 = results1.SDE
    sde2 = results2.SDE

    tls_t0 = results2.T0

    # Compute all expected transit times within observed time span
    epochs = np.arange(-1000, 1000)
    transit_times = tls_t0 + (period2 * epochs)
    # Compute a y-position slightly below the light curve's minimum flux
    y_marker = np.nanmin(flat_flux) - 0.005  # or adjust the offset

    # Only keep transits that fall within your light curve time span
    in_transit = (transit_times > pdc_time_binned.min()) & (transit_times < pdc_time_binned.max())
    visible_transits = transit_times[in_transit]

    axs[3].scatter(results1.folded_phase, results1.folded_y, marker = 'o', s = 0.25, color = 'black', label = f'SAP phase-folded\nTLS Period = {period1:.4f} d\nSDE = {sde1:.2f}')
    axs[3].plot(results1.model_folded_phase, results1.model_folded_model, color = 'red', label = 'TLS MODEL for SAP Flux')
    axs[4].scatter(results2.folded_phase, results2.folded_y, marker = 'o', s = 0.25, color = 'black', label = f'PDCSAP phase-folded\nTLS Period = {period2:.4f} d\nSDE = {sde2:.2f}')
    axs[4].plot(results2.model_folded_phase, results2.model_folded_model, color = 'red', label = 'TLS MODEL for PDCSAP Flux')
    periods = results1.periods
    powers = results1.power

    axs[5].plot(periods, powers, color='black')
    axs[5].set_xlabel("Trial Period (days)")
    axs[5].set_ylabel("TLS Power (SDE)")
    axs[5].set_title("TLS Detection Spectrum")
    for t in visible_transits:
        axs[0].scatter(t, y_marker, marker='^', color='blue', s=20, zorder=3, label='Transit time' if t==visible_transits[0] else "")
        
    counts = count_points_per_window(pdc_time_binned, dictionary['window_length'])


    #axs[3].scatter(pdc_time_binned, counts, lw=0.8)
    #axs[3].set_xlabel("Time [days]")
    #axs[3].set_ylabel("# of points in window")


    print(f"Number of sliding windows used by notch: {len(notch.t)}")
    print(f"NUMBER OF DATA POINTS IN TOTAL: {len(pdc_time_binned)}")
    txtpath = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/scripts/timechecker.txt"

    with open(txtpath, "w") as f:
        for i in pdc_time_binned:
            f.write(str(i) + "\n")

    
    print(prot)

    print(delbic)

for ax in axs:
        ax.legend()
print(delbic)
print(np.max(notch.deltabic))
sde = sde_calc(results1.power, results1.period, window_width= 0.1)
print(f"SDE: {sde:.2f}")
with PdfPages(pdfpath) as pdf:
    pdf.savefig(fig, bbox_inches = 'tight')
    plt.close(fig)
