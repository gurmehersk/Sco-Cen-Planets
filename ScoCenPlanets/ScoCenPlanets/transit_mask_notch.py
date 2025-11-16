

from astropy.io import fits
#import mypipeline as mp
#import notch_binned_module as nb 
import numpy as np
import logging 
import matplotlib.pyplot as plt
from wotan import flatten
from transitleastsquares import transitleastsquares
from transitleastsquares import transit_mask 
from astropy.timeseries import LombScargle
import sys 
from astropy.timeseries import BoxLeastSquares 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

def Lombscargle(time, flux):
    frequency_PDCSAP, power_PDCSAP = LombScargle(time, flux).autopower()
    mask2 = frequency_PDCSAP < 20
    frequency_PDCSAP = frequency_PDCSAP[mask2]
    power_PDCSAP = power_PDCSAP[mask2]
    best_frequency_PDCSAP = frequency_PDCSAP[np.argmax(power_PDCSAP)]
    best_period_PDCSAP = 1 / best_frequency_PDCSAP
    return best_period_PDCSAP

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

'''FUNCTIONS *ONLY* FOR NOTCH IMPLEMENTATION END'''
def run_transit_mask(time, flux, period, duration, T0):
    '''
    Function to run transit_mask on given time and flux data.

    Parameters
    ----------
    time : array-like
        Time data of the lightcurve
    
    flux : array-like
        Flux data of the lightcurve

    Returns
    -------
    Time and flux with transits masked :D

    '''
    intransit = transit_mask(time, period, 1.5*duration, T0)
    time_masked = time[~intransit]
    flux_masked = flux[~intransit]

    return time_masked, flux_masked

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

if __name__ == "__main__":

    tic_id = 88297141

    # Load the lightcurve
    lcpath = "/ar1/TESS/SPOC/s0092/tess2025127075000-s0092-0000000088297141-0289-s_lc.fits"
    data, hdr = fits.getdata(lcpath, header=True)
    objective = "likelihood"
    
    time, tessmag, tempeff, sap_flux, pdcsap_flux, qual, bkgd = get_data(data, hdr)

    time, sap_flux, pdcsap_flux, bkgd = clean_data(qual, time, sap_flux, pdcsap_flux, bkgd)

    q1,q3,iqr,masks = clip_masks(pdcsap_flux)
    time = time[masks]
    pdcsap_flux = pdcsap_flux[masks]
    normalized_flux = pdcsap_flux / np.nanmedian(pdcsap_flux)    

    mask2 = np.isfinite(time) & np.isfinite(normalized_flux)
    normalized_flux = normalized_flux[mask2]
    time = time[mask2]

    if len(time) == len(normalized_flux):
        logger.info("Data loaded successfully with {} points.".format(len(time)))
    else:
        logger.error("Data length mismatch after masking, exiting....")
        sys.exit(1)
    
    prot = Lombscargle(time, normalized_flux)

    real_binned_time, real_flux_binned = bin_lightcurve(time, normalized_flux)

    period = 4.64423 # found from mcmc fit
    t0 = 3803.24126 # found from mcmc fit
    duration = 0.14583 # 3.5 hours, in days
    transit_model = "bls"     

    # We will mask first, then detrend

    time_masked, flux_masked = run_transit_mask(time, normalized_flux, period, duration, t0)


    if prot > 2:
        dictionary = {"window_length" : 1}
    else:
        dictionary = {"window_length" : 0.5}
    
    clipped_flux = slide_clip(
            time_masked, flux_masked, window_length=dictionary['window_length'],
            low=100, high=2, method='mad', center='median') # extra clipping for NOTCH, from the wotan code

    sel = np.isfinite(time_masked) & np.isfinite(clipped_flux)
    time_masked = time_masked[sel]
    flux_masked = 1.* clipped_flux[sel]
    assert len(time_masked) == len(flux_masked)

    flatten_lc2, trend_lc2, notch = _run_notch(time_masked, flux_masked, dictionary)
    delbic = notch.deltabic * -1
    

    time_clean, flatten_lc2_clean = clean_arrays(time_masked, flatten_lc2)

    time_flat, flat_flux_binned = bin_lightcurve(time_clean, flatten_lc2_clean, bin_minutes=30)

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
            EEEEEEEEEEEEEEEEE
            EEEEEEEEEEEEEEEEE
            EEEEEEEEEEEEEEEEE
            FFFFGGGHHHIIIJJJJ
            FFFFGGGHHHIIIJJJJ
            """)
    subplotter, plaxes = plt.subplot_mosaic(mosaic, figsize=(19, 14 ))

    plaxes['B'].scatter(time_flat , flat_flux_binned, s = 0.5)
    plaxes['A'].plot(time_masked, trend_lc2, color = 'red', linewidth = 1.5, zorder = 1)

    plaxes['A'].scatter(real_binned_time, real_flux_binned, zorder =2,  s = 5, color = 'black')

    plaxes['C'].scatter(time_masked, delbic, color = 'pink', s =0.5)

    model1 = BoxLeastSquares(time_masked, delbic) # setting dy = None
    model2 = BoxLeastSquares(time_masked, flatten_lc2)

    min_period = 0.5  # days, or a bit more than your cadence
    max_period = (time_masked.max() - time_masked.min()) / 2

    durations = np.linspace(0.01, 0.49, 75)

    results1 = model1.autopower(durations, minimum_period = 1, maximum_period = (np.nanmax(time_masked) - np.nanmin(time_masked))/2.0, objective = 'likelihood')
    results2 = model2.autopower(durations, minimum_period = 1, maximum_period = (np.nanmax(time_masked) - np.nanmin(time_masked ))/2.0, objective = 'likelihood')

        #best_idx = np.argmin(np.abs(results1.period - 4.64))
    best_idx = np.argmax(results1.power) # 28th September --> going back to our original implementation for a second
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


    plaxes['D'].plot(results1.period, results1.power)
    plaxes['D'].axvline(left, color = 'red', linestyle ='--')
    plaxes['D'].axvline(right, color = 'red', linestyle ='--')

    epochs = np.arange(-1000, 1000)
    recovered_transit_times = best_transit_time + (best_period * epochs)
    transit_times = t0 + (period * epochs) # Blue marker will always show where the detection period occured


    in_transit2 = (recovered_transit_times > time.min()) & (recovered_transit_times < time.max())
    visible_transits2 = recovered_transit_times[in_transit2]
    y_marker2 = np.nanmin(flatten_lc2) - 0.005

    in_transit = (transit_times > time.min()) & (transit_times < time.max())
    visible_transits = transit_times[in_transit]
    y_marker = np.nanmin(flatten_lc2) - 0.005

    for t in visible_transits:
        plaxes['A'].scatter(t, y_marker, marker='^', color='blue', s=20, zorder=3, label='Transit time' if t==visible_transits[0] else "")
        plaxes['B'].scatter(t, y_marker, marker = '^', color = 'red', s=20, zorder=3, label='Transit time' if t == visible_transits[0] else "")
        plaxes['C'].scatter(t,y_marker, marker = '^', color = 'blue', s=20, zorder=3, label='Transit time' if t == visible_transits[0] else "")
    for t in visible_transits2:
        plaxes['B'].scatter(t, y_marker, marker = '^', color = 'blue', s=20, zorder=3, label='Transit time' if t == visible_transits2[0] else "")
    for i, (h_left, h_right) in enumerate(harmonic_lines):
        plaxes['D'].axvline(h_left, color='green', linestyle='--', label=f"Harmonic {i+1} - 0.1")
        plaxes['D'].axvline(h_right, color='purple', linestyle='--', label=f"Harmonic {i+1} + 0.1")

    plaxes['E'].plot(results1.period, results1.power) # deltabics' power spectrum 

    ylimitsap = plaxes['A'].get_ylim()
    pdcsapstd = np.nanstd(normalized_flux)
    yupsap = 1 + 2.5*pdcsapstd if np.isfinite(pdcsapstd) else 1.02
    ylowsap = 0.93
    if np.isfinite(ylowsap) and np.isfinite(yupsap):
        plaxes['A'].set_ylim(ylowsap, yupsap)

    print(f"Period = {best_period}")
    print(f"Depth = {best_depth}")
    print(f"Transit Time = {best_transit_time}")
    #print(f"SNR = {best_snr}")
    ### peak snr only needed if we are using the snr method 
    print(f"Power = {best_power}")

    print(delbic)
    print(np.max(notch.deltabic))
    #sde = sde_calc(results1.power, results1.period, results1, window_width= 0.1)
    #print(f"SDE: {sde:.2f}")
    ''''
    with PdfPages(pdfpath) as pdf:
        pdf.savefig(fig, bbox_inches = 'tight')
        plt.close(fig)'''

    subplotter.tight_layout()
    pdfpath = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/ScoCenPlanets/transit_mask_notch.pdf"
    subplotter.savefig(pdfpath, bbox_inches='tight', format='pdf')
    plt.close(subplotter)



