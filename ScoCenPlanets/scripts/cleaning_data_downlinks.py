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

import numpy as np
import matplotlib.pyplot as plt
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
# to clarify for my own understanding since I was getting an error, np.ndarray is the object type whereas np.array is a function
# you call to create a list so when you are returning something its okay for it to be np.array but if youre accepting a sort of 
# variable, use np.ndarray
def create_downlink_mask(
    time: np.ndarray,
    gap_threshold: float = 0.5,
    pre_gap_window: float = 0.5,
    post_gap_window: float = 0.5): # returns an np.ndarray
    """
    Creates a boolean mask to find &ignore data points immediately
    preceding and proceeding a "significant" data gap, the significance 
    is mentioned in the arguments section.

    I did this mainly because as mentioned in the main NOTCHBINNED code, 
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
    gap_indices = np.where(dt > gap_threshold)[0]

    if gap_indices.size == 0: # basically no elements in the array
        print("No significant data gaps found.")
        return mask

    print(f"Found {len(gap_indices)} data gap(s). Masking pre-/post-gap windows")

    # For each gap found, we can mask the data in the window preceding and proceeding it
    for idx in gap_indices:
        '''Masking the window BEFORE the gap'''

        last_point_before_gap_time = time[idx - 1]
        pre_window_start = last_point_before_gap_time - pre_gap_window # this pre gap window is susceptible 
        # to change acc to what we set it
        pre_window_end = last_point_before_gap_time
        
        points_to_mask_pre = (time >= pre_window_start) & (time <= pre_window_end)
        mask[points_to_mask_pre] = False
        print(f"Masking PRE-gap data between T={pre_window_start:.3f} and T={pre_window_end:.3f}")

        '''Masking the window AFTER the gap'''
        # same thing but now we add, kinda like the window slider code in many ways
        first_point_after_gap_time = time[idx]
        post_window_start = first_point_after_gap_time
        post_window_end = first_point_after_gap_time + post_gap_window

        points_to_mask_post = (time >= post_window_start) & (time <= post_window_end)
        mask[points_to_mask_post] = False
        print(f"Masking POST-gap data between T={post_window_start:.3f} and T={post_window_end:.3f}")

    return mask
    

# regularly scheduled code to test whether it has happened or not

tic_id = 166527623
path = "/home/gurmeher/.lightkurve/cache/mastDownload/TESS/tess2023096110322-s0064-0000000166527623-0257-s/tess2023096110322-s0064-0000000166527623-0257-s_lc.fits"
pdfpath = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/scripts/TIC_{tic_id}.pdf"

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

downlink_mask = create_downlink_mask(pdc_time_binned, gap_threshold = 1.0, pre_gap_window = 0.5, post_gap_window = 0.5)

# 3. Visualize the results
plt.style.use('seaborn-v0_8-white')
fig, ax = plt.subplots(figsize=(15, 7))

# Plotting all the original data points in a light color
ax.plot(pdc_time_binned, pdc_flux_binned, 'o', color='lightgray', markersize=3, label='All Data Points')

# Overplotginb the good data points (where mask = True) in a darker color
ax.plot(pdc_time_binned[downlink_mask], pdc_flux_binned[downlink_mask], 'o', color='dodgerblue', markersize=4, label='Good Data (Kept)')

# Highlighting the masked points in red by inverting the mask through the bitwise NOT operator : ~ 
ax.plot(pdc_time_binned[~downlink_mask], pdc_flux_binned[~downlink_mask], 'o', color='crimson', markersize=4, label='Artifact Data (Masked)')

ax.set_title("TESS Pre/Post-Downlink Artifact Masking", fontsize=16)
ax.set_xlabel("Time (BJD - XXXX)", fontsize=12)
ax.set_ylabel("Normalized Flux", fontsize=12)
ax.legend(loc='lower left')


plt.tight_layout()

with PdfPages(pdfpath) as pdf:
    pdf.savefig(fig, bbox_inches = 'tight')
    plt.close(fig)
