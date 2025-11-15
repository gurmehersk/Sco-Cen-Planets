'''
transit_mask.py

Module to run transit_mask on TIC 88297141 lightcurve.
Let's use notch this time, since we are only looking at one PC.

Update, we will use wotan, cuz notch required BLS, 
and we want to use transit_mask right now, available on TLS.

'''
from astropy.io import fits
import mypipeline as mp
import transitleastsquares as tls
import notch_binned_module as nb 
import numpy as np
import logging 
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transit_mask():



if __name__ == "__main__":
    transit_mask()    # Load lightcurve
    tic_id = 88297141

    # Load the lightcurve
    lcpath = "/ar1/TESS/SPOC/s0092/tess2025127075000-s0092-0000000088297141-0289-s_lc.fits"
    data, hdr = fits.getdata(lcpath, header=True)
    objective = "likelihood"
    
    time, tessmag, tempeff, sap_flux, pdcsap_flux, qual, bkgd = mp.get_data(data, hdr)

    time, sap_flux, pdcsap_flux, bkgd = mp.clean_data(qual, time, sap_flux, pdcsap_flux, bkgd)

    q1,q3,iqr,masks = nb.clip_masks(pdcsap_flux)
    time = time[masks]
    pdcsap_flux = pdcsap_flux[masks]
    normalized_flux = pdcsap_flux / np.nanmedian(pdcsap_flux)    

    mask2 = np.isfinite(time) & np.isfinite(normalized_flux)
    normalized_flux = normalized_flux[mask2]
    time = time[mask2]

    if len(time) == len(normalized_flux):
        logger.info("Data loaded successfully with {} points.".format(len(time)))
    else:
        logger.warning("Data length mismatch after masking.")
        return
    
    prot = nb.Lombscargle(time, normalized_flux)

    if prot > 2:
        dictionary = {"window_length" : 1}

    else :
        dictionary = {"window_length" : 0.5}

    clipped_flux = nb.slide_clip(
            time, normalized_flux, window_length=dictionary['window_length'],
            low=100, high=2, method='mad', center='median'
        )

    sel = np.isfinite(time) & np.isfinite(clipped_flux)
    pdc_time_binned = time[sel]
    pdc_flux_binned = 1.* clipped_flux[sel]
    assert len(time) == len(normalized_flux)

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

    real_binned_time, real_flux_binned = nb.bin_lightcurve(time, normalized_flux)

    plt.subplots_adjust(hspace=0.3)
    plaxes['A'].set_title(f"DETRENDING TIC {tic_id}")
    plaxes['A'].scatter(real_binned_time, real_flux_binned, zorder =2,  s = 5, color = 'black')
    plaxes['A'].set_xticks([])


    flat_flux, trend_flux, notch = nb._run_notch(time, normalized_flux, dictionary)
    time_flat, flat_flux_binned = nb.bin_lightcurve(time, flat_flux, bin_minutes=30)

    plaxes['B'].scatter(time_flat , flat_flux_binned, s = 0.5)
    plaxes['A'].plot(time, trend_flux, color = 'red', linewidth = 1.5, zorder = 1)

    transit_model = "tls"

    if transit_model == "tls":


