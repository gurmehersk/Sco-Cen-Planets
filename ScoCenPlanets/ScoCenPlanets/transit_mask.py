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
from wotan import flatten
from transitleastsquares import transit_mask 
import sys 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transit_mask(time, flux, period, duration, T0):
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
    intransit = transit_mask(time, period, duration, T0)
    time_masked = time[~intransit]
    flux_masked = flux[~intransit]

    return time_masked, flux_masked




if __name__ == "__main__":

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
        logger.error("Data length mismatch after masking, exiting....")
        sys.exit(1)
    
    prot = nb.Lombscargle(time, normalized_flux)

    wdwle = 15
    window_length = (wdwle/100.0)*prot

    flatten_lc2, trend_lc2 = flatten(time, pdcsap_flux/np.nanmedian(pdcsap_flux), window_length = window_length, return_trend = True, method = 'biweight')
    pdc_time_clean, flatten_lc2_clean = mp.clean_arrays(time, flatten_lc2)

    real_binned_time, real_flux_binned = nb.bin_lightcurve(time, normalized_flux)

    period = 4.64423 # found from mcmc fit
    t0 = 3803.24126 # found from mcmc fit
    duration = 0.14583 # 3.5 hours, in days
    transit_model = "tls"

    time_masked, flux_masked = transit_mask(pdc_time_clean, flatten_lc2_clean, period, duration, t0)

    try:
        model2 = tls(time_masked, flux_masked)
        min_period = 0.5  # days, or a bit more than your cadence
        max_period = (time_masked.max() - time_masked.min()) / 2
    
    except Exception as e:
        logger.error("Error initializing TLS model: {}".format(e))
        sys.exit(1)

    if transit_model == "tls":
        results2 = model2.power(period_min = min_period, period_max = max_period)

        tls_period = results2.period
        tls_t0 = results2.T0

        sde = results2.SDE

        logger.info(f"TLS detected period: {tls_period} days with SDE: {sde}")
        logger.info(f"TLS detected T0: {tls_t0} days")

        epochs = np.arange(-1000, 1000)
        transit_times = tls_t0 + (tls_period * epochs)
        # Compute a y-position slightly below the light curve's minimum flux
        
        y_marker = np.nanmin(flatten_lc2) - 0.005  # or adjust the offset


        in_transit = (transit_times > time_masked.min()) & (transit_times < time_masked.max())
        visible_transits = transit_times[in_transit]
        ticker = True
        harmonic = 0
        possible_harmonics = [0.25, 0.5, 1, 2, 3, 4]
        for h in possible_harmonics:
            harmonic_checker = ((np.abs(prot-(h*tls_period)))/prot)*100
            if harmonic_checker <= 1:
                ticker = False
                harmonic = h
                break

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
            """)
        
        subplotter, plaxes = plt.subplot_mosaic(mosaic, figsize=(19, 14 ))

        plt.subplots_adjust(hspace=0.3)

        plaxes['A'].set_title(f"DETRENDING TIC {tic_id}")
        plaxes['A'].scatter(real_binned_time, real_flux_binned, zorder =2,  s = 5, color = 'black')
        plaxes['A'].set_xticks([])

        plaxes['A'].plot(time, trend_lc2, linewidth = 2, zorder = 1, color = 'red')

        time_flatten2_binned, flatten_lc2_binned = mp.bin_lightcurve(time, flatten_lc2)

        plaxes['B'].scatter(time_flatten2_binned, flatten_lc2_binned, s = 1.5, color = 'black')

        for t in visible_transits:
            plaxes['A'].scatter(t, y_marker, marker='^', color='blue', s=20, zorder=3, label='Transit time' if t==visible_transits[0] else "")

        period2 = results2.period
        sde2 = results2.SDE

        periods = results2.periods
        power = results2.power

        plaxes['C'].scatter(results2.folded_phase, results2.folded_y, marker = 'o', zorder =1, s = 0.25, color = 'black')
        plaxes['C'].plot(results2.model_folded_phase, results2.model_folded_model, zorder =3, linewidth = 1, color = 'red')


        outpath = "/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/ScoCenPlanets/transit_mask.pdf"
        subplotter.tight_layout()
        subplotter.savefig(outpath, bbox_inches='tight', format='pdf')
        plt.close(subplotter)