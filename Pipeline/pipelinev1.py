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

# MAIN GOAL OF THIS SETUP IS TO AUTOMATE AND MAKE A COMPREHENSIVE PDF THAT CAN GENERATE THESE PLOTS FOR SINGLE STAR
# HOWEVER, BEFORE WE START AUTOMATING, LET'S MAKE A GOOD TEMPLATE FOR WHAT WE WANT ON EACH PAGE.


'''BINNING TO 30 MINUTES'''

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


# Work in Progress on cleaning code to make them chunks of different functions to make it more coherent 
'''def wotan_flattening(best_period_SAP, pdc_time_binned, sap_time_binned, pdc_flux_binned, sap_flux_binned, axs):
    from wotan import flatten
    wdwl = 0.1 * best_period_SAP

    flatten_lc1, trend_lc1 = flatten(sap_time_binned, sap_flux_binned, window_length = wdwl, return_trend = True, method = 'biweight')
    flatten_lc2, trend_lc2 = flatten(pdc_time_binned, pdc_flux_binned, window_length = wdwl, return_trend = True, method = 'biweight')
    axs[0].plot(sap_time_binned, trend_lc1, linewidth = 2, color = 'red')
    axs[1].plot(pdc_time_binned, trend_lc2, linewidth = 2, color = 'red')

    axs[4].scatter(sap_time_binned, flatten_lc1, s=1, color='black', label = 'Flattened SAP')
    axs[5].scatter(pdc_time_binned, flatten_lc2, s = 1, color = 'black', label = 'Flattened PDCSAP')

    for ax in axs:
        ax.legend()
    #fig.savefig(savpath, bbox_inches='tight', format = "pdf")
    plt.close(fig)'''



''' AUTOMATING THE PROCESS AGAIN '''

sector_number = 91 # sector number can be changed 
lcpaths = glob(f"/ar1/TESS/SPOC/s00{sector_number}/*.fits") # accesses all the fits files in whatever path you've stored them in

for lcpath in lcpaths:
    print(lcpath)
    hdu_list = fits.open(lcpath)
    hdr = hdu_list[0].header
    data = hdu_list[1].data
    hdu_list.close()
    tic_id = hdr.get('TICID', 'unknown') # automated tic id since we don't know what the tic would be, different form pipelinev0

    ''' CACHING TO IGNORE ANY ITERATIONS THAT HAVE ALREADY OCCURED ''' 

    multipage_pdf_path = f"/home/gurmeher/gurmeher/detrending/combined_TIC_{tic_id}.pdf" # change this to a new path 
    if os.path.exists(multipage_pdf_path):
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
    #savpath = f"/home/gurmeher/gurmeher/detrending/TIC_{tic_id}.pdf" # change this to a new directory 

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

    fig, axs = plt.subplots(nrows=6, figsize=(12,18))
    plt.subplots_adjust(hspace=0.3)
    axs[0].scatter(sap_time_binned, sap_flux_binned, c='k', s=0.8, label = 'SAP')
    axs[0].set_ylabel("SAP", fontsize = 8 )
    axs[0].set_title(f"TIC {tic_id} — TESS mag = {tessmag} at Temp = {tempeff} Binned to 30 minutes", fontsize=8)

    axs[1].scatter(pdc_time_binned, pdc_flux_binned, c='k', s=0.8, label = 'PDCSAP')
    axs[1].set_ylabel("PDCSAP", fontsize = 8)


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

    #print(f'Best period : {best_period_PDCSAP}')
    #print(f'Best frequency: {best_frequency_PDCSAP}')

    axs[2].plot(frequency_SAP, power_SAP, label = 'SAP LS')
    axs[2].set_ylabel("Power", fontsize = 8)
    axs[2].set_xlabel('Frequency', fontsize = 8)

    axs[3].plot(frequency_PDCSAP, power_PDCSAP, label = 'PDC LS')
    axs[3].set_ylabel("Power", fontsize = 8)
    axs[3].set_xlabel('Frequency', fontsize = 8)


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


''' NOW WE DO THE WOTAN FLATTENING OF THE LIGHT CURVE '''

    wdwl = 0.1 * best_period_SAP

    flatten_lc1, trend_lc1 = flatten(sap_time_binned, sap_flux_binned, window_length = wdwl, return_trend = True, method = 'biweight')
    flatten_lc2, trend_lc2 = flatten(pdc_time_binned, pdc_flux_binned, window_length = wdwl, return_trend = True, method = 'biweight')
    axs[0].plot(sap_time_binned, trend_lc1, linewidth = 2, color = 'red')
    axs[1].plot(pdc_time_binned, trend_lc2, linewidth = 2, color = 'red')

    axs[4].scatter(sap_time_binned, flatten_lc1, s=1, color='black', label = 'Flattened SAP')
    axs[5].scatter(pdc_time_binned, flatten_lc2, s = 1, color = 'black', label = 'Flattened PDCSAP')

    for ax in axs:
        ax.legend()

       
    #fig.savefig(savpath, bbox_inches='tight', format = "pdf")
    plt.close(fig)

    ''' NOW WE DO THE TLS PHASE FOLDING AND PLOTTING ON A NEW GRAPH '''
    
    model1 = transitleastsquares(sap_time_binned, flatten_lc1)
    model2 = transitleastsquares(pdc_time_binned, flatten_lc2)

    results1 = model1.power() # not inputting minimum and maximum period right now, can add if required
    results2 = model2.power()

    figure2, axs2 = plt.subplots(nrows = 2, figsize = (10,12))
    plt.subplots_adjust(hspace=0.3)

    # whats the TLS found orbiital period 
    # import IPython; IPython.embed() --> to mess around and investigate inside the code

    axs2[0].scatter(results1.folded_phase, results1.folded_y, marker = 'o', s = 0.25, color = 'black', label = 'SAP binned and flattened data phase folded')
    axs2[0].plot(results1.model_folded_phase, results1.model_folded_model, color = 'red', label = 'TLS MODEL for SAP Flux')

    axs2[1].scatter(results2.folded_phase, results2.folded_y, marker = 'o', s = 0.25, color = 'black', label = 'PDCSAP binned and flattened data phase folded')
    axs2[1].plot(results2.model_folded_phase, results2.model_folded_model, color = 'red', label = 'TLS MODEL for PDCSAP Flux')

    #savpath2 = f"/home/gurmeher/gurmeher/detrending/TLS_TIC_{tic_id}.pdf"
    for ax in axs2:
        ax.legend()

    #figure2.savefig(savpath2, bbox_inches ='tight', format = 'pdf')
    plt.close(figure2)

    with PdfPages(multipage_pdf_path) as pdf:
        # Save first figure as page 1
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Save second figure as page 2
        pdf.savefig(figure2, bbox_inches='tight')
        plt.close(figure2)
