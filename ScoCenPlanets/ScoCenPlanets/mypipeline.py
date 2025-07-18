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
# creating a rapid rotators text file right now to just ensure whether all stars are actually undergoing processing.

# create a subdirectory for everything for the windowlength as well. --> add this 
# create a subdirectory for the method wotan vs notch as well.

'''***IMPORTANT*** REMEMBER! --> WHENEVER YOU ARE USING A NEW SLIDER, 
OR THE NOTCH METHOD, JUST CREATE THE SUBDIRECTORY BEFORE YOU RUN THE CODE!'''


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
    ticids = []
    if make_plots:
        highsdetic_path = f"/home/gurmeher/gurmeher/Sco-Cen-Planets/ScoCenPlanets/results/"+detrend+"/window"+ wdwstr +f"/mainlc/sector{sector_number}/highsdetic10.txt" # changed the file to highsdetic10.txt now 
        with open(highsdetic_path, 'r') as f: # opening it in read only mode
            ticids = set(line.strip() for line in f if line.strip())
    else:
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

# i get that this makes the tic list

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

def pipeline(detrender, sect_no, wdwle, make_plots = True):
    "IMPORTED EVERYTHING OUTSIDE NOW THAT I'M CHUNKING EVERYTHING"
    sector_number = sect_no
    sector_str = str(sect_no)
    sdethreshold = 10
    wdwstr = str(wdwle)
    detrend = detrender # defining the detrending method, this is not relevant rn, will get relevant when we have notch as alt.
    lcpaths = glob(f"/ar1/TESS/SPOC/s00{sector_number}/*.fits") # wherever your fits lightcurves are saved 
    #lcpaths = [ lcpaths[1] ]
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

        sap_time_binned, sap_flux_binned = bin_lightcurve(time[mask], sap_flux[mask]/np.nanmedian(sap_flux[mask]))
        pdc_time_binned, pdc_flux_binned = bin_lightcurve(time[mask], pdcsap_flux[mask]/np.nanmedian(pdcsap_flux[mask]))
        bkg_time_binned, bkg_flux_binned = bin_lightcurve(time[mask], bkgd[mask]/np.nanmedian(bkgd[mask]))

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
            flatten_lc1, trend_lc1 = flatten(sap_time_binned, sap_flux_binned, window_length = wdwl, return_trend = True, method = 'biweight')
            flatten_lc2, trend_lc2 = flatten(pdc_time_binned, pdc_flux_binned, window_length = wdwl, return_trend = True, method = 'biweight')
        elif detrend.lower() == "notch":
            continue
        

        ''' NOW WE DO THE TLS PHASE FOLDING AND PLOTTING ON A NEW GRAPH '''
        sap_time_clean, flatten_lc1_clean = clean_arrays(sap_time_binned, flatten_lc1)
        pdc_time_clean, flatten_lc2_clean = clean_arrays(pdc_time_binned, flatten_lc2)

        DEBUG and print(f"TIC {tic_id}: SAP clean time length = {len(sap_time_clean)}, flatten length = {len(flatten_lc1_clean)}")
        DEBUG and print(f"TIC {tic_id}: PDCSAP clean time length = {len(pdc_time_clean)}, flatten length = {len(flatten_lc2_clean)}")

        DEBUG and print(f"TIC {tic_id}: SAP clean time NaNs = {np.isnan(sap_time_clean).sum()}, flatten NaNs = {np.isnan(flatten_lc1_clean).sum()}")
        DEBUG and print(f"TIC {tic_id}: PDCSAP clean time NaNs = {np.isnan(pdc_time_clean).sum()}, flatten NaNs = {np.isnan(flatten_lc2_clean).sum()}")

        if len(sap_time_clean) == 0 or len(flatten_lc1_clean) == 0:
            DEBUG and print(f"TIC {tic_id}: Empty SAP arrays for TLS.")
        

        if len(pdc_time_clean) == 0 or len(flatten_lc2_clean) == 0:
            DEBUG and print(f"TIC {tic_id}: Empty PDCSAP arrays for TLS.")


        if len(sap_time_clean) < 50 or (sap_time_clean.max() - sap_time_clean.min()) < 5:
            DEBUG and print(f"TIC {tic_id}: Not enough data points or too short time span for TLS.")

            # not skipping them now though, i will let them go into the exception error and be saved in the failedtics.txt list
        
        model1 = transitleastsquares(sap_time_clean, flatten_lc1_clean)
        model2 = transitleastsquares(pdc_time_clean, flatten_lc2_clean)

        min_period = 0.5  # days, or a bit more than your cadence
        max_period = (sap_time_clean.max() - sap_time_clean.min()) / 2  # maximum orbtial period is half baseline
        DEBUG and print(max_period)


        #import IPython; IPython.embed() # --> to mess around and investigate inside the code

        # getting an error on the results module here 
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

            results1 = model1.power(period_min = min_period, period_max = max_period) # now inputting minimum and maximum period to try and fix valueError of empty TLS
            results2 = model2.power(period_min = min_period, period_max = max_period)

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

            axs[0].plot(sap_time_binned, trend_lc1, linewidth = 1.5, zorder = 1, color = 'red')
            axs[1].plot(pdc_time_binned, trend_lc2, linewidth = 1.5, zorder = 1, color = 'red')

            axs[4].scatter(sap_time_binned, flatten_lc1, s=1, color='black', label = 'Flattened SAP')
            axs[5].scatter(pdc_time_binned, flatten_lc2, s = 1, color = 'black', label = 'Flattened PDCSAP')

            for ax in axs:
                ax.legend()

                
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
                
            axs2[0].scatter(results1.folded_phase, results1.folded_y, marker = 'o', s = 0.25, color = 'black', label = f'SAP phase-folded\nTLS Period = {period1:.4f} d\nSDE = {sde1:.2f}')
            axs2[0].plot(results1.model_folded_phase, results1.model_folded_model, color = 'red', label = 'TLS MODEL for SAP Flux')
            axs2[0].set_title(f" TLS result algorithm on TIC {tic_id}")
            axs2[1].scatter(results2.folded_phase, results2.folded_y, marker = 'o', s = 0.25, color = 'black', label = f'PDCSAP phase-folded\nTLS Period = {period2:.4f} d\nSDE = {sde2:.2f}')
            axs2[1].plot(results2.model_folded_phase, results2.model_folded_model, color = 'red', label = 'TLS MODEL for PDCSAP Flux')

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
    
            



            

