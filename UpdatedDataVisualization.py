import os
import numpy as np
from glob import glob
from astropy.io import fits
import matplotlib.pyplot as plt


sector_number = 87
lcpaths = glob(f"/ar1/TESS/SPOC/s00{sector_number}/*.fits")

for lcpath in lcpaths:

    print(lcpath)

    hdu_list = fits.open(lcpath)
    hdr = hdu_list[0].header
    data = hdu_list[1].data
    hdu_list.close()
    tic_id = hdr.get('TICID', 'unknown')
    time = data['TIME']
    sap_flux = data['SAP_FLUX']
    pdcsap_flux = data['PDCSAP_FLUX']
    qual = data['QUALITY']
    bkgd = data['SAP_BKG'] # TODO : PLOT ME!
    savpath = f"/home/gurmeher/gurmeher/lightcurves/sector_{sector_number}/TIC_{tic_id}.pdf"
    # savpath2 = f"/home/gurmeher/gurmeher/lightcurves/sector_{sector_number}/bkgd/TIC_{tic_id}.pdf"

    if os.path.exists(savpath):
            print(f"Skipping TIC {tic_id} — already cached.")
            continue

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
    fig, axs = plt.subplots(nrows=3, figsize=(12,3), sharex = True)
    plt.subplots_adjust(hspace=0.3)
    axs[0].scatter(time[mask], sap_flux[mask]/np.nanmedian(sap_flux[mask]), c='k', s=0.5, label = 'SAP')
    axs[0].set_ylabel("Normalized SAP")
    axs[1].scatter(time[mask], pdcsap_flux[mask]/np.nanmedian(pdcsap_flux[mask]), c='k', s=0.5, label = 'PDCSAP')
    axs[1].set_ylabel("Normalized PDCSAP")
    axs[2].scatter(time[mask], bkgd[mask]/np.nanmedian(bkgd[mask]), c= 'k', s = 0.5, label = 'Background')
    axs[2].set_ylabel("Normalized Background")
    axs[2].set_xlabel('Time (BTJD)')
# added new folder inside each sector which has "fluxes" and "bkgd" so I can separately save them
# keeping the sap cutoff for the pdc_sap cutoff since they should be about the same when considering flares, wouldn't change much i believe
    for ax in axs:
        
       # initially had put the q1, q3 and everything here, caused such problems
        ax.set_xlim(time.min()-2, time.max())
        #ax.set_ylim([0, upper_bound])


    fig.savefig(savpath, bbox_inches='tight', format = "pdf")
    plt.close(fig)
   # assert 0 # remove this when not testing and actually running code