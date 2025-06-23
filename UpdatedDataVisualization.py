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
    savpath = f"/home/gurmeher/gurmeher/lightcurves/sector_{sector_number}/fluxes/TIC_{tic_id}.pdf"
    savpath2 = f"/home/gurmeher/gurmeher/lightcurves/sector_{sector_number}/bkgd/TIC_{tic_id}.pdf"

    if os.path.exists(savpath):
            print(f"Skipping TIC {tic_id} — already cached.")
            continue

    sel = (qual == 0)
    time = time[sel]
    sap_flux = sap_flux[sel]
    pdcsap_flux = pdcsap_flux[sel]
    bkgd = bkgd[sel]

    fig, axs = plt.subplots(nrows=2, figsize=(12,3))
    axs[0].scatter(time, sap_flux/np.nanmedian(sap_flux), c='k', s=0.5)
    axs[1].scatter(time, pdcsap_flux/np.nanmedian(pdcsap_flux), c='k', s=0.5)
    
# added new folder inside each sector which has "fluxes" and "bkgd" so I can separately save them
# keeping the sap cutoff for the pdc_sap cutoff since they should be about the same when considering flares, wouldn't change much i believe
    for ax in axs:
        
        q1 = np.nanpercentile(sap_flux, 25)
        q3 = np.nanpercentile(sap_flux, 75)
        iqr = q3 - q1

        # Define upper limit to clip flares
        upper_bound = q3 + 1.5 * iqr

        ax.set_xlim(time.min()-2, time.max())
        ax.set_ylim([0, upper_bound])

    fig2, axs2 = plt.subplots(nrows = 1, figsize = (12,3))
    axs2.scatter(time, bkgd/np.nanmedian(bkgd), c = 'k', s = 0.5)

    fig.savefig(savpath, bbox_inches='tight', format = "pdf")
    fig2.savefig(savpath2, format = "pdf")
    plt.close(fig)
    plt.close(fig2)
    

   # assert 0 # remove this when not testing and actually running code