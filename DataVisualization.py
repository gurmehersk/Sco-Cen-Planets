import pandas as pd
import matplotlib.pyplot as plt
from lightkurve import search_lightcurve
import os
import numpy as np

df = pd.read_csv("/Users/gurmeherkathuria/Downloads/updated_TESS_targets.txt", comment = "#")
print(df.columns)
print(df['ticid'].dtype)
df['ticid'] = df['ticid'].astype(int).astype(str) # conversion from float to int to string because the df['ticid'] is a float, unreadable
lst = []
sector_number = 91

for index, row in df.iterrows():
    if row['sector'] == sector_number:
        tic_id = row['ticid']
        save_path = f"/Users/gurmeherkathuria/lightcurves/sector_{sector_number}/lc_TIC_{tic_id}.pdf"

        if os.path.exists(save_path):
            print(f"Skipping TIC {tic_id} — already cached.")
            continue
        r = search_lightcurve(f"TIC {tic_id}")
        if len(r) > 0:
            lc = r[0].download()
            flux = lc.flux.value
            time = lc.time.value

            q1 = np.nanpercentile(flux, 25)
            q3 = np.nanpercentile(flux, 75)
            iqr = q3 - q1

            # Define upper limit to clip flares
            upper_bound = q3 + 1.5 * iqr

            # Mask flares
            mask = flux < upper_bound
            fig = plt.figure(figsize=(16, 4))
            print(f"Creating the light curve for TIC {tic_id}")
            plt.scatter(time[mask], flux[mask]/np.nanmedian(flux[mask]), s=0.5, c='k') # didn't notice a difference with or without
            plt.xlim(time.min()-2, time.max())
            print(f'Saving image for TIC {tic_id}')
            fig.savefig(save_path, format = 'pdf')
            plt.close('all')
            #plt.show()
            #lst.append(row)
        else:
            print(f"could not create image for TIC {tic_id}")
            continue
#df_new = pd.DataFrame(lst)
# print(df_new)
       

    
