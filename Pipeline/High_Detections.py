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
import pickle

''' similar code to SDEc.py'''
sector_number = 87
path = f"/home/gurmeher/gurmeher/TLSperiodogram/sector_{sector_number}/"
objects = []
high_detection = []
highsde_path = os.path.join(path, "highsdetic10.txt")
''' LOADING ALL PICKLE FILES INTO A LIST '''
sdethreshold = 10

# it is important to mention that we are also masking for the rotaional period through the lombcargle peridoogram but we are not doing that right now  
# that addition will be done in the automation process, after the highsdetics are already calculated

# a bit of caching 
existing_tics = set()
if os.path.exists(highsde_path):
    with open(highsde_path, "r") as f:
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
            with open(highsde_path, "a") as f:
                f.write(f"{tic_id}\n")





# import IPython; IPython.embed()
