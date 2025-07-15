import os, socket
from ScoCenPlanets import __path__

DATADIR = os.path.join(os.path.dirname(__path__[0]), 'data')
RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')
# just to clarify here, if inside results we have more subdirectories to organize stuff like lightcurves, images, etc., 

''' 
Do we do 

RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results/lightcurves/')
etc.? 

After checking a little bit more, i guess the lightcurves part would/should be added in the outpath of the pipeline code, not here.
Is that right?

'''

dirs = [DATADIR, RESULTSDIR]
for d in dirs:
    if not os.path.exists(d):
        os.mkdir(d)