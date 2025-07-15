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

