#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from time import time
from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2
from mpi4py import MPI


import numpy as np
from numpy import linalg as LA
import sys
import os
from scipy import io


try:
    from pyfftw.interfaces.numpy_fft import fft, ifft, irfft2, rfft2
    import pyfftw
    pyfftw.interfaces.cache.enable()

except ImportError:
    pass