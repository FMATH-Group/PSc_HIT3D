#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import sys
import os
from scipy import io

from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2
from mpi4py import MPI

try:
    from pyfftw.interfaces.numpy_fft import fft, ifft, irfft2, rfft2
    import pyfftw
    pyfftw.interfaces.cache.enable()

except ImportError:
    pass

# %%##########################################################################


solver_type = sys.argv[-1]

sim_in = sys.argv[1]
end_time_in = sys.argv[2]
IC_in = sys.argv[3]
Out_freq_in = sys.argv[4]
Stats_freq_in = sys.argv[5]
nu_in = sys.argv[7]
Kf_in = sys.argv[8]
forcing_type = sys.argv[9]
N_in = sys.argv[10]
dt_in = sys.argv[11]
if_writeP = sys.argv[12]
nout_in = sys.argv[13]

inp = int(IC_in)
nu = float(nu_in)
end_time = float(end_time_in)
Out_freq = int(Out_freq_in)
Stats_freq = int(Stats_freq_in)
dt = float(dt_in)
N = int(N_in)
kf = float(Kf_in)
if_forced = bool(kf)
nout = int(nout_in)

dt_dx = dt*(2*np.pi)/N
K_cut = np.sqrt(2.0)/3.0
Kmax = K_cut*N

if solver_type == 'HIT':
    mag = float(sys.argv[6])


elif solver_type == 'Scalar':
    Sc = float(sys.argv[6])

    beta = 1.0
    gamma = nu/Sc
    frc_dir = 1


dirname = 'Sim_'+sim_in+'-IC_'+IC_in
dirpath = os.path.join(os.getcwd(),dirname)
rst = os.path.join(dirpath,'Restart')


# %%##########################################################################