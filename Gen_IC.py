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

# %%##########################################################################

sim_in = sys.argv[1] 
N_in = sys.argv[2]
N = int(N_in)
Kf_in = sys.argv[3]
kf = float(Kf_in)
nproc_in = sys.argv[4]
nproc = int(nproc_in)

dirpath = os.path.join(os.getcwd(),'N_'+N_in+'-Kf_'+Kf_in+'-num_'+sim_in)
os.mkdir(dirpath)
os.chdir(dirpath)

# %%##########################################################################

comm = MPI.COMM_WORLD
n_proc = comm.Get_size()
rank = comm.Get_rank()

Np = N//n_proc
N2 = N//2+1
NN = Np*N**2


U = np.empty((3, Np, N, N))
dUdX = np.empty((9, Np, N, N))
U_hat = np.empty((3, N, Np, N2), dtype=complex)
dU = np.empty((3, N, Np, N2), dtype=complex)
Uc_hat = np.empty((N, Np, N2), dtype=complex)
Uc_hatT = np.empty((Np, N, N2), dtype=complex)