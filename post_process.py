#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import sys
import os
from scipy import io

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
IC_in = sys.argv[2]
N_in = sys.argv[3]
nout_in = sys.argv[4]


inp = int(IC_in)
N = int(N_in)

K_cut = np.sqrt(2.0)/3.0
Kmax = K_cut*N


dirname = 'Sim_'+sim_in+'-IC_'+IC_in
dirpath = os.path.join(os.getcwd(),dirname)
rst = os.path.join(dirpath,'Out_'+nout_in)


# %%##########################################################################

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()

Np = N//nproc
N2 = N//2+1
NN = Np*N**2

U = np.empty((3, Np, N, N))
dUdX = np.empty((9, Np, N, N))
H_ij = np.empty((6, Np, N, N))
U_hat = np.empty((3, N, Np, N2), dtype=complex)
dU = np.empty((3, N, Np, N2), dtype=complex)
Uc_hat = np.empty((N, Np, N2), dtype=complex)
Uc_hatT = np.empty((Np, N, N2), dtype=complex)
P = np.empty((Np, N, N))
P_hat = np.empty((N, Np, N2), dtype=complex)
Omega = np.empty((3, Np, N, N))

if solver_type == 'Scalar':
    Phi = np.empty((Np, N, N))
    dPhidX = np.empty((3, Np, N, N))
    Phi_hat = np.zeros((N, Np, N2), dtype=complex)
    
# %%##########################################################################