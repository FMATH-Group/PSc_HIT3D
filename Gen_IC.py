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


# %%##########################################################################

kx = fftfreq(N, 1./N)
kz = kx[:N2].copy()
kz[-1] *= -1
K = np.array(np.meshgrid(kx, kx[rank*Np:(rank+1)*Np], kz, indexing='ij'),
             dtype=int)
K2 = np.sum(K*K, 0, dtype=int)
K_over_K2 = K.astype(float) / np.where(K2 == 0, 1, K2).astype(float)


# %%##########################################################################

def fftn_mpi(u, fu):

    Uc_hatT[:] = rfft2(u, axes=(1, 2))
    fu[:] = np.rollaxis(Uc_hatT.reshape(Np, n_proc, Np, N2),
                        1).reshape(fu.shape)
    comm.Alltoall(MPI.IN_PLACE, [fu, MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis=0)
    return fu

def ifftn_mpi(fu, u):

    Uc_hat[:] = ifft(fu, axis=0)
    comm.Alltoall(MPI.IN_PLACE, [Uc_hat, MPI.DOUBLE_COMPLEX])
    Uc_hatT[:] = np.rollaxis(Uc_hat.reshape((n_proc, Np, Np, N2)),
                             1).reshape(Uc_hatT.shape)
    u[:] = irfft2(Uc_hatT, axes=(1, 2))
    return u

#%%###########################################################################