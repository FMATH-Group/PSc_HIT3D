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

kx = fftfreq(N, 1./N)
kz = kx[:N2].copy()
kz[-1] *= -1
K = np.array(np.meshgrid(kx, kx[rank*Np:(rank+1)*Np], kz, indexing='ij'),
             dtype=int)
K2 = np.sum(K*K, 0, dtype=int)
K_over_K2 = K.astype(float) / np.where(K2 == 0, 1, K2).astype(float)

kmax_dealias = 2 * K_cut * N2
dealias = np.array((np.abs(K[0]) < kmax_dealias)*(np.abs(K[1]) < kmax_dealias)*
                   (np.abs(K[2]) < kmax_dealias), dtype=bool)

# %%##########################################################################

def fftn_mpi(u, fu):

    Uc_hatT[:] = rfft2(u, axes=(1, 2))
    fu[:] = np.rollaxis(Uc_hatT.reshape(Np, nproc, Np, N2),
                        1).reshape(fu.shape)
    comm.Alltoall(MPI.IN_PLACE, [fu, MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis=0)
    return fu

def ifftn_mpi(fu, u):

    Uc_hat[:] = ifft(fu, axis=0)
    comm.Alltoall(MPI.IN_PLACE, [Uc_hat, MPI.DOUBLE_COMPLEX])
    Uc_hatT[:] = np.rollaxis(Uc_hat.reshape((nproc, Np, Np, N2)),
                             1).reshape(Uc_hatT.shape)
    u[:] = irfft2(Uc_hatT, axes=(1, 2))
    return u

#%%###########################################################################

def Read_field():

    # Read the velocity field
    uu = io.loadmat('Vel'+str(N)+'-p_'+str(rank)+'.mat')

    for i in range(3):
        U[i] = np.reshape(uu['u'+str(i+1)][0,:].T,(Np,N,N))
        U_hat[i] = fftn_mpi(U[i], U_hat[i])

    # Read the passive scalar field
    if solver_type == 'Scalar':
        pp = io.loadmat('Phi'+str(N)+'-p_'+str(rank)+'.mat')

        Phi = np.reshape(pp['phi'][0,:].T,(Np,N,N))
        Phi_hat[:] = fftn_mpi(Phi, Phi_hat)

    return

# %%##########################################################################