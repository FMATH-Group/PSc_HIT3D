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

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
rank = comm.Get_rank()
root = 0

Np = N//nproc
N2 = N//2+1

U = np.empty((3, Np, N, N))
U_low = np.empty((3, Np, N, N))
dUdX = np.empty((9, Np, N, N))
U_hat = np.empty((3, N, Np, N2), dtype=complex)
U_hat0 = np.empty((3, N, Np, N2), dtype=complex)
U_hat1 = np.empty((3, N, Np, N2), dtype=complex)
dU = np.empty((3, N, Np, N2), dtype=complex)
Uc_hat = np.empty((N, Np, N2), dtype=complex)
Uc_hatT = np.empty((Np, N, N2), dtype=complex)
P = np.empty((Np, N, N))
P_hat = np.empty((N, Np, N2), dtype=complex)
curl = np.empty((3, Np, N, N))

if solver_type == 'Scalar':

    Phi = np.empty((Np, N, N))
    Phi_old = np.empty((Np, N, N))
    dPhidX = np.empty((3, Np, N, N))
    Phi_hat = np.zeros((N, Np, N2), dtype=complex)
    dPhi = np.empty((N, Np, N2), dtype=complex)
    dPhi0 = np.empty((3, N, Np, N2), dtype=complex)
    
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

a = np.array([1./6., 1./3., 1./3., 1./6.])
b = np.array([0.5, 0.5, 1.])


alpha = 0.0
alpha2 = 0.0

k2_mask = np.where(K2 <= kf**2, 1, 0)

# %%##########################################################################

mu2 = np.zeros(3)
mu3 = np.zeros(3)
mu4 = np.zeros(3)

if solver_type == 'Scalar':
    Phi_mean = 0.0
    Phi_mean_p = 0.0
    ScalarFlux = np.zeros(3)
    mu2_dphi = np.zeros(3)
    mu3_dphi = np.zeros(3)
    mu4_dphi = np.zeros(3)

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

# %%##########################################################################

def Stochastic_forcing():

    sigma = .5  # Standard deviation.
    mu = 10.  # Mean.
    tau = .05  # Time constant.

    T = 1000.  # Total time.
    n = int(T/dt)  # Number of time steps.

    sigma_bis = sigma * np.sqrt(2./tau)
    sqrtdt = np.sqrt(dt)

    xp = np.zeros((6,n))

    for i in range(n - 1):

        for j in range(6):

            xp[j,i + 1] = xp[j,i] + dt * (-(xp[j,i] - mu) / tau) + \
                    sigma_bis * sqrtdt * np.random.randn()


    os.chdir(rst)
    xp = np.mean(xp[:,1000::],axis=0)/mu
    io.savemat('OU_process.mat', {'OU': xp})
    os.chdir('../')

    return xp

# %%##########################################################################