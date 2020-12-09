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

def comp_VGT(a, c):

    c[0] = ifftn_mpi(1j*K[0]*a[0],c[0])
    c[1] = ifftn_mpi(1j*K[1]*a[0],c[1])
    c[2] = ifftn_mpi(1j*K[2]*a[0],c[2])

    c[3] = ifftn_mpi(1j*K[0]*a[1],c[3])
    c[4] = ifftn_mpi(1j*K[1]*a[1],c[4])
    c[5] = ifftn_mpi(1j*K[2]*a[1],c[5])

    c[6] = ifftn_mpi(1j*K[0]*a[2],c[6])
    c[7] = ifftn_mpi(1j*K[1]*a[2],c[7])
    c[8] = ifftn_mpi(1j*K[2]*a[2],c[8])
    return c

def comp_VGT_moments(a, m2, m3, m4):

    m2[0] = comm.reduce(np.mean(a[0]**2)/nproc)
    m2[1] = comm.reduce(np.mean(a[4]**2)/nproc)
    m2[2] = comm.reduce(np.mean(a[8]**2)/nproc)

    m3[0] = comm.reduce(np.mean(a[0]**3)/nproc)
    m3[1] = comm.reduce(np.mean(a[4]**3)/nproc)
    m3[2] = comm.reduce(np.mean(a[8]**3)/nproc)

    m4[0] = comm.reduce(np.mean(a[0]**4)/nproc)
    m4[1] = comm.reduce(np.mean(a[4]**4)/nproc)
    m4[2] = comm.reduce(np.mean(a[8]**4)/nproc)
    return m2, m3, m4

def comp_singlepoint(TKE, epsilon, max_V):

    # Kolmogorov length-scale
    eta = (nu**3 / epsilon)**0.25

    # r.m.s. velocity
    u_rms = (2*TKE/3.)**.5

    # Integral length-scale
    l0 = u_rms**3/epsilon

    # Taylor micro-scale
    lamb = u_rms*(15*nu/epsilon)**0.5

    # Large-scale Reynolds number
    Re_l0 = u_rms*l0/nu

    # Taylor micro-scale Reynolds number
    Re_lamb = u_rms*lamb/nu

    # Large-eddy turnover time
    turnover_t = l0/u_rms

    # compute CFL
    CFL = dt_dx * np.max(max_V)
    
    
    # Writing statistical quantities out
    f1 = open('Stats_Vel.txt', 'a')
    print(format(t, 'g'),
          format(TKE, 'g'), format(epsilon, 'g'), format(Kmax*eta, 'g'),
          format(l0, 'g'), format(Re_l0, 'g'), format(Re_lamb, 'g'),
          format(turnover_t, 'g'),
          sep=" ", end='\n', file = f1, flush=False)
    f1.close()

    # Writing out CFL record
    f2 = open('CFL.txt', 'a')
    print(format(t, 'g'), format(CFL, 'g'),
          sep=" ", end='\n', file = f2, flush=False)
    f2.close()

    return


def comp_VGT_HO_Stat(m2, m3, m4):

    Sk = np.zeros(3)
    Fl = np.zeros(3)

    # Computing Skweness and Flatness of VGT
    for i in range(3):
        Sk[i] = m3[i]/m2[i]**1.5
        Fl[i] = m4[i]/m2[i]**2

    # Writing statistical quantities out
    f1 = open('VGT_Moments.txt', 'a')
    print(format(t, 'g'),
          format(Sk[0], 'g'), format(Sk[1], 'g'), format(Sk[2], 'g'),
          format(Fl[0], 'g'), format(Fl[1], 'g'), format(Fl[2], 'g'),
          sep=" ", end='\n', file = f1, flush=False)
    f1.close()
    return


def comp_ScalarVar_Budget(VarPhi, VarPhi_old, Forcing_Flux, epsilon_phi):

    # Computing the rate of scalar variance
    dVarPhi_dt = 0.5*(VarPhi - VarPhi_old)/dt

    f1 = open('ScalarVariance_Balance.txt', 'a')
    print(format(t, 'g'),
          format(VarPhi, 'g'), format(dVarPhi_dt, 'g'),
          format(-Forcing_Flux, 'g'), format(-epsilon_phi, 'g'),
          sep=" ", end='\n', file = f1, flush=False)
    f1.close()
    return


def comp_Scalar_Stats(VarPhi, mu3_phi, mu4_phi, mu2, mu3, mu4, Flux):

    S_dphi = np.zeros(3)
    F_dphi = np.zeros(3)

    if nout > 0:
        # Computing Skweness and Flatness of scalar gradients
        for i in range(3):
            S_dphi[i] = mu3[i]/mu2[i]**1.5
            F_dphi[i] = mu4[i]/mu2[i]**2

        # Computing Skweness and Flatness of scalar fluctuations
        S_phi = mu3_phi/VarPhi**1.5
        F_phi = mu4_phi/VarPhi**2

    f1 = open('Moments_Scalar.txt', 'a')
    print(format(t, 'g'),
          format(mu2[0], 'g'), format(mu2[1], 'g'), format(mu2[2], 'g'),
          format(S_dphi[0], 'g'),
          format(S_dphi[1], 'g'),
          format(S_dphi[2], 'g'),
          format(F_dphi[0], 'g'),
          format(F_dphi[1], 'g'),
          format(F_dphi[2], 'g'),
          format(VarPhi, 'g'), format(S_phi, 'g'), format(F_phi, 'g'),
          format(Flux[0], 'g'), format(Flux[1], 'g'), format(Flux[2], 'g'),
          sep=" ", end='\n', file = f1, flush=False)
    f1.close()
    return

# %%##########################################################################

def Cross(a, b, c):
    c[0] = fftn_mpi(a[1]*b[2]-a[2]*b[1], c[0])
    c[1] = fftn_mpi(a[2]*b[0]-a[0]*b[2], c[1])
    c[2] = fftn_mpi(a[0]*b[1]-a[1]*b[0], c[2])
    return c

def Curl(a, c):
    c[2] = ifftn_mpi(1j*(K[0]*a[1]-K[1]*a[0]), c[2])
    c[1] = ifftn_mpi(1j*(K[2]*a[0]-K[0]*a[2]), c[1])
    c[0] = ifftn_mpi(1j*(K[1]*a[2]-K[2]*a[1]), c[0])
    return c

def get_Pressure(a_hat, a, b):
    curl[:] = Curl(a_hat, curl)
    dU[:] = Cross(a, curl, dU)
    dU[:] *= dealias
    P_hat[:] = np.sum(dU*K_over_K2, 0, out=P_hat)
    b = ifftn_mpi(P_hat, b)
    return b

def U_Phi(a, b, c):
    for i in range(3):
        dPhi0[i] = fftn_mpi(a[i]*b[i], dPhi0[i])
    c[:] = np.sum(dPhi0, 0, out=c)
    return c

#%%###########################################################################