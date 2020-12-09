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


def get_VectorGrad(a, c):

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

def get_ScalarGrad(a, c):

    for i in range(3):
        c[i] = ifftn_mpi(1j*(K[i]*a), c[i])
    return c

def get_Hessian(a_hat, c):

    c[0] = ifftn_mpi(-K[0]*K[0]*a_hat,c[0])
    c[1] = ifftn_mpi(-K[0]*K[1]*a_hat,c[1])
    c[2] = ifftn_mpi(-K[0]*K[2]*a_hat,c[2])
    c[3] = ifftn_mpi(-K[1]*K[1]*a_hat,c[4])
    c[4] = ifftn_mpi(-K[1]*K[2]*a_hat,c[5])
    c[5] = ifftn_mpi(-K[2]*K[2]*a_hat,c[8])
    return c

def Curl(a, c):
    c[0] = ifftn_mpi(1j*(K[1]*a[2]-K[2]*a[1]), c[0])
    c[1] = ifftn_mpi(1j*(K[2]*a[0]-K[0]*a[2]), c[1])
    c[2] = ifftn_mpi(1j*(K[0]*a[1]-K[1]*a[0]), c[2])
    return c

def get_Pressure(a, da, b):

    dU[0] = fftn_mpi(a[0]*da[0]+a[1]*da[1]+a[2]*da[2], dU[0])
    dU[1] = fftn_mpi(a[0]*da[3]+a[1]*da[4]+a[2]*da[5], dU[1])
    dU[2] = fftn_mpi(a[0]*da[6]+a[1]*da[7]+a[2]*da[8], dU[2])

    dU[:] *= dealias
    P_hat[:] = np.sum(dU*K_over_K2, 0, out=P_hat)
    b = ifftn_mpi(1j*P_hat, b)
    return b

# %%##########################################################################

os.chdir(rst)
Read_field()


if rank == 0:
    os.mkdir('Vorticity')
    os.mkdir('VGT')
    os.mkdir('Pressure')

    if solver_type == 'Scalar':
        os.mkdir('ScalarGradient')
comm.Barrier()



# Computing vorticity
Omega[:] = Curl(U_hat, Omega)

os.chdir('Vorticity')
io.savemat('Vor'+str(N)+'-p_'+str(rank)+'.mat', {
    'w1': np.reshape(Omega[0], NN), 'w2': np.reshape(Omega[1], NN),
    'w3': np.reshape(Omega[2], NN)})
os.chdir('../')


# Computing VGT
dUdX[:] = get_VectorGrad(U_hat, dUdX)

os.chdir('VGT')
io.savemat('VGT'+str(N)+'-p_'+str(rank)+'.mat', {
    'dudx': np.reshape(dUdX[0], NN), 'dudy': np.reshape(dUdX[1], NN),
    'dudz': np.reshape(dUdX[2], NN), 'dvdx': np.reshape(dUdX[3], NN),
    'dvdy': np.reshape(dUdX[4], NN), 'dvdz': np.reshape(dUdX[5], NN),
    'dwdx': np.reshape(dUdX[6], NN), 'dwdy': np.reshape(dUdX[7], NN),
    'dwdz': np.reshape(dUdX[8], NN)})
os.chdir('../')


# Computing pressure
P = get_Pressure(U, dUdX, P)

os.chdir('Pressure')
io.savemat('P'+str(N)+'-p_'+str(rank)+'.mat', {
    'P': np.reshape(P, NN**2)})


# Computing pressure Hessian
P_hat = fftn_mpi(P, P_hat)
H_ij[:] = get_Hessian(P_hat, H_ij)

io.savemat('Hess'+str(N)+'-p_'+str(rank)+'.mat', {
    'dPdxx': np.reshape(H_ij[0], NN), 'dPdxy': np.reshape(H_ij[1], NN),
    'dPdxz': np.reshape(H_ij[2], NN), 'dPdyy': np.reshape(H_ij[3], NN),
    'dPdyz': np.reshape(H_ij[4], NN), 'dPdzz': np.reshape(H_ij[5], NN)})
os.chdir('../')


# Computing scalar gradient
if solver_type == 'Scalar':

    dPhidX[:] = get_ScalarGrad(Phi_hat, dPhidX)

    os.chdir('ScalarGradient')
    io.savemat('PhiGrad'+str(N)+'-p_'+str(rank)+'.mat', {
        'dphidx': np.reshape(dPhidX[0], NN),
        'dphidy': np.reshape(dPhidX[1], NN),
        'dphidz': np.reshape(dPhidX[2], NN)})
    os.chdir('../')

#%%###########################################################################