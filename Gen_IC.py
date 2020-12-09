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

def gen_IC(res,Kf):
    
    PI=np.pi

    u_w=np.zeros((res**3,3),dtype=complex)
    
    wave_n=np.array([0.0,0.0,0.0])
    max_wave=int(res/2)
    ndx=0
    
    for k in range(0,res):
        for j in range(0,res):
            for i in range(0,res):
                
                wave_n[0]=i
                if i > max_wave:
                    wave_n[0]=i-res
                wave_n[1]=j
                if j > max_wave:
                    wave_n[1]=j-res
                wave_n[2]=k
                if k > max_wave:
                    wave_n[2]=k-res
            
                k_tmp=LA.norm(wave_n, ord=2)
                Esp=k_tmp
                
                theta=np.random.uniform(0.0,2*PI,2)
                psi=np.random.uniform(0.0,2*PI)
                
                phs1=np.exp(1j*theta[0])
                phs2=np.exp(1j*theta[1])
                Amp=1.0/(4.0*PI)
                
                if Esp <= Kf:
                    A1=np.sqrt(Amp/Kf**3)
                    alpha=A1*np.cos(psi)*phs1
                    beta=A1*np.sin(psi)*phs2
                else:
                    A1=np.sqrt(Amp*(Kf**2/Esp**11)**(1.0/3.0))
                    alpha=A1*np.cos(psi)*phs1
                    beta=A1*np.sin(psi)*phs2
                
                den12=Esp*np.sqrt(wave_n[0]**2+wave_n[1]**2)
                den3=Esp
                
                if den3 == 0.0:
                    den3 = 1.0
                    
                if den12 == 0.0:
                    den12 = 1.0
                    
                u_w[ndx,0]=(alpha*Esp*wave_n[1]+alpha*wave_n[0]*wave_n[2])/den12
                u_w[ndx,1]=(beta*wave_n[1]*wave_n[2]-alpha*Esp*wave_n[0])/den12
                u_w[ndx,2]=beta*np.sqrt(wave_n[0]**2+wave_n[1]**2)/den3
                
                ndx +=1
        
    u1_w=u_w[:,0].reshape(res,res,res)
    u2_w=u_w[:,1].reshape(res,res,res)
    u3_w=u_w[:,2].reshape(res,res,res)
    
    
    u=np.real(np.fft.ifftn(u1_w,axes=(0,1,2)))
    v=np.real(np.fft.ifftn(u2_w,axes=(0,1,2)))
    w=np.real(np.fft.ifftn(u3_w,axes=(0,1,2)))
    
    return u,v,w

# %%##########################################################################

U[0], U[1], U[2] = gen_IC(N,kf)

U *= N**3

for i in range(3):
    U_hat[i] = fftn_mpi(U[i], U_hat[i])


U_hat[:] -= (K[0]*U_hat[0]+K[1]*U_hat[1]+K[2]*U_hat[2])*K_over_K2

if rank == 0:
    U_hat[:, 0, 0, 0] = 0.0


for i in range(3):
    U[i] = ifftn_mpi(U_hat[i], U[i])


# %%##########################################################################
        
TKE = 0.5*np.mean(U[0]**2+U[1]**2+U[2]**2)

dUdX[0] = ifftn_mpi(1j*K[0]*U_hat[0],dUdX[0])
dUdX[1] = ifftn_mpi(1j*K[1]*U_hat[0],dUdX[1])
dUdX[2] = ifftn_mpi(1j*K[2]*U_hat[0],dUdX[2])

dUdX[3] = ifftn_mpi(1j*K[0]*U_hat[1],dUdX[3])
dUdX[4] = ifftn_mpi(1j*K[1]*U_hat[1],dUdX[4])
dUdX[5] = ifftn_mpi(1j*K[2]*U_hat[1],dUdX[5])

dUdX[6] = ifftn_mpi(1j*K[0]*U_hat[2],dUdX[6])
dUdX[7] = ifftn_mpi(1j*K[1]*U_hat[2],dUdX[7])
dUdX[8] = ifftn_mpi(1j*K[2]*U_hat[2],dUdX[8])

Sk = np.zeros(3)  
Fl = np.zeros(3)


Sk[0] = np.mean(dUdX[0]**3)/np.mean(dUdX[0]**2)**1.5
Sk[1] = np.mean(dUdX[4]**3)/np.mean(dUdX[4]**2)**1.5
Sk[2] = np.mean(dUdX[8]**3)/np.mean(dUdX[8]**2)**1.5

Fl[0] = np.mean(dUdX[0]**4)/np.mean(dUdX[0]**2)**2
Fl[1] = np.mean(dUdX[4]**4)/np.mean(dUdX[4]**2)**2
Fl[2] = np.mean(dUdX[8]**4)/np.mean(dUdX[8]**2)**2


f1 = open('Turb_Stats.txt', 'a')
print(format(TKE, 'g'), 
      format(np.mean(U[0]), 'g'),
      format(np.mean(U[1]), 'g'),
      format(np.mean(U[2]), 'g'),
      format(np.mean(U[0]**3)/np.mean(U[0]**2)**1.5, 'g'),
      format(np.mean(U[1]**3)/np.mean(U[1]**2)**1.5, 'g'),
      format(np.mean(U[2]**3)/np.mean(U[2]**2)**1.5, 'g'),
      format(np.mean(U[0]**4)/np.mean(U[0]**2)**2, 'g'),
      format(np.mean(U[1]**4)/np.mean(U[1]**2)**2, 'g'),
      format(np.mean(U[2]**4)/np.mean(U[2]**2)**2, 'g'),
      sep=" ", end='\n', file = f1, flush=False)
f1.close()

f2 = open('VGT_Moments.txt', 'a')
print(format(Sk[0], 'g'), format(Sk[1], 'g'), format(Sk[2], 'g'),
      format(Fl[0], 'g'), format(Fl[1], 'g'), format(Fl[2], 'g'), 
      sep=" ", end='\n', file = f2, flush=False)
f2.close()

# %%##########################################################################