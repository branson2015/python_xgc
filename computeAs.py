import os, sys, math

import adios2
import numpy as np
from numpy.fft import fft, ifft
from matplotlib.tri import Triangulation, LinearTriInterpolator
import matplotlib.pyplot as plt
import xgc
import string

PHI_END=32

def computeAndSaveAs(directory, inFile, outFile) :
    print('loading: ', inFile)
    os.system(f'rm {os.path.join(directory, "xgc.3d.00001.bp")}')
    os.system(f'ln -s %s {os.path.join(directory, "./xgc.3d.00001.bp")}' % inFile)

    phi_start=0
    phi_end=PHI_END-1
    t_start = 1
    t_end = 1
    dT = 1

    fileDir = directory

    loader=xgc.load(fileDir,phi_start=phi_start,phi_end=phi_end,t_start=t_start,t_end=t_end,dt=dT,skiponeddiag=True, skip_fluc=False)
    print('loading As:', loader.As.shape, np.min(loader.As), np.max(loader.As))

    Ri=np.linspace (loader.Rmin,loader.Rmax,400)
    Zi=np.linspace (loader.Zmin,loader.Zmax,400)
    RI,ZI=np.meshgrid(Ri,Zi)
    #interpolate using the TriInterpolator class. Should be what tricontourf() uses
    triObj=Triangulation(loader.RZ[:,0],loader.RZ[:,1],loader.tri)
    sepInds = np.where(np.abs(loader.psin-1.0)<1e-4)[0]

    dAs        = loader.GradAll(loader.As[:,:,0])
    As_phi_ff  = loader.conv_real2ff(loader.As[:,:,0])
    dAs_phi_ff = -loader.conv_real2ff(dAs)

    f = adios2.open(outFile, 'w')
    nphi  = dAs_phi_ff.shape[0]
    nnode = dAs_phi_ff.shape[1]
    f.write('nphi',np.array([nphi]))
    f.write('nnode',np.array([nnode]))

    dum = np.ascontiguousarray(As_phi_ff[:,:,:,0])
    f.write('As_phi_ff',dum, dum.shape, [0]*len(dum.shape), dum.shape)
    dum = np.ascontiguousarray(dAs_phi_ff)
    f.write('dAs_phi_ff',dum, dum.shape, [0]*len(dum.shape), dum.shape)
    f.close()


directory = sys.argv[1]
inFile = os.path.join(directory, sys.argv[2])
idx = inFile.rfind('.bp')
outFile = os.path.join(directory, inFile[:idx] + '.As' + '.bp')

computeAndSaveAs(directory, inFile, outFile)
