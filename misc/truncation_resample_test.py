__author__ = 'wperkins'
import os
import sys

import netCDF4 as ncf
import numpy as np
import tables as tb
from spharm import Spharmt, regrid

import pylim.DataTools as DT
import pylim.old.LIMTools as LT
from pylim import LIM

sys.path.append('/home/disk/p/wperkins/Research/LMR/')


def regrid_sphere(nlat,nlon,Nens,X):

    """
    Truncate lat,lon grid to another resolution in spherical harmonic space. Triangular truncation

    Inputs:
    nlat            : number of latitudes
    nlon            : number of longitudes
    Nens            : number of ensemble members
    X               : data array of shape (nlat*nlon,Nens)
    ntrunc          : triangular truncation (e.g., use 42 for T42)

    Outputs :
    lat_new : 2D latitude array on the new grid (nlat_new,nlon_new)
    lon_new : 2D longitude array on the new grid (nlat_new,nlon_new)
    X_new   : truncated data array of shape (nlat_new*nlon_new, Nens)
    """
    # Originator: Greg Hakim
    #             University of Washington
    #             May 2015

    # create the spectral object on the original grid
    specob_lmr = Spharmt(nlon,nlat,gridtype='regular',legfunc='computed')

    # truncate to a lower resolution grid (triangular truncation)
    # ifix = np.remainder(ntrunc,2.0).astype(int)
    # nlat_new = ntrunc + ifix
    # nlon_new = int(nlat_new*1.5)
    ntrunc = 42
    nlat_new = 64
    nlon_new = 128

    # create the spectral object on the new grid
    specob_new = Spharmt(nlon_new,nlat_new,gridtype='regular',legfunc='computed')

    # create new lat,lon grid arrays
    dlat = 90./((nlat_new-1)/2.)
    dlon = 360./nlon_new
    veclat = np.arange(-90.,90.+dlat,dlat)
    veclon = np.arange(0.,360.,dlon)
    blank = np.zeros([nlat_new,nlon_new])
    lat_new = (veclat + blank.T).T
    lon_new = (veclon + blank)

    # transform each ensemble member, one at a time
    X_new = np.zeros([nlat_new*nlon_new,Nens])
    for k in range(Nens):
        X_lalo = np.reshape(X[:,k],(nlat,nlon))
        Xbtrunc = regrid(specob_lmr, specob_new, X_lalo, ntrunc=nlat_new-1, smooth=None)
        vectmp = Xbtrunc.flatten()
        X_new[:,k] = vectmp

    return X_new,lat_new,lon_new

if os.name == 'nt':
    pass
else:
    filename = '/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc'
    outf = '/home/chaos2/wperkins/data/pyLIM/truncated_20CR.h5'

varname = 'air'
fcast_times = list(range(1, 10))
hold_chk = 0.10
trials = 20
lag1 = True
detrend = False

hf5 = tb.open_file(outf, 'w', filters=tb.Filters(complevel=2,
                                                 complib='blosc'))
reanal_dat = ncf.Dataset(filename, 'r')
lat = reanal_dat.variables['lat']
lon = reanal_dat.variables['lon']
dat = reanal_dat.variables[varname]




calib_object = DT.BaseDataObject()




wsize = 12
num_eigs = 20
test_resample = LIM.ResampleLIM(calib_obj, wsize, fcast_times, num_eigs,
                                hold_chk, trials, h5file=hf5,
                                detrend_data=detrend)
test_resample.forecast(use_lag1=lag1)
test_resample.save_attrs()
LT.fcast_corr(h5file=hf5)
hf5.close()
