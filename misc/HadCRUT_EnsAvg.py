__author__ = 'wperkins'

import tables as tb
import netCDF4 as nc4
import numpy as np


files = '/home/chaos2/wperkins/data/HadCRUT/HadCRUT.4.3.0.0.anomalies.{:d}.nc'
outfile = '/home/chaos2/wperkins/data/HadCRUT/HadCRUT.4.3.0.0.ens_avg_anom.h5'
num_members = 100

# Load initial data from a single ensemble member
f = nc4.Dataset(files.format(1), 'r')
lats = f.variables['latitude'][:]
lons = f.variables['longitude'][:]
time = f.variables['time'][:]
f.close()

#Create h5 output locations
outf = tb.open_file(outfile, mode='w', title='HadCRUT 4.3.0.0 Ensemble average anomalies',
                    filters=tb.Filters(complevel=2, complib='blosc'))

latout = outf.create_carray(outf.root, 'lats',
                            atom=tb.Atom.from_dtype(lats.dtype),
                            shape=lats.shape)
lonout = outf.create_carray(outf.root, 'lons',
                   atom=tb.Atom.from_dtype(lons.dtype),
                   shape=lons.shape)
timout = outf.create_carray(outf.root, 'time',
                   atom=tb.Atom.from_dtype(time.dtype),
                   shape=time.shape)
latout[:] = lats
lonout[:] = lons
timout[:] = time

# Output for individual members
ens_t_anoms = np.ma.empty((num_members, time.size, lats.size, lons.size), dtype=np.float32)
ens_anom_out = outf.create_carray(outf.root, 'ens_members',
                   atom=tb.Atom.from_dtype(ens_t_anoms.dtype),
                   shape=ens_t_anoms.shape)


for i in xrange(0, num_members):
    f = nc4.Dataset(files.format(i+1))
    T_anom = f.variables['temperature_anomaly'][:]
    ens_t_anoms[i] = T_anom
    ens_anom_out[i] = T_anom.filled(np.nan)
    f.close()

# Calc mean and std
ens_avgT = ens_t_anoms.mean(axis=0)
ens_stdT = ens_t_anoms.std(axis=0, ddof=1)

avgT_out = outf.create_carray(outf.root,
                            'ens_avgT',
                            title='{}-member Ensemble Average Temp'.format(num_members),
                            atom=tb.Atom.from_dtype(ens_avgT.dtype),
                            shape=ens_avgT.shape)
stdT_out = outf.create_carray(outf.root,
                            'ens_stdT',
                            title='{}-member Ensemble Average Temp'.format(num_members),
                            atom=tb.Atom.from_dtype(ens_stdT.dtype),
                            shape=ens_stdT.shape)
avgT_out[:] = ens_avgT.filled(np.nan)
stdT_out[:] = ens_stdT.filled(np.nan)

outf.close()



