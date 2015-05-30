import pylim.DataTools as DT
import pylim.LIM as LIM
import tables as tb
import os

CASE = 1

if CASE == 1:
    if os.name == 'nt':
        pass
    else:
        filename = '/home/chaos2/wperkins/data/ccsm4_last_mil/tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.h5'
        outf = '/home/chaos2/wperkins/data/pyLIM/CCSM4_LastMill_09fcst_20trial.h5'

    fcast_times = range(10)
    hold_chk = 0.05
    trials = 20


if CASE == 2:
    filename = '/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc'
    outf = '/home/chaos2/wperkins/data/pyLIM/test.h5'
    varname = 'air'
    fcast_times = range(10)
    hold_chk = 0.1
    trials = 20

# node_cache_slots reduced for large dataset resample exp
hf5 = tb.open_file(outf, 'w', filters=tb.Filters(complevel=2,
                                                 complib='blosc'),
                   node_cache_slots=1)

if os.path.splitext(filename)[1] == 'nc':
    calib_obj = DT.netcdf_to_data_obj(filename, varname, h5file=hf5)
else:
    calib_obj = DT.hdf5_to_data_obj(filename, varname, h5file=hf5)


wsize = 12
num_eigs = 20
test_resample = LIM.ResampleLIM(calib_obj, wsize, fcast_times, num_eigs,
                                hold_chk, trials, h5file=hf5)
test_resample.forecast()
test_resample.save_attrs()
hf5.close()
