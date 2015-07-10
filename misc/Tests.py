import pylim.DataTools as DT
import pylim.LIMTools as LT
import pylim.LIM as LIM
import tables as tb
import os

CASE = 4

if CASE == 1:
    filename = '/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc'
    outf = '/home/chaos2/wperkins/data/pyLIM/20CR_anomtest_detrended.h5'
    varname = 'air'
    fcast_times = range(1, 10)
    hold_chk = 0.1
    trials = 30
    lag1 = True
    detrend = True


elif CASE == 2:
    filename = '/home/chaos2/wperkins/data/ccsm4_last_mil/tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.h5'
    outf = '/home/chaos2/wperkins/data/pyLIM/CCSM4_19fcast_50pcthold.h5'
    varname = 'tas'
    fcast_times = range(10)
    hold_chk = 0.5
    trials = 2
    lag1 = True
    detrend = False

elif CASE == 3:
    if os.name == 'nt':
        filename = r'G:\Research\Hakim Research\data\20CR\air.2m.mon.mean.nc'
        outf = r'G:\Research\Hakim Research\data\pyLIM\20CR_check.h5'
    else:
        filename = '/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc'
        outf = '/home/chaos2/wperkins/data/pyLIM/20CR_check.h5'
    varname = 'air'
    fcast_times = range(10)
    hold_chk = 0.1
    trials = 30
    lag1 = True
    detrend = False

elif CASE == 4:
    if os.name == 'nt':
        pass
    else:
        filename = '/home/chaos2/wperkins/data/ccsm4_last_mil/tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.h5'
        outf = '/home/chaos2/wperkins/data/pyLIM/CCSM4_19fcst_detrended.h5'

    varname = 'tas'
    fcast_times = range(1, 10)
    hold_chk = 0.05
    trials = 20
    lag1 = True
    detrend = False

elif CASE == 5
    filename = '/home/chaos2/wperkins/data/'

# node_cache_slots reduced for large dataset resample exp
hf5 = tb.open_file(outf, 'w', filters=tb.Filters(complevel=2,
                                                 complib='blosc'),
                   node_cache_slots=1)

if os.path.splitext(filename)[1] == '.nc':
    calib_obj = DT.netcdf_to_data_obj(filename, varname, h5file=hf5)
else:
    calib_obj = DT.hdf5_to_data_obj(filename, varname, h5file=hf5)


wsize = 12
num_eigs = 20
test_resample = LIM.ResampleLIM(calib_obj, wsize, fcast_times, num_eigs,
                                hold_chk, trials, h5file=hf5,
                                detrend_data=detrend)
test_resample.forecast(use_lag1=lag1)
test_resample.save_attrs()
LT.fcast_corr(h5file=hf5)
hf5.close()
