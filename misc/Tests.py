import pylim.DataTools as DT
import pylim.LIM as LIM
import tables as tb
import os

# filename = '/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc'
# outf = '/home/chaos2/wperkins/data/pyLIM/test.h5'
# outf2 = '/home/chaos2/wperkins/data/pyLIM/test2.h5'
# varname = 'air'

if os.name == 'nt':
    filename = r'G:\Research\Hakim Research\data\ccsm4_last_millennium\tas_Amon_CCSM_lastmill.h5'
    # filename = r'G:\Research\Hakim Research\data\20CR\air.2m.mon.mean.nc'
    outf = r'G:\Research\Hakim Research\data\pyLIM\test_lrg.h5'
    outf2 = r'G:\Research\Hakim Research\data\pyLIM\test2_lrg.h5'
    varname = 'tas'
else:
    filename = '/home/chaos2/wperkins/data/ccsm4_last_mil/tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.nc'
    outf = '/home/chaos2/wperkins/data/pyLIM/test_lrg.h5'
    outf2 = '/home/chaos2/wperkins/data/pyLIM/test2_lrg.h5'
    varname = 'tas'

# h5f = tb.open_file(outf, 'w', filters=tb.Filters(complevel=0,
#                                                  complib='blosc'))
# calib_obj = DT.hdf5_to_data_obj(filename, varname, h5file=h5f)
# test_lim = LIM.LIM(calib_obj, 12, [0, 1], 20, h5file=h5f)
# fcast_obj = DT.BaseDataObject(calib_obj.anomaly[0:16*12],
#                               force_flat=True,
#                               dim_coords={'time': (0, range(16*12))},
#                               is_run_mean=True,
#                               is_anomaly=True)
# test_lim.forecast(fcast_obj)
# h5f.close()


h5f2 = tb.open_file(outf2, 'w', filters=tb.Filters(complevel=0,
                                                   complib='blosc'))
calib_obj = DT.hdf5_to_data_obj(filename, varname, h5file=h5f2)
# calib_obj = DT.netcdf_to_data_obj(filename, 'air', h5file=h5f2)
wsize = 12
fcast_times = [0, 1]
num_eigs = 20,
hold_chk = 0.1
trials = 2
test_resample = LIM.ResampleLIM(calib_obj, wsize, fcast_times, num_eigs,
                                hold_chk, trials, h5file=h5f2)
test_resample.forecast()
h5f2.close()
