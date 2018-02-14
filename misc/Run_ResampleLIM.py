__author__ = 'wperkins'

import pylim.DataTools as DT
import pylim.LIM as LIM
import tables as tb
import os

if os.name == 'nt':
    filename = r'G:\Research\Hakim Research\data\ccsm4_last_millennium\tas_Amon_CCSM_lastmill.h5'
    outf = r'G:\Research\Hakim Research\data\pyLIM\test_lrg.h5'
    varname = 'tas'
else:
    filename = '/home/chaos2/wperkins/data/ccsm4_last_mil/tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.h5'
    outf = '/home/chaos2/wperkins/data/pyLIM/CCSM4_LastMill_0_9yr_20tr.h5'
    varname = 'tas'

wsize = 12
fcast_times = list(range(10))
num_eigs = 20
hold_chk = 0.05
trials = 20

h5f = tb.open_file(outf, 'w',
                   filters=tb.Filters(complevel=2, complib='blosc'),
                   node_cache_slots=1)
calib_obj = DT.hdf5_to_data_obj(filename, varname, h5file=h5f)
test_resample = LIM.ResampleLIM(calib_obj, wsize, fcast_times, num_eigs,
                                hold_chk, trials, h5file=h5f2)
test_resample.forecast()
test_resample.save_attrs()
h5f.close()
