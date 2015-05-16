import pylim.DataTools as DT
import pylim.LIM as LIM
import tables as tb

# filename = '/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc'
# outf = '/home/chaos2/wperkins/data/pyLIM/test.h5'
# outf2 = '/home/chaos2/wperkins/data/pyLIM/test2.h5'
# varname = 'air'

filename = '/home/chaos2/wperkins/data/ccsm4_last_mil/tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.nc'
outf = '/home/chaos2/wperkins/data/pyLIM/test_lrg.h5'
outf2 = '/home/chaos2/wperkins/data/pyLIM/test2_lrg.h5'
varname = 'tas'

h5f = tb.open_file(outf, 'w', filters=tb.Filters(complevel=0,
                                                 complib='blosc'))
h5f2 = tb.open_file(outf2, 'w', filters=tb.Filters(complevel=0,
                                                   complib='blosc'))
calib_obj = DT.netcdf_to_data_obj(filename, varname, h5file=h5f)
fcast_obj = DT.Hdf5DataObject(calib_obj.orig_data[0:16*12], h5f,
                              force_flat=True,
                              dim_coords={'time': (0, range(16*12))})
test_lim = LIM.LIM(calib_obj, 12, [0, 1], 20, h5file=h5f)
test_lim.forecast(fcast_obj)
h5f.close()

calib_obj = DT.netcdf_to_data_obj(filename, varname, h5file=h5f2)
test_resample = LIM.ResampleLIM(calib_obj, 12, [0, 1], 20, 0.1, 20, h5file=h5f2)
test_resample.forecast()
h5f2.close()