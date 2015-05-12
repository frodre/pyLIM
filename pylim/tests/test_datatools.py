__author__ = 'wperkins'

import tables as tb
import numpy as np
import pytest
import os
from pylim import DataTools as Dt
from pylim.DataTools import BaseDataObject as BDO
from pylim.DataTools import BaseDataObject as BO


@pytest.fixture(scope='module')
def tb_file(request):
    tbf = tb.open_file('test.h5', 'w',
                       filters=tb.Filters(complevel=0, complib='blosc'))

    def fin():
        tbf.close()
        os.system('rm -f test.h5')

    request.addfinalizer(fin)
    return tbf

@pytest.fixture(scope='module',
                params=[BDO, BO])
def dat_obj(request):
    return request.param



def test_empty_carray_node(tb_file):
    shape = (5, 10)
    Dt.empty_hdf5_carray(tb_file, '/', 'empty',
                         tb.Atom.from_dtype(np.dtype(np.float64)),
                         shape)
    assert tb_file.__contains__('/empty')
    assert tb_file.root.empty.shape == shape


def test_empty_carray_node_createparents(tb_file):
    shape = (5, 10)
    Dt.empty_hdf5_carray(tb_file, '/path/to/node', 'empty',
                         tb.Atom.from_dtype(np.dtype(np.float64)),
                         shape,
                         createparents=True)
    assert tb_file.__contains__('/path/to/node/empty')
    assert tb_file.get_node('/path/to/node/empty').shape == shape


def test_empty_carray_node_already_exists(tb_file):
    shape = (5, 10)
    Dt.empty_hdf5_carray(tb_file, '/', 'empty',
                         tb.Atom.from_dtype(np.dtype(np.float64)),
                         shape)
    Dt.empty_hdf5_carray(tb_file, '/', 'empty',
                         tb.Atom.from_dtype(np.dtype(np.float64)),
                         shape)
    assert tb_file.__contains__('/empty')


def test_var_to_carray_node(tb_file):
    shape = (5, 10)
    data = np.arange(50).reshape(shape)
    Dt.var_to_hdf5_carray(tb_file, '/', 'data', data)
    assert tb_file.__contains__('/data')
    assert tb_file.root.data.shape == shape
    assert np.array_equal(data, tb_file.root.data[:])


def test_var_to_carray_node_createparents(tb_file):
    shape = (5, 10)
    data = np.arange(50).reshape(shape)
    Dt.var_to_hdf5_carray(tb_file, '/path/to/node', 'data', data,
                          createparents=True)
    assert tb_file.__contains__('/path/to/node/data')
    assert tb_file.get_node('/path/to/node/data').shape == shape
    assert np.array_equal(data, tb_file.root.path.to.node.data[:])


def test_var_to_carray_node_already_exists(tb_file):
    shape = (5, 10)
    data1 = np.arange(50).reshape(shape)
    data2 = np.arange(50, 100).reshape(shape)

    Dt.var_to_hdf5_carray(tb_file, '/', 'data', data1)
    Dt.var_to_hdf5_carray(tb_file, '/', 'data', data2)
    assert tb_file.__contains__('/data')
    assert np.array_equal(tb_file.root.data[:], data2)


#### DataObject Tests ####

## When no dimensions are provided 1D-3D data could have spatial and temporal
## components.  Should do not change shape or data.  4D data we do not know
## the time dimension so it should also be left alone by default.
@pytest.mark.parametrize("shape",
    [(24),
    ((4, 6)),
    ((4, 3, 2)),
    ((2, 2, 3, 2)),
    pytest.mark.xfail(((1, 2, 2, 3, 2))),
    ])
def test_basedataobj_data_nodim(shape, dat_obj):
    data = np.arange(np.product(np.array(shape))).reshape(shape)
    obj = dat_obj(data)
    assert np.array_equal(obj.data, data)
    assert obj.data.shape == data.shape
    assert obj._full_shp == data.shape


def test_basedataobj_data_force_flat(dat_obj):
    data = np.arange(20).reshape(2, 2, 5)
    obj = dat_obj(data, force_flat=True)
    assert np.array_equal(data.flatten(), obj.data[:])
    assert obj.data.shape == data.flatten().shape


def test_baseddataobj_dim_matching(dat_obj):
    data = np.arange(24).reshape(2, 2, 3, 2)
    coords = {dat_obj.TIME: (0, [1, 2]),
              dat_obj.LEVEL: (1, [1000, 900]),
              dat_obj.LAT: (2, [45, 50, 55]),
              dat_obj.LON: (3, [-80, -90])}
    obj = dat_obj(data, dim_coords=coords)
    assert obj._leading_time
    assert obj._dim_idx == {key: value[0] for key, value in coords.items()}
    assert data.shape[0] == obj._time_shp[0]
    assert data.shape[1:] == obj._spatial_shp
    assert data.shape == obj.data.shape

@pytest.mark.xfail
def test_baseddataobj_dim_noleadsample(dat_obj):
    data = np.arange(24).reshape(2, 2, 3, 2)
    coords = {dat_obj.TIME: (2, [1, 2, 3]),
              dat_obj.LEVEL: (1, [1000, 900]),
              dat_obj.LAT: (0, [50, 55]),
              dat_obj.LON: (3, [-80, -90])}
    obj = dat_obj(data, dim_coords=coords)


def test_baseddataobj_dim_mismatched(dat_obj):
    data = np.arange(24).reshape(2, 2, 3, 2)
    coords = {dat_obj.TIME: (0, [1, 2]),
              dat_obj.LEVEL: (1, [1000, 900, 800]),
              dat_obj.LAT: (2, [45, 50, 55]),
              dat_obj.LON: (3, [-80, -90, 100])}
    obj = dat_obj(data, dim_coords=coords)
    assert len(obj._dim_idx.keys()) == 2
    assert dat_obj.TIME in obj._dim_idx
    assert dat_obj.LAT in obj._dim_idx


def test_baseddataobj_dim_notime(dat_obj):
    data = np.arange(24).reshape(2, 2, 3, 2)
    coords = {dat_obj.LEVEL: (1, [1000, 900]),
              dat_obj.LAT: (2, [45, 50, 55]),
              dat_obj.LON: (3, [-80, -90])}
    obj = dat_obj(data, dim_coords=coords)
    assert len(obj._dim_idx.keys()) == 3
    assert obj._full_shp == data.shape


def test_baseddataobj_nanentry_noleadtime(dat_obj):
    data = np.arange(24).reshape(3, 4, 2).astype(np.float32)
    data[2, 3, 1] = np.nan
    obj = dat_obj(data)
    assert obj.is_masked
    np.testing.assert_array_equal(data, obj.orig_data)
    assert np.array_equal(data.flatten()[:-1], obj.compressed_data)
    assert np.array_equal(data.flatten()[:-1], obj.data)
    np.testing.assert_array_equal(data,
                                  obj.inflate_full_grid(reshape_orig=True),
                                  err_msg='Inflation to full grid failed.')


def test_baseddataobj_nanentry_leadtime(dat_obj):
    data = np.arange(24).reshape(3, 2, 2, 2).astype(np.float32)
    data[1, 0, 0, 1] = np.nan
    data[2, 1, 1, 1] = np.nan
    dim = {dat_obj.TIME: (0, [1, 2, 3])}

    valid = np.isfinite(data[0])
    for time in data:
        valid &= np.isfinite(time)
    full_valid = np.ones_like(data, dtype=np.bool) * valid

    obj = dat_obj(data, dim_coords=dim)
    assert obj._leading_time
    assert obj.is_masked
    assert obj.data.size == 18  # remove entire nan loc from entire sample
    assert np.array_equal(obj.data, data[:, valid])

    data[~full_valid] = np.nan
    np.testing.assert_array_equal(data,
                                  obj.inflate_full_grid(reshape_orig=True))


def test_basedataobj_compressed_noleadtime(dat_obj):
    data = np.arange(24).reshape(3, 4, 2).astype(np.float32)
    data[2, 3, 1] = np.nan
    data[1, 0, 1] = np.nan
    valid = np.isfinite(data)
    comp = data[valid]
    valid = valid.flatten()
    obj = dat_obj(comp, valid_data=valid)

    assert obj.is_masked
    assert obj.data.shape == comp.shape
    assert valid.shape == tuple(obj._full_shp)
    np.testing.assert_array_equal(data.flatten(),
                                  obj.inflate_full_grid())

def test_basedataobj_compressed_leadtime(dat_obj):
    data = np.arange(24).reshape(3, 4, 2).astype(np.float32)
    data[2, 3, 1] = np.nan
    data[1, 0, 1] = np.nan
    dim = {dat_obj.TIME: (0, [1, 2, 3])}

    valid = np.isfinite(data[0])
    for time in data:
        valid &= np.isfinite(time)

    comp = data[:, valid]
    inflated = np.copy(data).reshape(3,8)
    inflated[:, ~valid.flatten()] = np.nan

    obj = dat_obj(comp, dim_coords=dim, valid_data=valid.flatten())
    assert obj._leading_time
    assert obj.is_masked
    np.testing.assert_array_equal(obj.inflate_full_grid(), inflated)





if __name__ == '__main__':
    # try:
    #     f = tb.open_file('test.h5', 'w')
    #     test_var_to_carray_node_already_exists(f)
    # finally:
    #     f.close()
    #     os.system('rm -f test.h5')
    test_basedataobj_compressed_leadtime()




