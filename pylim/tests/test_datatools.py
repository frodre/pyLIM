__author__ = 'wperkins'

import tables as tb
import numpy as np
import pytest
import os
from pylim import DataTools as Dt
from pylim.DataTools import BaseDataObject as BDO
from pylim.DataTools import Hdf5DataObject as HDO


@pytest.fixture(scope='module')
def tb_file(request):
    tbf = tb.open_file('test.h5', 'w',
                       filters=tb.Filters(complevel=0, complib='blosc'))

    def fin():
        tbf.close()
        os.system('rm -f test.h5')

    request.addfinalizer(fin)
    return tbf

@pytest.fixture(params=['a', 'w'])
def writeable_tb_file(request):
    tbf = tb.open_file('test1.h5', request.param,
                       filters=tb.Filters(complevel=0, complib='blosc'))

    def fin():
        tbf.close()
        os.system('rm -f test1.h5')

    request.addfinalizer(fin)
    return tbf

@pytest.fixture()
def readonly_tb_file(request):
    tbf = tb.open_file('test2.h5', 'r',
                       filters=tb.Filters(complevel=0, complib='blosc'))

    def fin():
        tbf.close()
        os.system('rm -f test2.h5')

    request.addfinalizer(fin)
    return tbf

@pytest.fixture()
def closed_tb_file(request):
    tbf = tb.open_file('test3.h5', 'r',
                       filters=tb.Filters(complevel=0, complib='blosc'))
    tbf.close()

    def fin():
        os.system('rm -f test3.h5')

    request.addfinalizer(fin)
    return tbf

@pytest.fixture()
def hdf5_obj(tb_file):
    dat = np.arange(10)
    return HDO(dat, tb_file)



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


### BaseDataObject ####
@pytest.mark.parametrize("shape",
    [(24),
    ((4, 6)),
    ((4, 3, 2)),
    ((2, 2, 3, 2)),
    pytest.mark.xfail(((1, 2, 2, 3, 2))),
    ])
def test_basedataobj_data_nodim(shape):
    data = np.arange(np.product(np.array(shape))).reshape(shape)
    obj = BDO(data)
    assert np.array_equal(obj.data, data)
    assert obj.data.shape == data.shape
    assert obj._full_shp == data.shape


def test_basedataobj_data_force_flat():
    data = np.arange(20).reshape(2, 2, 5)
    obj = BDO(data, force_flat=True)
    assert np.array_equal(data.flatten(), obj.data[:])
    assert obj.data.shape == data.flatten().shape
    assert obj.orig_data.shape == data.flatten().shape


def test_basedataobj_dim_matching():
    data = np.arange(24).reshape(2, 2, 3, 2)
    coords = {BDO.TIME: (0, [1, 2]),
              BDO.LEVEL: (1, [1000, 900]),
              BDO.LAT: (2, [45, 50, 55]),
              BDO.LON: (3, [-80, -90])}
    obj = BDO(data, dim_coords=coords)
    assert obj._leading_time
    assert obj._dim_idx == {key: value[0] for key, value in coords.items()}
    assert data.shape[0] == obj._time_shp[0]
    assert data.shape[1:] == obj._spatial_shp
    assert data.shape == obj.data.shape

@pytest.mark.xfail
def test_basedataobj_dim_noleadsample():
    data = np.arange(24).reshape(2, 2, 3, 2)
    coords = {BDO.TIME: (2, [1, 2, 3]),
              BDO.LEVEL: (1, [1000, 900]),
              BDO.LAT: (0, [50, 55]),
              BDO.LON: (3, [-80, -90])}
    obj = BDO(data, dim_coords=coords)


def test_basedataobj_dim_mismatched():
    data = np.arange(24).reshape(2, 2, 3, 2)
    coords = {BDO.TIME: (0, [1, 2]),
              BDO.LEVEL: (1, [1000, 900, 800]),
              BDO.LAT: (2, [45, 50, 55]),
              BDO.LON: (3, [-80, -90, 100])}
    obj = BDO(data, dim_coords=coords)
    assert len(obj._dim_idx.keys()) == 2
    assert BDO.TIME in obj._dim_idx
    assert BDO.LAT in obj._dim_idx


def test_basedataobj_dim_notime():
    data = np.arange(24).reshape(2, 2, 3, 2)
    coords = {BDO.LEVEL: (1, [1000, 900]),
              BDO.LAT: (2, [45, 50, 55]),
              BDO.LON: (3, [-80, -90])}
    obj = BDO(data, dim_coords=coords)
    assert len(obj._dim_idx.keys()) == 3
    assert obj._full_shp == data.shape


def test_basedataobj_nanentry_noleadtime():
    data = np.arange(24).reshape(3, 4, 2).astype(np.float32)
    data[2, 3, 1] = np.nan
    obj = BDO(data)
    assert obj.is_masked
    np.testing.assert_array_equal(data.flatten(), obj.orig_data)
    assert np.array_equal(data.flatten()[:-1], obj.compressed_data)
    assert np.array_equal(data.flatten()[:-1], obj.data)
    np.testing.assert_array_equal(data,
                                  obj.inflate_full_grid(reshape_orig=True),
                                  err_msg='Inflation to full grid failed.')


def test_basedataobj_nanentry_leadtime():
    data = np.arange(24).reshape(3, 2, 2, 2).astype(np.float32)
    data[1, 0, 0, 1] = np.nan
    data[2, 1, 1, 1] = np.nan
    dim = {BDO.TIME: (0, [1, 2, 3])}

    valid = np.isfinite(data[0])
    for time in data:
        valid &= np.isfinite(time)
    full_valid = np.ones_like(data, dtype=np.bool) * valid

    obj = BDO(data, dim_coords=dim)
    assert obj._leading_time
    assert obj.is_masked
    assert obj.data.size == 18  # remove entire nan loc from entire sample
    assert np.array_equal(obj.data, data[:, valid])

    data[~full_valid] = np.nan
    np.testing.assert_array_equal(data,
                                  obj.inflate_full_grid(reshape_orig=True))


def test_basedataobj_compressed_noleadtime():
    data = np.arange(24).reshape(3, 4, 2).astype(np.float32)
    data[2, 3, 1] = np.nan
    data[1, 0, 1] = np.nan
    valid = np.isfinite(data)
    comp = data[valid]
    valid = valid.flatten()
    obj = BDO(comp, valid_data=valid)

    assert obj.is_masked
    assert obj.data.shape == comp.shape
    assert valid.shape == tuple(obj._full_shp)
    np.testing.assert_array_equal(data.flatten(),
                                  obj.inflate_full_grid())


def test_basedataobj_compressed_leadtime():
    data = np.arange(24).reshape(3, 4, 2).astype(np.float32)
    data[2, 3, 1] = np.nan
    data[1, 0, 1] = np.nan
    dim = {BDO.TIME: (0, [1, 2, 3])}

    valid = np.isfinite(data[0])
    for time in data:
        valid &= np.isfinite(time)

    comp = data[:, valid]
    inflated = np.copy(data).reshape(3,8)
    inflated[:, ~valid.flatten()] = np.nan

    obj = BDO(comp, dim_coords=dim, valid_data=valid.flatten())
    assert obj._leading_time
    assert obj.is_masked
    np.testing.assert_array_equal(obj.inflate_full_grid(), inflated)


@pytest.mark.xfail
def test_basedataobj_grid_nomatch():
    data = np.arange(24).reshape(4, 3, 2)
    coords = {BDO.TIME: (0, [1, 2, 3, 4]),
              BDO.LAT: (1, [45, 50, 55, 60]),
              BDO.LON: (2, [-80, -90])}
    obj = BDO(data, dim_coords=coords)
    obj.get_coordinate_grids(BDO.LAT)


def test_basedataobj_grid():
    data = np.arange(24).reshape(4, 3, 2)
    coords = {BDO.TIME: (0, [1, 2, 3, 4]),
              BDO.LAT: (1, [45, 50, 55]),
              BDO.LON: (2, [-80, -90])}
    obj = BDO(data, dim_coords=coords)
    tmp = obj.get_coordinate_grids([BDO.LAT, BDO.LON])
    longrd, latgrd = np.meshgrid(coords[BDO.LON][1],
                                 coords[BDO.LAT][1])
    assert np.array_equal(tmp[BDO.LON], longrd)
    assert np.array_equal(tmp[BDO.LAT], latgrd)


def test_basedataobj_grid_flat():
    data = np.arange(24).reshape(4, 3, 2)
    coords = {BDO.TIME: (0, [1, 2, 3, 4]),
              BDO.LAT: (1, [45, 50, 55]),
              BDO.LON: (2, [-80, -90])}
    obj = BDO(data, dim_coords=coords, force_flat=True)
    tmp = obj.get_coordinate_grids([BDO.LAT, BDO.LON])
    longrd, latgrd = np.meshgrid(coords[BDO.LON][1],
                                 coords[BDO.LAT][1])
    assert np.array_equal(tmp[BDO.LON], longrd.flatten())
    assert np.array_equal(tmp[BDO.LAT], latgrd.flatten())


def test_basedataobj_grid_masked_compressed():
    data = np.arange(24).reshape(4, 3, 2).astype(np.float16)
    data[2, 2, 1] = np.nan
    data[1, 0, 1] = np.nan
    coords = {BDO.TIME: (0, [1, 2, 3, 4]),
              BDO.LAT: (1, [45, 50, 55]),
              BDO.LON: (2, [-80, -90])}
    obj = BDO(data, dim_coords=coords, force_flat=True)
    tmp = obj.get_coordinate_grids([BDO.LAT, BDO.LON])
    longrd, latgrd = np.meshgrid(coords[BDO.LON][1],
                                 coords[BDO.LAT][1])

    valid = np.isfinite(data[0])
    for time in data:
        valid &= np.isfinite(time)

    assert np.array_equal(tmp[BDO.LON], longrd[valid].flatten())
    assert np.array_equal(tmp[BDO.LAT], latgrd[valid].flatten())


def test_basedataobj_grid_masked_full():
    data = np.arange(24).reshape(4, 3, 2).astype(np.float16)
    data[2, 2, 1] = np.nan
    data[1, 0, 1] = np.nan
    coords = {BDO.TIME: (0, [1, 2, 3, 4]),
              BDO.LAT: (1, [45, 50, 55]),
              BDO.LON: (2, [-80, -90])}
    obj = BDO(data, dim_coords=coords, force_flat=True)
    tmp = obj.get_coordinate_grids([BDO.LAT, BDO.LON], compressed=False)
    longrd, latgrd = np.meshgrid(coords[BDO.LON][1],
                                 coords[BDO.LAT][1])
    longrd = longrd.astype(np.float16)
    latgrd = latgrd.astype(np.float16)

    valid = np.isfinite(data[0])
    for time in data:
        valid &= np.isfinite(time)

    longrd[valid] = np.nan
    latgrd[valid] = np.nan

    np.testing.assert_array_equal(tmp[BDO.LON], longrd.flatten())
    np.testing.assert_array_equal(tmp[BDO.LAT], latgrd.flatten())


### Hdf5DataObject ####
@pytest.mark.xfail
def test_hdf5dataobj_noh5file():
    data = np.arange(10)
    obj = HDO(data, 'Hello')


@pytest.mark.xfail
def test_hdf5dataobj_readonly(readonly_tb_file):
    data = np.arange(10)
    obj = HDO(data, readonly_tb_file)


@pytest.mark.xfail
def test_hdf5dataobj_closed_file(closed_tb_file):
    data = np.arange(10)
    obj = HDO(data, closed_tb_file)


def test_hdf5dataobj_setgroup_string_nopre_xist(tb_file):
    data = np.arange(10)
    obj = HDO(data, tb_file, default_grp='/lol')
    grp = tb_file.get_node('/lol')
    assert obj._default_grp == grp


def test_hdf5dataobj_setgroup_group_nopre_xist(tb_file, hdf5_obj):
    grp = tb_file.create_group(tb_file.root, 'wut')
    hdf5_obj.set_databin_grp(grp)
    assert hdf5_obj._default_grp == grp


def test_hdf5dataobj_setgroup_string_pre_xist(tb_file, hdf5_obj):
    hdf5_obj.set_databin_grp('/lol')
    assert hdf5_obj._default_grp == tb_file.get_node('/lol')


def test_hdf5dataobj_setgroup_group_pre_xist(tb_file, hdf5_obj):
    grp = tb_file.get_node('/wut')
    hdf5_obj.set_databin_grp(grp)
    assert hdf5_obj._default_grp == grp


def test_hdf5dataobj_setgroup_samename_notgrouptypenode(tb_file, hdf5_obj):
    carray = Dt.var_to_hdf5_carray(tb_file, '/wut', 'rofl', np.arange(10))
    hdf5_obj.set_databin_grp('/wut/rofl')
    grp = tb_file.get_node('/wut/rofl')
    assert carray != hdf5_obj._default_grp
    assert hdf5_obj._default_grp == grp
    assert type(hdf5_obj._default_grp) == tb.Group


@pytest.mark.xfail
def test_hdf5dataobj_setgroup_samename_notgrouptypenode(tb_file, hdf5_obj):
    carray = Dt.var_to_hdf5_carray(tb_file, '/lol', 'rofl', np.arange(10))
    hdf5_obj.set_databin_grp(carray)


@pytest.mark.parametrize("shape",
    [(24),
    ((4, 6)),
    ((4, 3, 2)),
    ((2, 2, 3, 2)),
    pytest.mark.xfail(((1, 2, 2, 3, 2))),
    ])
def test_hdf5dataobj_data_nodim(shape, tb_file):
    data = np.arange(np.product(np.array(shape))).reshape(shape)
    obj = HDO(data, tb_file)
    assert np.array_equal(obj.data, data)
    assert obj.data.shape == data.shape
    assert obj._full_shp == data.shape


def test_hdf5dataobj_nanentry_noleadtime(tb_file):
    data = np.arange(24).reshape(3, 4, 2).astype(np.float32)
    data[2, 3, 1] = np.nan
    obj = HDO(data, tb_file)
    assert obj.is_masked
    np.testing.assert_array_equal(data.flatten(), obj.orig_data)
    assert np.array_equal(data.flatten()[:-1], obj.compressed_data)
    assert np.array_equal(data.flatten()[:-1], obj.data)
    np.testing.assert_array_equal(data,
                                  obj.inflate_full_grid(reshape_orig=True),
                                  err_msg='Inflation to full grid failed.')


if __name__ == '__main__':
    try:
        f = tb.open_file('test.h5', 'w')
        test_hdf5dataobj_data_nodim((24), f)
    finally:
        f.close()
        os.system('rm -f test.h5')
    # test_basedataobj_compressed_leadtime()




