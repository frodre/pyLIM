"""
A simple collection of tests for various operations performed in pyLIM

"""

import numpy


def reshapeTest(data):
    """
    Tests reshaping of data to confirm that times stay in order.

    Parameters
    ----------
    data: ndarray
        Data to be reshaped and tested for consistency

    Returns
    -------
    bool
        Did it pass the test?    
    """
    shp = data.shape
    num_years = shp[0]/12.
    tmp_data = data.reshape(num_years, 12, shp[1], shp[2])

    #Just to make sure it finds falses uncommment lines below
    #tmp_copy = numpy.array(tmp_data, copy=True)
    #tmp_data = tmp_copy
    #tmp_data[0,0,:,:] = tmp_data[0,1,:,:]

    for  i,tmp_map in enumerate(data):
	reshaped_map = tmp_data[i/12, i%12, :, :]
        if not (tmp_map==reshaped_map).all():
            return False
    return True 
