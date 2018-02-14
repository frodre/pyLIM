""" Script for creating EOFs using different data manipulations, area-weighting,
land/sea separation, etc.
"""

import tables as tb
import numpy as np

import LIMTools as lt
import Stats as st

### Script Paramteters ###
datafile = '/home/chaos2/wperkins/data/pyLIM/Trend_LIM_data.h5'
outfile = '/home/chaos2/wperkins/data/pyLIM/EOF_comparison.h5'
maskfile = '/home/chaos2/wperkins/data/20CR/land.nc'
group_name = 'std_eofs'
sep_neigs = 10
full_neigs = 20

#Load data
f = tb.open_file(datafile, 'r')
obs = f.root.data.obs.read()
lats = f.root.data.lats.read()
lats, trash = np.meshgrid(lats, range(192), indexing='ij')
f.close()

#Create the output file
outf = tb.open_file(outfile, 'a',
                    filters = tb.Filters(complevel = 2, complib = 'blosc') )

#Create group for data, if it already exists delete it and recreate
try:
    data_grp = outf.create_group( outf.root, group_name, title="Standard EOFs")
except tb.NodeError:
    outf.remove_node( outf.get_node(outf.root, group_name), recursive = True)
    data_grp = outf.create_group( outf.root, group_name, title = "Standard EOFs")
    
#Calculate obs anomalies
run_mean, _, _ = st.runMean(obs, 12, shaveYr=True)
anomaly = lt.calc_anomaly(run_mean, 12)

#Standard EOFs
eofs, eigs, pct_var = st.calcEOF(anomaly.T, full_neigs)
eofs = eofs.T
container = outf.create_carray(data_grp, 'standard',
                               atom = tb.Atom.from_dtype(eofs.dtype),
                               shape = eofs.shape,
                               title = "Standard EOFs")
container[:] = eofs
data_grp._v_attrs.std_var = pct_var

#Area-weighted EOFs
aweight_anomaly =  lt.area_wgt(anomaly, lats.flatten())
eofs, egs, pct_var = st.calcEOF(aweight_anomaly.T, full_neigs)
eofs = eofs.T
container = outf.create_carray(data_grp, 'aweight',
                               atom = tb.Atom.from_dtype(eofs.dtype),
                               shape = eofs.shape,
                               title = "Area-Weighted EOFs")
container[:] = eofs
data_grp._v_attrs.awgt_var = pct_var


#Load masks
landmask, seamask = lt.load_landsea_mask(maskfile, len(aweight_anomaly))
ocean_data = np.ma.masked_array(aweight_anomaly, landmask).filled(0)
land_data = np.ma.masked_array(aweight_anomaly, seamask).filled(0)

#Land Only EOFs
eofs, egs, pct_var = st.calcEOF(land_data.T, full_neigs)
eofs = eofs.T
container = outf.create_carray(data_grp, 'land_only',
                               atom = tb.Atom.from_dtype(eofs.dtype),
                               shape = eofs.shape,
                               title = "EOFs over land")
container[:] = eofs
data_grp._v_attrs.land_var = pct_var

#Ocean Only EOFs
eofs, egs, pct_var = st.calcEOF(ocean_data.T, full_neigs)
eofs = eofs.T
container = outf.create_carray(data_grp, 'ocean_only',
                               atom = tb.Atom.from_dtype(eofs.dtype),
                               shape = eofs.shape,
                               title = "EOFs over ocean")
container[:] = eofs
data_grp._v_attrs.ocn_var = pct_var

outf.close()