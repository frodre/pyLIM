import numpy as np
from itertools import product
from math import isclose
from scipy.optimize import fsolve
from multiprocessing import Pool

import pylim.LIM as LIM
import pylim.Stats as ST

import data_utils as dutils

class LIMState(object):
    """
    Create a state usable for LIM calibration

    Parameters
    ----------
    dobjs: dict of DataTools.BaseDataObject
        List of data objects that you would like to combine.  Must have the
        eof projection already completed.

    Attributes
    ----------
    var_span: dict of tuples(int)
        The index location for each variable in the state.
    data: ndarray
        A combined state numpy array
    """
    def __init__(self, dobjs, dobj_key='eof_proj'):
        self.var_span = {}
        self.data = []
        self.eofs = None
        self.svals = None
        self.is_eof_proj = False
        self.dobjs = dobjs
        self.var_order = []
        
        start =  0
        for key, dobj in dobjs.items():
            self.var_order.append(key)
            dobj_data = getattr(dobj, dobj_key)
            self.data.append(dobj_data[:])
            end = start + dobj_data.shape[1]
            self.var_span[key] = (start, end)
            start = end
            
        self.data = np.concatenate(self.data, axis=1)
        self.untruncated = self.data
        self.eof_truncated = None
        
    def get_var_from_state(self, key, data=None):
        """
        Retrieve a single variable from the state array. Defaults to
        retrieving from self, but allowed to provide outside data of
        the same shape.
        """
            
        if self.is_eof_proj and data is None:
            raise ValueError('Cannot retrieve variables unless state is in original basis.'
                             ' Please use proj_state_into_phys().')
        
        if data is None:
            data = self.data
            
        start, end = self.var_span[key]
        return data[..., start:end]
    
    def get_var_eofs(self, key):
        "Retrieve the EOFs of a single variable"
        
        if self.eofs is None:
            raise ValueError('No EOFs have been calculated.')
            
        start, end = self.var_span[key]
        return self.eofs[start:end, :]
    
    def calc_state_eofs(self, num_eofs):
        
        var_stats = {}
        state_eofs, state_svals = ST.calc_eofs(self.data, num_eofs, 
                                               var_stats_dict=var_stats)
        self.eofs = state_eofs
        self.svals = state_svals
        
        return var_stats
    
    def proj_state_onto_eofs(self):
        
        if self.eofs is None:
            raise ValueError('State EOFs have not been calculated!')
            
        if self.is_eof_proj:
            print('Data is already in EOF space.')
            return None
        
        if self.eof_truncated is None:        
            self.data = self.data @ self.eofs
        else:
            self.data = self.eof_truncated
        self.is_eof_proj = True
    
    # TODO: change name to 'return_state_to_phys'
    def proj_state_into_phys(self):
        
        if not self.is_eof_proj:
            print('State is already in original basis.')
            return None
        
        self.data = self.untruncated
        self.is_eof_proj = False
        
    def proj_data_into_orig_basis(self, data, unstandardize=False):
        orig_basis = data @ self.eofs.T
        
        if unstandardize:
            orig_basis = self.unstandardize_data(orig_basis)
            
        return orig_basis
    
    def proj_data_into_eof_basis(self, data):
        return data @ self.eofs
    
    def unstandardize_data(self, state_data):
        for key, dobj in self.dobjs.items():
            std_factor = dobj._std_scaling
            data = self.get_var_from_state(key, data=state_data)
            unstd_data = data / std_factor
            data[:] = unstd_data
            
        return state_data

# Mode Calculation
def check_constraints(c_real, c_imag, orig_vector):
    new_vector = complex(c_real, c_imag) * orig_vector
    a_new = new_vector.real
    b_new = new_vector.imag

    aa = np.dot(a_new, a_new)
    ab = np.dot(a_new, b_new)
    bb = np.dot(b_new, b_new)

    a_norm = np.linalg.norm(a_new)
    b_norm = np.linalg.norm(b_new)

    aa_const = isclose(aa, 1.0, rel_tol=1e-6, abs_tol=1e-6)
    ab_const = isclose(ab, 0.0, rel_tol=1e-6, abs_tol=1e-6)
    b_great = bb >= 1.0

    passed = aa_const and ab_const and b_great

    print('cr: {:2.2f}, ci: {:2.2f}'.format(c_real, c_imag))
    print('\tNew a.a: {:2.2f}; Pass: {}'.format(aa, aa_const))
    print('\tNew a.b: {:2.2f}; Pass: {}'.format(ab, ab_const))
    print('\tNew b.b >= 1: {}'.format(bb >= 1))
    print('\tNew |a| >= |b|: {}'.format(a_norm >= b_norm))
    print('\tPass: {}'.format(passed))

    return passed, new_vector


def get_ortho_complex_basis(eig_vector):
    a = eig_vector.real
    b = eig_vector.imag

    aa = np.dot(a, a)
    ab = np.dot(a, b)
    bb = np.dot(b, b)

    def constraint_func(c):
        cr, ci = c

        eq1 = cr ** 2 * ab + cr * ci * (aa - bb) - ci ** 2 * ab
        eq2 = cr ** 2 * aa - 2 * ci * cr * ab + ci ** 2 * bb - 1
        return [eq1, eq2]

    ic_range = (10, 0, -10)
    func_ics = product(ic_range, ic_range)

    for ic in func_ics:
        c_real, c_imag = fsolve(constraint_func, ic)
        passes, new_basis = check_constraints(c_real, c_imag, eig_vector)
        if passes:
            return new_basis
    else:
        raise ValueError('No solution found for given basis.')


def get_eigen_modes(matrix):
    e_vals, e_vecs = np.linalg.eig(matrix)
    sort_idx = e_vals.real.argsort()
    return e_vals[sort_idx][::-1], e_vecs[:, sort_idx][:, ::-1]


def get_enso_factor(eofs, latgrid, longrid):
    enso_34_mask = ((longrid >= 240) & (longrid <= 290) &
                    (latgrid >= -5) & (latgrid <= 5))
    num_pts_enso = enso_34_mask.sum()
    enso_avg = np.zeros_like(latgrid)
    enso_avg[enso_34_mask] = 1 / num_pts_enso
    enso_avg_factor = eofs.T @ enso_avg

    return enso_avg_factor, enso_avg


def get_pdo_factor(eofs, data, latgrid, longrid):
    npac_mask = ((latgrid >= 20) & (latgrid <= 70) &
                 (longrid >= 110) & (longrid <= 250))
    # Have to perform a read here because fancy indexing doesn't work for CArr?
    data = data[:][:, npac_mask]
    npac_eofs, npac_svals = ST.calc_eofs(data, 1)
    npac_eofs_full = np.zeros_like(latgrid)
    npac_eofs_full[npac_mask] = npac_eofs[:, 0]
    pdo_factor = eofs.T @ npac_eofs_full

    return pdo_factor, npac_eofs_full


def get_glob_mean_factor(eofs, cell_area):

    wgt_avg_factor = cell_area / cell_area.sum()
    eof_avg_factor = eofs.T @ wgt_avg_factor

    return eof_avg_factor, wgt_avg_factor


def pool_func(args):
    # TODO: Samples are correlated in time, need to do block resample instead
    i, (fcast, ref, n_samples, metric) = args

    np.random.seed(i)
    sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
    test_fcast = fcast[sample_idx]
    test_ref = ref[sample_idx]
    if metric == 'r':
        out = np.corrcoef(test_fcast, test_ref)[0, 1]
    else:
        out = ST.calc_ce(test_fcast, test_ref)

    return out


def conf_bound95(fcast, reference, metric='r', use_sample_size=None):

    n_samples = len(fcast)
    n_iters = 10000
    threshold = 1e-5

    if metric == 'r':
        mean_target = np.corrcoef(fcast, reference)[0, 1]
    elif metric == 'ce':
        mean_target = ST.calc_ce(fcast, reference)
    else:
        raise KeyError('Unknown metric specified: {}'.format(metric))

    args = [(fcast, reference, n_samples, metric), ]
    seeds = np.random.choice(n_iters*10, size=n_iters, replace=False)
    iter_args = product(seeds, args)
    with Pool(processes=10) as calc_pool:
        result = calc_pool.map(pool_func, iter_args)

    result = np.array(result)
    diff = abs(result.mean() - mean_target)
    print('Mean Target {} diff: {:1.3e}'.format(metric, diff))

    # Create bounds for plt.errorbars
    lower_bnd = np.percentile(result, 2.5)
    upper_bnd = np.percentile(result, 97.5)
    res_mean = result.mean()
    conf_bounds = (upper_bnd, lower_bnd)

    return res_mean, conf_bounds


def fcast_1yr_state(lim, orig_state, new_state, nelem_in_yr):

    for var_orig, var_new in zip(orig_state.var_order, new_state.var_order):
        if var_orig != var_new:
            raise ValueError('State fields are not in the same order.'
                             '\n{}\n{}'.format(orig_state.var_order,
                                               new_state.var_order))

    reproj_new_state = orig_state.proj_data_into_eof_basis(new_state.data)

    init_t0 = reproj_new_state[:-nelem_in_yr]

    fcast_1yr = lim.forecast(init_t0, [1])[0]

    fcast_1yr_full = orig_state.proj_data_into_orig_basis(fcast_1yr, unstandardize=True)
    init_t0_full = new_state.data[:-nelem_in_yr]

    return fcast_1yr_full, init_t0_full


def ens_1yr_fcast(nens, lim, t0, timesteps=1440):

    seeds = np.random.choice(nens*100, size=nens, replace=False)
    args = product(seeds, ((t0, lim, timesteps), ))
    with Pool(processes=2) as fcast_pool:
        ens_output = fcast_pool.map(_noise_int_func, args)

    ens_output = np.stack(ens_output, axis=0)

    return ens_output


def ens_long_integration(nens, length, lim, t0, timesteps=1440):

    ens_t0 = np.tile(t0, (nens, 1))
    ens_out = np.empty((1441, nens, t0.shape[1]))
    long_int_out = np.zeros((length, nens, t0.shape[1]))
    long_int_out_avg = np.zeros_like(long_int_out)

    for i in range(length):

        ens_t0 = lim.noise_integration(ens_t0, 1,
                                       timesteps=timesteps,
                                       out_arr=ens_out)
        end_val = ens_out[-1]
        avg = ens_out.mean(axis=0)

        long_int_out[i] = end_val
        long_int_out_avg[i] = avg

    return long_int_out, long_int_out_avg


def _noise_int_func(args):
    i, (t0, lim, timesteps) = args

    return lim.noise_integration(t0, 1, timesteps=timesteps, seed=i)










