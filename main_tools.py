import numpy as np
from numpy import shape as shape_get
from numpy import arange
from numpy import exp
import scipy.optimize
import matplotlib.pyplot as plt
from IPython.display import display, Math, clear_output
import math
from copy import deepcopy
import unittest


kb = 1.38 * (10 ** (-23))
J = 0
H = 1*kb
states = [1, -1]

def user_gen_lattices(shape, states=states):
    return gen_lattices(shape, states=states)

def user_Ising(lattice, J=J, H=H):
    return Ising(lattice, J=J, H=H)

def user_avg_nrg_exact(nrg_func, system_list, T):
    return avg_nrg(nrg_func, system_list, T)

def user_Onsager_nrg(particles, T, J = J):
    return Onsager(particles, T, J)

def user_MC_sampler(shape, nrgfunc, states=states, samples=1000, T=1, merr = True):
    if merr:
        Ebar, E_std, Mbar, std_M, M_convgd = MC_sampler_handler(shape, nrgfunc, states=states, samples=samples, T=T, merr=merr)
        return np.average(Ebar), E_std, np.average(Mbar), std_M, M_convgd
    else:
        Ebar, E_std, Mbar = MC_sampler_handler(shape, nrgfunc, states=states, samples=samples, T=T, merr=merr)
        return Ebar, E_std, Mbar

####################################################################################################
def gen_lattices(shape, states=states):
    ###
    configs = gen_configs(shape, states)
    configs_lat = []
    ###

    for i in arange(len(configs)):
        configs_lat.append(cast_lattice(configs[i], shape))
    return configs_lat

def cast_lattice(array, shape):
    assert (len(array) == flatten(shape))
    ###
    dim = len(shape)
    lattice = np.empty(shape)
    steps = get_steps(shape)
    ind_list = [0]*dim
    arr_ind = 0
    cont_bool = False
    ###
    while not cont_bool:
        lattice = cast(array, lattice, ind_list, arr_ind)
        ind_list, cont_bool = next_ind_list(ind_list, shape)
        arr_ind = get_arr_ind(dim, steps, ind_list)
    lattice = cast(array, lattice, ind_list, arr_ind)
    return lattice

def get_steps(shape):
    dim = len(shape)
    output = []
    for i in arange(dim):
        output.append(flatten(shape[(i + 1):]))
    return output

def flatten(shape):
    pts = 1
    for i in arange(len(shape)):
        pts *= shape[i]
    return pts

def cast(array, lattice, ind_list, arr_ind):
    lattice[tuple(ind_list)] = array[arr_ind]
    return lattice


def get_arr_ind(dim, steps, ind_list):
    arr_ind = 0
    for i in arange(dim):
        arr_ind += steps[i] * ind_list[i]
    return arr_ind

def next_ind_list(ind_list, shape):
    ###
    dim = len(ind_list)
    new_ind_list = ind_list.copy()
    new_ind_list = check_ind_list(new_ind_list, shape, dim, 0)
    end = True
    ###
    for i in arange(dim):
        if new_ind_list[i] != (shape[i] - 1):
            end = False
            break
    return new_ind_list, end

def check_ind_list(new_ind_list, shape, dim, i):
    if i >= dim:
        return new_ind_list
    else:
        return iterate_ind_list(new_ind_list, shape, dim, i)

def iterate_ind_list(new_ind_list, shape, dim, i):
    ind_i_new = (new_ind_list[i] + 1) % shape[i]
    new_ind_list[i] = ind_i_new
    if ind_i_new == 0:
        new_ind_list = check_ind_list(new_ind_list, shape, dim, i + 1)
    return new_ind_list

def gen_configs(shape, states=states):
    ###
    configs = []
    pts = flatten(shape)
    config = [states[0]]*pts
    n_states = len(states)
    cont_bool = False
    ###
    while not cont_bool:
        configs.append(config.copy())
        config, cont_bool = next_config(config, states, pts, n_states)
    configs.append(config.copy())
    return configs

def next_config(config_old, states, pts, n_states):
    ###
    config_new = config_old.copy()
    config_new = check_config(config_new, states, pts, n_states, 0)
    end = True
    ###
    for i in arange(pts):
        if states.index(config_new[i]) != (n_states - 1):
            end = False
            break
    return config_new, end


def check_config(config_new, states, length, n_states, i):
    if i >= length:
        return config_new
    else:
        return iterate_config(config_new, states, length, n_states, i)


def iterate_config(config_new, states, length, n_states, i):
    ###
    stat_i = states.index(config_new[i])
    stat_i_new = (stat_i + 1) % n_states
    config_new[i] = states[stat_i_new]
    ###
    if stat_i_new == 0:
        config_new = check_config(config_new, states, length, n_states, i + 1)
    return config_new


def Ising(lattice, J=J, H=H):
    ###
    shape = shape_get(lattice)
    dim = len(shape)
    start_ind = [0]*dim
    ###
    magE = -H * get_magn(lattice, shape, start_ind)
    intrxE = -J * get_intrx(lattice, shape, dim, start_ind)
    return magE + intrxE


def get_magn(lattice, shape, start_ind):
    ###
    magn = 0
    cont_bool = False
    cur_ind = start_ind.copy()
    ###
    while not cont_bool:
        magn += lattice[tuple(cur_ind)]
        cur_ind, cont_bool = next_ind_list(cur_ind, shape)
    magn += lattice[tuple(cur_ind)]
    return magn


def get_intrx(lattice, shape, dim, start_ind):
    ###
    intrx = 0
    cont_bool = False
    budrange = arange(dim*2)
    dimrange = arange(dim)
    cur_ind = start_ind.copy()
    ###
    while not cont_bool:
        intrx += comp_intrx(lattice, cur_ind, shape, budrange, dimrange)
        cur_ind, cont_bool = next_ind_list(cur_ind, shape)
    intrx += comp_intrx(lattice, cur_ind, shape, budrange, dimrange)
    return intrx


def comp_intrx(lattice, cur_ind, shape, budrange, dimrange):
    ###
    # budrange = arange(dim*2)
    # dimrange = arange(dim)
    cur_nrgJ = 0
    cur_stat = lattice[tuple(cur_ind)]
    buddy_inds = get_buddy_inds(cur_ind, shape, dimrange)
    buddy_stats = get_buddy_stats(lattice, buddy_inds, budrange)
    buddy_bools = get_buddy_bools(cur_ind, buddy_inds, budrange, dimrange)
    ###
    for i in budrange:
        if buddy_bools[i]:
            cur_nrgJ += buddy_stats[i] * cur_stat
    return cur_nrgJ

def get_buddy_inds(cur_ind, shape, dimrange):
    buddy_inds = []
    for i in dimrange:
        cap = shape[i]
        cur_val = cur_ind[i]
        buddy_inds.append(get_plus_neighbor_ind(i, cap, cur_val, cur_ind))
        buddy_inds.append(get_min_neighbor_ind(i, cap, cur_val, cur_ind))
    return buddy_inds

def get_plus_neighbor_ind(i, cap, cur_val, cur_ind):
    p_val = (cur_val + 1) % cap
    buddy_ind_p = cur_ind.copy()
    buddy_ind_p[i] = p_val
    return buddy_ind_p

def get_min_neighbor_ind(i, cap, cur_val, cur_ind):
    m_val = (cur_val - 1) % cap
    buddy_ind_m = cur_ind.copy()
    buddy_ind_m[i] = m_val
    return buddy_ind_m

def get_buddy_stats(lattice, buddy_inds, budrange):
    buddy_stats = []
    for i in budrange:
        buddy_stats.append(lattice[tuple(buddy_inds[i])])
    return buddy_stats

def get_buddy_bools(cur_ind, buddy_inds, budrange, dimrange):
    buddies = len(buddy_inds)
    buddy_bools = [True]*buddies
    for i in budrange:
        buddy_bools[i] = check_bud_ind(cur_ind, buddy_inds[i], dimrange)
    return buddy_bools

def check_bud_ind(cur_ind, bud_ind, dimrange):
    for j in dimrange:
        if bud_ind[j] < cur_ind[j]:
            return False
    return True

def nrgs(nrg_func, system_list):
    output = []
    for i in np.arange(len(system_list)):
        output.append(nrg_func(system_list[i]))
    return output

def avg_nrg(nrg_func, system_list, T):
    ###
    β = 1 / (kb * T)
    Q, nrgs = partition_func(nrg_func, system_list, β)
    pOb = 0
    ###

    for i in arange(len(system_list)):
        pOb += boltz_prop(nrgs[i],β) * nrgs[i]
    return pOb / Q

def simple_avg_nrg(nrg_func, shape, T, states):
    system_list = gen_lattices(shape, states)
    return avg_nrg(nrg_func, system_list, T)

def quick_avg_nrg(nrg_list, T):
    Q, boltz_props = quick_partition_func(nrg_list, T)
    pOb = 0
    for i in arange(len(nrg_list)):
        pOb += boltz_props[i] * nrg_list[i]
    return pOb / Q

def expectation_val(observable_func, nrg_func, system_list, T):
    ###
    β = 1 / (kb * T)
    Q, nrgs = partition_func(nrg_func, system_list, β)
    pOb = 0
    ###
    for i in arange(len(system_list)):
        pOb += boltz_prop(nrgs[i],β) * observable_func(system_list[i])
    return pOb / Q

def partition_func(nrg_func, system_list, β):
    ###
    system_size = len(system_list)
    nrgs_list = nrgs(nrg_func, system_list)
    output = 0
    ###
    for i in arange(system_size):
        output += boltz_prop(nrgs_list[i], β)
    return output, nrgs_list

def quick_partition_func(nrg_list, T):
    system_size = len(nrg_list)
    boltz_props = [0]*system_size
    output = 0
    β = 1 / (kb * T)
    for i in arange(system_size):
        boltz_props[i] = boltz_prop(nrg_list[i], β)
        output += boltz_props[i]
    return output, boltz_props

def boltz_prop(nrg, β):
    return exp(-β * nrg)


def Onsager(particles, T, J = J):
    β = 1/(kb * T)
    return particles * U(J, β)

def trapezoid(func, x_i, x_f, traps = 10000):
    Dx = x_f - x_i
    dx = Dx/traps
    xs = np.arange(x_i, x_f, dx)
    fs = np.empty_like(xs)
    Fs = []
    intg = 0
    for i in np.arange(len(xs)):
        fs[i] = func(xs[i])
    for i in np.arange(len(fs)-1):
        Fs.append((fs[i] + fs[i+1])/2)
    for i in np.arange(len(Fs)):
        intg += Fs[i]*dx
    return intg

def U(J, β, integrator = trapezoid):
    A = -J*coth(2*β*J)
    B = (2/np.pi)*(2*(np.tanh(2*β*J))**2-1)
    C = U_intg(J, β, integrator)
    U_val = A*(1+B*C)
    return U_val

def U_intg(J, β, integrator):
    kv = k(J, β)
    u_func = lambda θ: 1/(np.sqrt(1 - 4*kv*((1+kv)**(-2))*((np.sin(θ))**2)))
    return integrator(u_func, 0, np.pi/2)

def k(J, β):
    k_val = 1/((np.sinh(2*β*J))**2)
    return k_val

def coth(x):
    return 1/(np.tanh(x))

def MC_sampler_handler(shape, nrgfunc, states=states, samples=1000, T=1, merr = True):
    have_sample = False
    while not have_sample:
        Ebar, Mbar, lattices = MC_sampler(shape, nrgfunc, states=states, samples=samples, T=T)
        std_E, conv_E = get_std(Ebar)
        have_sample = conv_E
    if merr:
        std_M, conv_M = get_M_std(Mbar)
        return Ebar, std_E, Mbar, std_M, conv_M
    else:
        return Ebar, std_E, Mbar

def MC_sampler(shape, nrgfunc, states=states, samples=1000, T=1):
    β = 1 / (kb * T)
    start_ind = [0] * len(shape)
    Ebar = []
    Mbar = []
    lattices_used = []
    arr_len = flatten(shape)
    n_states = len(states)
    cur_array = init_random(arr_len, n_states, states)
    cur_lattice = cast_lattice(cur_array, shape)
    cur_nrg = nrgfunc(cur_lattice)
    cur_mag = get_magn(cur_lattice, shape, start_ind)
    for i in arange(samples):
        u = np.random.random()
        new_array = change_random(cur_array, arr_len, n_states, states)
        new_lattice = cast_lattice(new_array, shape)
        new_nrg = nrgfunc(new_lattice)
        d_nrg = new_nrg - cur_nrg
        change_condition = boltz_prop(d_nrg, β)
        # if this criteria, redefine all cur_x from current new_x
        if u < change_condition:
            cur_array = new_array
            cur_lattice = new_lattice
            cur_nrg = new_nrg
            cur_mag = get_magn(cur_lattice, shape, start_ind)
        Ebar.append(cur_nrg)
        Mbar.append(cur_mag)
        lattices_used.append(cur_lattice.copy())
    return Ebar, Mbar, lattices_used

def init_random(arr_len, n_states, states):
    init_array = [0] * arr_len
    for i in arange(arr_len):
        init_array[i] = states[np.random.randint(0, n_states)]
    return init_array

def dif_rand_int(cap, cur):
    while True:
        new_int = np.random.randint(0, cap)
        if cur != new_int:
            return new_int

def change_random(first_array, arr_len, n_states, states):
    new_array = first_array.copy()
    # change_ind = np.random.randint(0, arr_len - 1)
    change_ind = np.random.randint(0, arr_len)
    cur_ind_stat_id = states.index(first_array[change_ind])
    new_ind_stat_id = dif_rand_int(n_states, cur_ind_stat_id)
    new_array[change_ind] = states[new_ind_stat_id]
    return new_array

# bin_dev: array, int -> float
# --- Returns standard deviation of array after binning with given bin length
def bin_dev(array, length):
    N = np.shape(array)[0]
    n_bin = N//length
    bin_avgs = np.zeros((n_bin,))
    for i in arange(n_bin):
        for j in range(length):
            bin_avgs[i] += array[i*length + j]
        bin_avgs[i] /= length
    return np.std(bin_avgs)

# extrap_bin_dev: array, int -> array, array
# --- Returns two arrays, computing bin_dev with bin length of increasing powers of 2
# ------ stddevs: array of Std Dev's computed with bin_dev
# ------ binSize: array of bin lengths used in computing stddevs
def extrap_bin_dev(array, samples):
    N = len(array)
    stddevs = np.zeros((samples,))
    binSize = 1.*stddevs
    for i in arange(samples):
        length = 2**i
        nbins = N//length
        stddevs[i] = bin_dev(array, length)/(np.sqrt(nbins))
        binSize[i] = length
    return stddevs, binSize

def convExp(x, m, t, b):
    return -m * np.exp(-t * x) + b

def get_std(Ebar, cap=8, conv_cut = 0.98):
    stddevs, bins = extrap_bin_dev(Ebar, cap)
    p0 = (1, 1, 5E-24)
    params, cv = scipy.optimize.curve_fit(convExp, bins, stddevs, p0)
    m, t, b = params
    sampleRate = 20_000 # Hz
    tauSec = (1 / t) / sampleRate
    squaredDiffs = np.square(stddevs - convExp(bins, m, t, b))
    squaredDiffsFromMean = np.square(stddevs - np.mean(stddevs))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    converged = rSquared > conv_cut
    return b, converged

def get_M_std(Mbar, cap=8, conv_cut = 0.98):
    stddevs, bins = extrap_bin_dev(Mbar, cap)
    p0 = (1, 1, 0.1) # guess m, t, b
    params, cv = scipy.optimize.curve_fit(convExp, bins, stddevs, p0)
    m, t, b = params
    sampleRate = 20_000 # Hz
    tauSec = (1 / t) / sampleRate
    squaredDiffs = np.square(stddevs - convExp(bins, m, t, b))
    squaredDiffsFromMean = np.square(stddevs - np.mean(stddevs))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    converged = rSquared > conv_cut
    return b, converged


def sig_round(a_number, significant_digits):
    return round(a_number, significant_digits - int(math.floor(math.log10(abs(a_number)))) - 1)

