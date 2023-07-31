import re
import numbers
import functools
from time import time
from termcolor import colored
from pyCEvNS.helper import _gaussian, _poisson
from pyCEvNS.constants import *
from scipy.stats import chi2
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.special import gamma
from scipy.interpolate import griddata, interp1d
import matplotlib.pyplot as plt


####################################################
#################### Model #########################
####################################################
def relative_speed(v1, v2, theta):
    """
    v1: velocity of particle 1
    v2: velocity of particle 2
    theta: angle between v1 and v2 [rad]. from 0 to pi
    return: relative velocity
    *** velocities are in any unit
    """
    return np.sqrt(v1**2 + v2**2 - 2*v1*v2*np.cos(theta))

def ermax(echi, mchi, dE, mA):
    """
    DM-nucleus scattering
    echi: DM energy
    mchi: DM mass
    dE: excitation energy
    mA: nuclear mass
    return: maximum recoil energy
    """
    if echi <= mchi + dE:
        return 0
    pchi = momentum(echi, mchi)
    pchip = momentum(echi - dE, mchi)
    return (pchi + pchip)**2 / (2 * mA)

def ermin(echi, mchi, dE, mA):
    """
    DM-nucleus scattering
    echi: DM energy
    mchi: DM mass
    dE: excitation energy
    mA: nuclear mass
    return: minimum recoil energy
    """
    if echi <= mchi + dE:
        return 0
    pchi = momentum(echi, mchi)
    pchip = momentum(echi - dE, mchi)
    return (pchi - pchip)**2 / (2 * mA)

def momentum(e, m):
    return np.sqrt(e**2 - m**2)

def kinetic_energy2speed(T, m):
    """
    T: kinetic energy
    m: mass
    return: speed in unit of c
    *** T and m are in the same unit
    """
    return np.sqrt(1 - m**2 / (T+m)**2)

def FHelm(A, er):
    """
    Helm form factor
    er: recoil energy [MeV]
    A: mass number
    """
    if er <= 0:
        return 0
    ker = er*1e3  # convet to keV
    q = 6.92e-3 * np.sqrt(A) * np.sqrt(ker)
    r = (1.23*A**(1/3) - 0.6)**2 + 7/3*np.pi**2 * 0.52**2 - 5 * 0.9**2
    r = r**0.5
    s = 0.9

    return 3 * (np.sin(q*r) - q*r*np.cos(q*r)) / (q*r)**3 * np.exp(-q**2 * s**2 / 2)

def conv_strength(dE, nucl_ex, conv_width=1):
    """
    Use gaussian to convlute the delta function (each normalized to its strength)
    dE: excitation energy = deexcitation photon energy
    nucl_ex: bigstick nuclear strength
    conv_width: width of the gaussian function [MeV]. must be at least greater than the deex energy bin width
    return: the convolution
    """
    bigstick_energy = nucl_ex[:, 0]
    bigstick_strength = nucl_ex[:, 1]
    def totals(x):
        gau = 0
        for i in range(len(bigstick_energy)):
            gau += gaussian(x, bigstick_energy[i], conv_width) * bigstick_strength[i]
        return gau

    return totals(dE)

def x_convert(xx_in, old, new, mass_ratio=3):
    """
    Convert between m_chi and m_A'
    xx_in: the input values
    old: 'chi', 'A'
    new: 'chi', 'A'
    mass_ratio: m_A' / m_chi
    """
    if old == 'chi':
        m_chi = xx_in
    elif old == 'A':
        m_chi = xx_in / mass_ratio
    else:
        raise ValueError('Wrong old value')

    if new == 'chi':
        xx_out = m_chi
    elif new == 'A':
        xx_out = m_chi * mass_ratio
    else:
        raise ValueError('Wrong new value')

    return xx_out

def y_convert(yy_in, old, new, mass_ratio=3):
    """
    Convert between Y, eps, alpha_B and gB
    yy_in: the input values
    old: 'eps', 'Y', 'alpha_B' or 'gB'
    new: 'eps', 'Y', 'alpha_B' or 'gB'
    mass_ratio: m_A' / m_chi
    """
    alpha_D = .5

    if old == 'eps':
        eps = yy_in
    elif old == 'Y':
        eps = np.sqrt(yy_in / alpha_D * mass_ratio**4)
    elif old == 'gB':
        eps = yy_in / e_charge
    elif old == 'alpha_B':
        eps = np.sqrt(4*np.pi * yy_in / e_charge**2)
    else:
        raise ValueError('Wrong old value')

    if new == 'eps':
        yy_out = eps
    elif new == 'Y':
        yy_out = eps**2 * alpha_D / mass_ratio**4
    elif new == 'gB':
        yy_out = eps * e_charge
    elif new == 'alpha_B':
        yy_out = eps**2 * e_charge**2 / (4*np.pi)
    else:
        raise ValueError('Wrong new value')

    return yy_out


####################################################
################# Experiment #######################
####################################################
def keV2PE(keV):
    """
    Convert keV to PE, only for COHERENT CsI
    """
    a = 0.0554628
    b = 4.30681
    c = -111.707
    d = 840.384
    pe = 13.35*(a* keV/10**3 + b*keV**2/10**6 + c*keV**3/10**9 + d*keV**4/10**12)* 10**3
    return pe

def PE2keV(pe_const):
    """
    Convert PE to keV, only for COHERENT CsI
    """
    if pe_const == 0:
        return 0
    a = 0.0554628
    b = 4.30681
    c = -111.707
    d = 840.384

    # 4 order polynomial (descending)
    pe_arr = 13.35 * np.array([d/10**12, c/10**9, b/10**6, a/10**3, 0]) * 10**3
    pe_arr[-1] -= pe_const

    roots = np.roots(pe_arr)
    for root in roots:
        root_abs = abs(root)
        if root_abs == root and 0 < root_abs < 46:
            # 60PE = 45.1481 keV, thus set a upper bound of 46 keV
            return root_abs

def smearingE(x, pe):
    """
    COHERENT CsI smearing function
    """
    a = 1 / pe
    b = 0.716 * pe
    term1 = (a*(1+b))**(1+b) / gamma(1+b)
    term2 = x**b * np.exp(-a*(1+b)*x)
    return term1 * term2

def effT(t):
    """
    COHERENT CsI timing efficiency
    t: time [mus]
    """
    a = 0.52
    b = 0.0494
    if t < a:
        return 1
    res = np.exp(-b*(t-a))
    return res

def effE(pe):
    """
    COHERENT CsI energy efficiency
    """
    a = 1.32045
    b = 0.285979
    c = 10.8646
    d = -0.333322
    res = a/(1+np.exp(-b * (pe - c))) + d
    if res < 0:
        return 0
    return res

def effE_MeV(er):
    """
    COHERENT CsI energy efficiency
    er: recoil energy [MeV]
    """
    pe_per_mev = 0.0878 * 13.348 * 1e3
    pe = pe_per_mev * er
    return effE(pe)

def efficiency(pe, t):
    """
    COHERENT CsI energy & timing efficiency
    """
    # return 1
    res = effE(pe) * effT(t)
    if res < 0:
        return 0
    return res

def pion_info2flux(pi0Info):
    """
    Parser the Monte Carlo data of pion flux (for COHERENT and CCM)
    pi0Info: the path of the data
    """
    pion_energy = pi0Info[:,4] - massofpi0
    pion_azimuth = np.arccos(pi0Info[:,3] / np.sqrt(pi0Info[:,1]**2 + pi0Info[:,2]**2 + pi0Info[:,3]**2))
    pion_cos = np.cos(np.pi/180 * pi0Info[:,0])
    pion_flux = np.array([pion_azimuth, pion_cos, pion_energy]).transpose()
    return pion_flux

def pion_info2flux_shielded():
    """
    Parser the Monte Carlo data of pion flux (for PIP2BD)
    """
    # px, py, pz, E, x, y, z, t
    # [GeV], [meter], [s]
    pi0Info = np.genfromtxt("Experiments/PIP2BD/pizero_combined2GeV_shieldedCTarget.txt")

    # down sample to 1e5
    random_idx = np.random.randint(len(pi0Info), size=int(1e5))
    pi0Info = pi0Info[random_idx,:]

    pi0Info[:, 0] *= 1e3 # convert to MeV
    pi0Info[:, 1] *= 1e3
    pi0Info[:, 2] *= 1e3
    pi0Info[:, 3] *= 1e3

    px = pi0Info[:, 0]
    py = pi0Info[:, 1]
    pz = pi0Info[:, 2]
    p = np.sqrt(px**2 + py**2 + pz**2)
    cos_angle = px / p

    pion_energy = pi0Info[:,3] - massofpi0
    pion_cos = cos_angle
    pion_azimuth = np.arccos(pi0Info[:,4] / np.sqrt(pi0Info[:,4]**2 + pi0Info[:,5]**2 + pi0Info[:,6]**2))

    pion_flux = np.array([pion_azimuth, pion_cos, pion_energy]).transpose()
    return pion_flux

def pionPlus_info2flux_BNB():
    """
    Parser the Monte Carlo data of pion plus flux (for BNB)
    """
    # momentum [eV], beam axis (rad), weight per POT
    pimInfo = np.genfromtxt("Experiments/BNB/bnb_posthorn_piplus_pMeV_thetaRad_wgts1POT.txt")

    pimInfo[:, 0] /= 1e6 # convert to MeV
    pim_vel = pimInfo[:, 0] / np.sqrt(pimInfo[:, 0]**2 + massofpi0**2)
    pim_p, pim_edges = np.histogram(pim_vel, bins=100, weights=pimInfo[:, 2])

    plt.figure(figsize=(8, 6))
    plot_hist(pim_p / np.sum(pim_p), pim_edges)
    plt.ylabel(r'Frequency', fontsize=18)
    plt.xlabel('Velocity [c]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Normalized pi plus velocity distribution', loc="right", fontsize=20)
    plt.tight_layout()
    plt.show()

def pion_info2flux_BNB():
    """
    Parser the Monte Carlo data of pion flux (for BNB)
    """
    # px, py, pz, E [GeV]
    pi0Info = np.genfromtxt("Experiments/BNB/bnb_pi_zero.txt")

    # down sample to 1e5
    random_idx = np.random.randint(len(pi0Info), size=int(1e5))
    pi0Info = pi0Info[random_idx,:]

    pi0Info[:, 0] *= 1e3 # convert to MeV
    pi0Info[:, 1] *= 1e3
    pi0Info[:, 2] *= 1e3
    pi0Info[:, 3] *= 1e3

    px = pi0Info[:, 0]
    py = pi0Info[:, 1]
    pz = pi0Info[:, 2]
    p = np.sqrt(px**2 + py**2 + pz**2)
    cos_angle = pz / p   # TODO: check the beam direciton

    pion_energy = pi0Info[:,3] - massofpi0
    pion_cos = cos_angle
    pion_azimuth = np.arccos(pi0Info[:,0] / np.sqrt(pi0Info[:,0]**2 + pi0Info[:,1]**2 + pi0Info[:,2]**2))

    pion_flux = np.array([pion_azimuth, pion_cos, pion_energy]).transpose()
    return pion_flux



####################################################
#################### Stats #########################
####################################################
def deltaChi2(sig, bkg, syst=0):
    return np.sum((sig)**2 / (bkg+syst*bkg**2))

def cl2sig(cl):
    """
    Convert confidence level to gaussian sigma (sigificance)
    0.68->1, 0.95->2, 0.997->3
    """
    if cl > 1 or cl < 0:
        raise ValueError('CL must be between 0 and 1')
    return abs(norm.ppf((1-cl)/2))

def gaussian(x, mu, sigma):
    return 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-np.power((x - mu)/sigma, 2.)/2)

def cauchy(x, x0, gamma):
    return 1/(np.pi*gamma*(1+((x-x0)/gamma)**2))

def maxwell(v, v0=220):
    """
    Maxwell-Boltzmann distribution for WIMP
    v: velocity [km/s]
    v0: rms velocity [km/s]
    """
    return 4*np.pi * v**2 / (np.pi**(3/2) * v0**3) * np.exp(-v**2 / v0**2)

def excessANDsig(n_obs, n_bg, n_nu):
    """
    Get the excess and significance
    n_obs: summed observed events
    n_bg: summed background events
    n_nu: summed neutrino events
    """
    excess = n_obs - n_nu - n_bg
    sig = excess / np.sqrt(n_nu + n_bg)
    return excess, sig

def loglike_csi(n_signal, n_obs, n_bg, sigma):
    """
    Mixed likelihood function
    n_signal: signal event array
    n_obs: observed event array
    n_bg: background event array
    sigma: uncertainty of the background (a number)
    """
    likelihood = np.ones(len(n_obs))
    for i in range(len(n_obs)):
        n_bg_list = np.arange(max(0, int(n_bg[i] - 2 * np.sqrt(n_bg[i]))),
                                max(10, int(n_bg[i] + 2 * np.sqrt(n_bg[i]))))
        for nbgi in n_bg_list:
            likelihood[i] += quad(lambda a: _poisson(n_obs[i], (1 + a) * n_signal[i] + nbgi) *
                                    _gaussian(a, 0, sigma), -3 * sigma, 3 * sigma)[0] * _poisson(n_bg[i], nbgi)
    return np.sum(np.log(likelihood))

def loglike_chi2(n_signal, n_obs, n_bg, err):
    """
    Chi2 likelihood function
    n_signal: signal event array
    n_obs: observed event array
    n_bg: background event array
    err: uncertainty array
    """
    err = err + 1
    likelihood = (n_obs - (n_signal + n_bg)) **2 / err**2
    return np.sum(likelihood)

def log_int(func, a, b, **kwargs):
    """
    Integrate a function in log space, designed for extreme small/large functions
    func: the function to integrate
    a, b: integration range (original, not in log space)
    return: integral
    """
    def ff(u):
        return np.exp(u)*func(np.exp(u))
    return quad(ff, np.log(a), np.log(b), **kwargs)[0]

# statistics test
def test_err(signals, bkgs, x, y, confidence_limit=0.9, test='chi2', crit_events=2.3, verbose=True, syst=0):
    """
    signals: the signal array
    bkgs: the background array
    x, y: the parameter of the test
    crit_events: the critical events for the 'nobkg' test
    syst: systematic uncertainty
    err = (statstic - limit) / limit
    """
    signal = np.sum(signals)
    bkg_sum = np.sum(bkgs)
    if test == 't':
        sig_limit = cl2sig(confidence_limit)
        test_type = 'sig'
        sig = signal / np.sqrt(bkg_sum)  # the fixed t-test
        err = (sig - sig_limit) / sig_limit
        statstic = sig

    elif test == 'chi2':
        deltaChi2_limit = chi2.ppf(confidence_limit, len(bkgs)) - len(bkgs)
        test_type = 'chi2'
        chi2_value = deltaChi2(signals, bkgs, syst)
        err = (chi2_value - deltaChi2_limit) / deltaChi2_limit
        statstic = chi2_value

    elif test == 'nobkg':
        test_type = 'signal sum'
        err = (signal - crit_events) / crit_events
        statstic = signal

    if verbose:
        print(f'x: {x:.5e}, y: {y:.5e}, signal: {signal:.5e}, bkg: {bkg_sum:.5e}, {test_type}: {statstic:.5e}, err: {err:.5e}\n')

    return err

def Grid_Search(x_point, y_array, signal_fn, bkg, crit_events, save_file, test='chi2', lock=None):
    """
    x_point: the x value of the grid search
    y_array: the y array of the grid search
    signal_fn: the function to generate the signal array
    bkg: the background array
    save_file: the file to save the limits
    """
    lower_bound, upper_bound = 0, 1

    for y_point in y_array:
        signals = signal_fn(x_point, y_point)
        err = test_err(signals, bkg, x_point, y_point, test=test, crit_events=crit_events)
        if err > 0:
            lower_bound = y_point
            break

    limit = f'{x_point} {lower_bound} {upper_bound}\n'
    print(f'\n\nlimit: {limit}\n\n')
    parallel_write(limit, save_file, lock=lock)

# Binary and grid mixing search
def Binary_Search(x_point, y_array, signal_fn, bkg, crit_events, save_file, test='chi2', lock=None):
    """
    x_point: the x value of the grid search
    y_array: the y array of the grid search
    signal_fn: the function to generate the signal array
    bkg: the background array
    save_file: the file to save the limits
    """
    while True:
        mid_index = int(len(y_array)/2)
        y_point = y_array[mid_index]

        signals = signal_fn(x_point, y_point)
        err = test_err(signals, bkg, x_point, y_point, test=test, crit_events=crit_events)

        # binary search for epsilon (was set to 1.8)
        if np.abs(err) < 0.99:
            break
        else:
            err_bar = 0.25
            if err < 0:
                left = np.floor((1-err_bar) * mid_index)
                y_array = y_array[int(left):]
            else:
                right = np.ceil((1+err_bar) * mid_index)
                y_array = y_array[:int(right)]

    Grid_Search(x_point, y_array, signal_fn, bkg, crit_events, save_file, test=test, lock=lock)



####################################################
############## Display and plotting#################
####################################################
def cleanLimitData(limit_data):
    """
    Clean the calculation of the limits (mass, lower limit, upper limit)
    """
    masses = limit_data[:,0]
    upper_limit = limit_data[:,2]
    lower_limit = limit_data[:,1]
    diff_upper_lower = upper_limit - lower_limit
    upper_limit = np.delete(upper_limit, np.where(diff_upper_lower < 0))
    lower_limit = np.delete(lower_limit, np.where(diff_upper_lower < 0))
    masses = np.delete(masses, np.where(diff_upper_lower < 0))
    joined_limits = np.append(lower_limit, upper_limit[::-1])
    joined_masses = np.append(masses, masses[::-1])
    return joined_masses, joined_limits

def separate_letters_numbers(string):
    """
    Separate letters and numbers in a string
    """
    match = re.match(r'([a-zA-Z]+)([0-9]+)', string)
    if match:
        letters = match.group(1)
        numbers = match.group(2)
        return letters, numbers
    else:
        return None, None

def latex_float(f, decimal=2):
    """
    Turn a number into scientific notation latex string
    1.2e9 -> 1.2 \times 10^{9}
    1e9/3 -> 3.33 \times 10^{8}
    1e-4 -> 10^{-4}
    2e-3 -> 2 \times 10^{-3}
    """
    float_str = "{0:.{decimal}E}".format(f, decimal=decimal)
    if "E" in float_str:
        base, exponent = float_str.split("E")
        expo = int(exponent)
        base = float(base)
        if expo == 0:
            base = int(base) if base.is_integer() else base
            return r"${0}$".format(base)
        if base == 1:
            return r"$10^{{{0}}}$".format(expo)
        if base.is_integer():
            return r"${0} \times 10^{{{1}}}$".format(int(base), expo)
        return r"${0} \times 10^{{{1}}}$".format(base, expo)
    else:
        return float_str

def grid_3D(surface_points=None):
    """
    surface_points: 2D array of points in 3D space
    return: xgrid, ygrid, zgrid (must be linear)
    Use "plt.contourf(xgrid, ygrid, zgrid)" to plot
    """
    X = surface_points[:,0]
    Y = surface_points[:,1]
    Z = surface_points[:,2]

    nx = 10*int(np.sqrt(surface_points.shape[0]))
    xg = np.linspace(X.min(), X.max(), nx)
    yg = np.linspace(Y.min(), Y.max(), nx)
    xgrid, ygrid = np.meshgrid(xg, yg)
    zgrid = griddata((X, Y), Z, (xgrid, ygrid), method='linear')

    return xgrid, ygrid, zgrid

def grid_3D_demo():
    N = 10000
    surface_points = np.random.rand(N, 3) # generate random points in 3D space
    xgrid, ygrid, zgrid = grid_3D(surface_points)

    plt.contourf(xgrid, ygrid, zgrid)
    plt.show()

def plot_hist(values, edges, bin_density=False, **kwargs):
    """
    Plot the values and edges as a histogram (bar plot). The values and edges can be obtained from np.histogram
    bin_density: whether to normalize the histogram by the bin width
    """
    assert len(values) == len(edges) - 1
    if bin_density:
        # normalize the histogram by the bin width
        for idx, b in enumerate(values):
            bin_width = edges[idx+1] - edges[idx]
            values[idx] = b / bin_width
    plt.bar(x=edges[:-1], height=values, width=np.diff(edges), align='edge', **kwargs)

def stem_plot(xx, yy, color, label=None, stem_width=3, marker_size=0, alpha=1):
    """
    Stem plot of (x, y) vertical to x-axis
    """
    for x,y in zip(xx, yy):
        markerline, stemlines, baseline = plt.stem(x, y, linefmt='-')
        plt.setp(stemlines, linewidth=stem_width, color=color, alpha=alpha)
        plt.setp(markerline, markersize=marker_size, color=color)
    plt.setp(stemlines, linewidth=1, label=label, color=color)



####################################################
##################### Misc #########################
####################################################
def timer_func(func):
    """
    This function shows the execution time of the function object passed
    This acts like a decorator, eg. @timer_func above the function definition
    """
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()

        timer_mesg = f'Function {func.__name__!r} executed in {(t2-t1):.4f}s'
        mesg_color = 'green'
        print(colored(timer_mesg, mesg_color))
        return result
    return wrap_func

def find_nearest(array, value):
    """
    Find nearest element in array
    array: the array to be searched in
    value: the value to be searched for
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def parallel_write(ss, save_file, lock=None):
    """
    Write the a string to the save_file in both parallel and nonparallel way
    ss: string to be written
    save_file: the path of the file to be written
    lock: the lock object for parallel writing. If None, then write in nonparallel way
    """
    if lock:
        with lock:
            with open(save_file, 'a') as f:
                f.write(ss)
    else:
        with open(save_file, 'a') as f:
            f.write(ss)

def resume_from_last(path, complete_array):
    """
    Resume the sensitivity search from the previous run
    path: the path of the file for the previous run
    complete_array: the array of all points to be searched
    return: the array that has not been calculated
    """
    file = np.genfromtxt(path)
    calculated = file[:, 0]
    calculated_set = set(calculated)
    complete_set = set(complete_array)

    difference = complete_set - calculated_set
    return np.array(list(difference))

def sort_file(outputFile, delimiter=None):
    """
    Sort the file by the first column
    outputFile: the file to be sorted
    """
    unsorted = np.genfromtxt(outputFile, delimiter=delimiter)
    sorted_data = np.array(sorted(unsorted, key=lambda x: x[0]))
    np.savetxt(outputFile, sorted_data)

def line_prepender(filename, line):
    """
    Prepend a line of string to a file
    filename: the file to be prepended
    line: the string to be prepended
    """
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

def phys_unit_convert(input_, new_unit):
    """
    input_: string, '1.2MeV', '10GeV'
    new_unit: string, 'MeV', 'GeV'
    return: value in the new unit
    """
    from pint import UnitRegistry
    ureg = UnitRegistry()
    Q_ = ureg.Quantity
    return Q_(input_).to(new_unit).magnitude

def interp(limits):
    """
    Interpolate and extrapolate the limits (from the digitized plot)
    """
    f = interp1d(limits[:, 0], limits[:, 1], fill_value='extrapolate')
    xx = np.arange(1, 300, 0.1)
    return f(xx)

def minimize_arrays(*arrs):
    """
    Output the minimum of multiple arrays (element-wise)
    """
    min_arr = np.array([np.inf]* len(arrs[0]))
    for arr in arrs:
        min_arr = np.minimum(min_arr, arr)
    return min_arr

def search_dict_key(dict_, input_key, tolerance=1e-2):
    """
    dict_: the dictionary to be searched in
    key: the key to be searched (floats, strings, and booleans are supported)
    tolerance: the tolerance of the search
    return: the key if found, otherwise None
    """
    for dict_key in dict_.keys():
        found = True
        for i in range(len(input_key)):
            input_key_i = input_key[i]

            if isinstance(input_key_i, numbers.Number):
                if abs(dict_key[i] - input_key_i)/input_key_i > tolerance:
                    found = False
                    break
            elif isinstance(input_key_i, str) or isinstance(input_key_i, bool):
                if dict_key[i] != input_key_i:
                    found = False
                    break
            else:
                raise ValueError('Does not support the type of the input key')
        if found:
            return dict_key
    return None

def scale_cache(pos, power, base, tolerance=1e-3):
    """
    Cache decorator for time-consuming functions. The func can have several parameters, exactly one of which must be scaleable in power law
    pos: the position of the scaleable parameter in the func
    power: the power of the scaleable parameter
    base: the nonzero base value of the scaleable parameter, which should be included in the parameter space
    tolerance: the tolerance of the parameters: (v-w)/v < tolerance
    **Note: the parameters to cached must be positional arguments of the func

    Eg. f(x,a,b) = a * x**2 / b
    f is the func, x is the scaleable parameter, pos=0, power=2, base=1

    Usage:
    @scale_cache(pos=0, power=2, base=1)
    def f(x,a,b, norm=True):
        if norm:
            return a*b* x**2
        return a * x**2 / b

    f(1,2,3) # calculate and cache. here x,a,b are all positional arguments, f(1,a=2,b=3) won't work
    f(-1,2,3) # read from cache
    f(5,7,-2, norm=False) # calculate and cache, here norm won't be cached
    f(-1,7,-2, norm=False) # read from cache

    @scale_cache(pos=2, power=6, base=1e-4)
    def g(a,b,x):
        return b**a * x**6

    g(7,-2, 1e-3) # calculate and cache, base value should be around the input x value (within a few orders is fine)
    g(7,-2, 5e-6) # read from cache
    """
    if base == 0:
        raise ValueError('base cannot be zero')

    def func_cache(func):
        cache_dict = {} # cache_dict doesn't have to be shared between processes since each process have different parameters
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            scaleable = args[pos]
            input_key = args[:pos] + args[pos+1:]

            key = search_dict_key(cache_dict, input_key, tolerance)
            if key:
                value = cache_dict[key] * (scaleable/base)**power
            else:
                new_parameters = list(args)
                new_parameters[pos] = base
                new_value = func(*new_parameters, **kwargs)
                cache_dict[input_key] = new_value
                value = new_value * (scaleable/base)**power
            return value
        return wrapper
    return func_cache

if __name__ == "__main__":
    pionPlus_info2flux_BNB()