# This includes some useful functions for the project.
from time import time
from termcolor import colored
from pyCEvNS.helper import _gaussian, _poisson
from pyCEvNS.constants import *
import numpy as np
from scipy.stats import chi2, norm
from scipy.integrate import quad
from scipy.special import gamma
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def momentum(e, m):
    return np.sqrt(e**2 - m**2)


def FHelm(A, er):
    """
    Helm form factor
    er: recoil energy in MeV
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


def gaussian(x, mu, sigma):
    return 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-np.power((x - mu)/sigma, 2.)/2)


# cauchy distribution
def cauchy(x, x0, gamma):
    return 1/(np.pi*gamma*(1+((x-x0)/gamma)**2))


# n_obs, n_bg, n_nu are the the sum of the arrays
def excessANDsig(n_obs, n_bg, n_nu):
    excess = n_obs - n_nu - n_bg
    sig = excess / np.sqrt(n_nu + n_bg)
    return excess, sig


def excessANDsig_(n_obs, n_bg, n_nu):
    excess = n_obs - n_nu - n_bg
    sig = (n_obs - n_nu - n_bg) / np.sqrt(n_nu + n_bg + 1)
    return np.sum(excess), np.sum(sig)


# The mixed likelihood function
def loglike_csi(n_signal, n_obs, n_bg, sigma):
    likelihood = np.ones(len(n_obs))
    for i in range(len(n_obs)):
        n_bg_list = np.arange(max(0, int(n_bg[i] - 2 * np.sqrt(n_bg[i]))),
                                max(10, int(n_bg[i] + 2 * np.sqrt(n_bg[i]))))
        for nbgi in n_bg_list:
            likelihood[i] += quad(lambda a: _poisson(n_obs[i], (1 + a) * n_signal[i] + nbgi) *
                                    _gaussian(a, 0, sigma), -3 * sigma, 3 * sigma)[0] * _poisson(n_bg[i], nbgi)
    return np.sum(np.log(likelihood))


# try to find the poisson likelihood
def loglike_chi2(n_signal, n_obs, n_bg, err):
    err = err + 1
    likelihood = (n_obs - (n_signal + n_bg)) **2 / err**2
    return np.sum(likelihood)


# PE to keV mapping (the old one)
def pe2(pe):
    pe2keV = np.genfromtxt('Experiments/COHERENT/PE2keV.txt', delimiter=' ')
    pe2keVmap = {}
    for i in range(pe2keV.shape[0]):
        pe2keVmap[str(int(pe2keV[i,0]))] = pe2keV[i,1]
    return pe2keVmap[str(pe)]


# convert keV to PE
def keV2PE(keV):
    a = 0.0554628
    b = 4.30681
    c = -111.707
    d = 840.384
    pe = 13.35*(a* keV/10**3 + b*keV**2/10**6 + c*keV**3/10**9 + d*keV**4/10**12)* 10**3
    return pe


# convert PE to keV
def PE2keV(pe_const):
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
    a = 1 / pe
    b = 0.716 * pe
    term1 = (a*(1+b))**(1+b) / gamma(1+b)
    term2 = x**b * np.exp(-a*(1+b)*x)
    return term1 * term2


def effT(t):
    a = 0.52
    b = 0.0494
    if t < a:
        return 1
    res = np.exp(-b*(t-a))
    return res

def effE(pe):
    a = 1.32045
    b = 0.285979
    c = 10.8646
    d = -0.333322
    res = a/(1+np.exp(-b * (pe - c))) + d
    if res < 0:
        return 0
    return res


def efficiency(pe, t):
    # return 1
    res = effE(pe) * effT(t)
    if res < 0:
        return 0
    return res


# This function shows the execution time of the function object passed
def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()

        timer_mesg = f'Function {func.__name__!r} executed in {(t2-t1):.4f}s'
        mesg_color = 'green'
        print(colored(timer_mesg, mesg_color))
        return result
    return wrap_func


def cleanLimitData(limit_data):
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


# find nearest element in array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def parallel_write(limit, save_file, lock=None):
    if lock:
        with lock:
            with open(save_file, 'a') as f:
                f.write(limit)
    else:
        with open(save_file, 'a') as f:
            f.write(limit)


def pion_info2flux(pi0Info):
    pion_energy = pi0Info[:,4] - massofpi0
    pion_azimuth = np.arccos(pi0Info[:,3] / np.sqrt(pi0Info[:,1]**2 + pi0Info[:,2]**2 + pi0Info[:,3]**2))
    pion_cos = np.cos(np.pi/180 * pi0Info[:,0])
    pion_flux = np.array([pion_azimuth, pion_cos, pion_energy]).transpose()
    return pion_flux


def pion_info2flux_shielded():
    pi0Info = np.genfromtxt("Experiments/PIP2BD/pizero_combined2GeV_shieldedCTarget.txt")

    # down sample to 6000
    random_idx = np.random.randint(len(pi0Info), size=6000)
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


def conv_strength(dE, nucl_ex, width_keV=150):
    """
    dE: excitation energy = deexcitation photon energy
    nucl_ex: bigstick nuclear strength
    width_keV: width of the gaussian function
    RETURN the convolution
    """
    bigstick_energy = nucl_ex[:, 0]
    bigstick_strength = nucl_ex[:, 1]
    def totals(x):
        gau = 0
        for i in range(len(bigstick_energy)):
            gau += gaussian(x, bigstick_energy[i], width_keV*1e-3) * bigstick_strength[i]
        return gau

    return totals(dE)


def resume_from_last(path):
    file = np.genfromtxt(path)
    calculated = file[:, 0]
    calculated_set = set(calculated)

    mass_array = np.logspace(np.log10(0.99), np.log10(184), 100) # m_chi
    mass_array_set = set(mass_array)

    difference = mass_array_set - calculated_set
    return np.array(list(difference))


# sort by 1st column
def sort_file(outputFile, delimiter=None):
    unsorted = np.genfromtxt(outputFile, delimiter=delimiter)
    sorted_data = np.array(sorted(unsorted, key=lambda x: x[0]))
    np.savetxt(outputFile, sorted_data)


# Prepend line (string) to a file
def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


def deltaChi2(sig, bkg):
    return np.sum((sig)**2 / bkg)


# Convert confidence level to gaussian sigma (sigificance)
# 0.68->1; 0.95->2; 0.997->3
def cl2sig(cl):
    if cl > 1 or cl < 0:
        raise ValueError('CL must be between 0 and 1')
    return abs(norm.ppf((1-cl)/2))


def grid_3D(surface_points=None):
    """
    :param surface_points: 2D array of points in 3D space
    :return: xgrid, ygrid, zgrid (must be linear)
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
