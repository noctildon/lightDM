"""
All energy in MeV
"""
import os
import sys
from util import *
from pyCEvNS.events import *
from pyCEvNS.flux import *
from pyCEvNS.constants import *
import numpy as np
from scipy.stats import chi2, norm
import multiprocess as mp
from filelock import FileLock
import matplotlib.pyplot as plt
from exp_config import *
from pip3body import pip3_DM_events
import warnings
warnings.filterwarnings("ignore")
global lock, outputFile

outputFile = "out/elastic/csi.txt"

# apply cuts or not
TIMING_CUT = False

# COHERENT_CsI, COHERENT_CsI_2018
EXPERIMENT = 'COHERENT_CsI'
mAs, Jis, Zs, nucl_exes = get_mA_Ji(EXPERIMENT)
prefactor, pot_rate_per_day, pim_rate, pion_rate, dist, secs, det_mass, atoms, pot_mu, pot_sigma = get_rates(EXPERIMENT)
pion_flux, brem_photons = get_pion_brem_flux(EXPERIMENT)
bkg, energy_edges = get_bkg(EXPERIMENT)
bkg += 1 # add 1 to avoid div by 0
energy_bins = (energy_edges[1:] + energy_edges[:-1])/2  # in PE

# convert PE to MeV
pe_per_mev = 0.0878 * 13.348 * 1e3
energy_edges_meV = energy_edges / pe_per_mev
energy_bins_meV = energy_bins / pe_per_mev
bin_width = energy_bins_meV[1] - energy_bins_meV[0]


def ermax(echi, mchi, mA):
    # maximum recoil energy
    er_tolerance = 5e-3
    if echi <= mchi:
        return 0
    pchi = momentum(echi, mchi)
    return (2*pchi)**2 / (2 * mA) * (1 - er_tolerance)


def dsigmadEr(er, echi, mchi, eps, mA, Z):
    """
    er: recoil energy
    echi: dark matter energy
    mchi: dark matter mass
    eps: coupling (epsilon)
    mA: nuclear mass
    Ji: nuclear spin
    Z: atomic number
    return: elastic DM-nucleus differential cross section
    """
    if echi <= mchi + er or er >= ermax(echi, mchi, mA):
        return 0

    e = 0.30282212
    gD = np.sqrt(4*np.pi*alpha_D)
    mmed = 3*mchi
    pchi = momentum(echi, mchi)

    res = e**2 * eps**2 * gD**2 * Z**2 * FHelm(mA/mN, er)**2 / (4*np.pi)
    res /= pchi**2 * (2*mA*er + mmed **2)**2
    res *= 2*echi**2 * mA * (1- er/echi - mA*er/(2*echi**2)) + er**2 * mA
    return res


def dsigmadErSum(er, echi, mchi, eps, mAs, Zs):
    if len(mAs) != len(Zs):
        raise ValueError("mAs and Zs must have same length")
    s = 0
    for mA, Z in zip(mAs, Zs):
        s += dsigmadEr(er, echi, mchi, eps, mA, Z)
    return s


def sigmaElastic(echi, mchi, eps, mA, Z):
    """
    echi: dark matter energy
    mchi: dark matter mass
    eps: coupling
    mA: nuclear mass
    Z: atomic number
    return: elastic DM-nucleus total cross section [MeV^-2]
    """
    ermax_ = ermax(echi, mchi, mA)
    if ermax_ <= 0:
        return 0
    return quad(dsigmadEr, 0, ermax_, args=(echi, mchi, eps, mA, Z))[0]


def sigmaElasticSum(echi, mchi, eps, mAs, Zs):
    """
    return: total cross section [MeV^-2] summed over all nuclei
    """
    if len(mAs) != len(Zs):
        raise ValueError("mAs and Zs must have same length")
    s = 0
    for mA, Z in zip(mAs, Zs):
        s += sigmaElastic(echi, mchi, eps, mA, Z)
    return s


def dm_signal_gen(m_chi, epsilon):
    m_med = m_chi*3

    # imiate eta flux by translating pi0 flux
    delta_m = 140 # MeV (mass translation)
    m_chi_eta = max(1, m_chi - delta_m) # at least 1 MeV
    m_med_eta = m_chi_eta * 3

    brem_flux = DMFluxIsoPhoton(brem_photons, dark_photon_mass=m_med, coupling=epsilon, sampling_size=1000, detector_distance=dist,
                                dark_matter_mass=m_chi, pot_rate=pot_rate_per_day, life_time=default_lifetime, pot_mu=pot_mu, pot_sigma=pot_sigma)
    pi0_flux = DMFluxFromPi0Decay(pi0_distribution=pion_flux, dark_photon_mass=m_med, coupling_quark=epsilon, pot_rate=pot_rate_per_day, detector_distance=dist,
                                    dark_matter_mass=m_chi, life_time=default_lifetime, pot_mu=pot_mu, pot_sigma=pot_sigma)
    pim_flux = DMFluxFromPiMinusAbsorption(dark_photon_mass=m_med, coupling_quark=epsilon, pot_rate=pot_rate_per_day, detector_distance=dist,
                                            dark_matter_mass=m_chi, pion_rate=pim_rate, life_time=default_lifetime, pot_mu=pot_mu, pot_sigma=pot_sigma)
    eta_flux = DMFluxFromPi0Decay(pi0_distribution=pion_flux, dark_photon_mass=m_med_eta, coupling_quark=epsilon, pot_rate=pot_rate_per_day, detector_distance=dist,
                                    dark_matter_mass=m_chi_eta, life_time=default_lifetime, pot_mu=pot_mu, pot_sigma=pot_sigma)

    pi0_wgts = pi0_flux.norm * np.ones_like(pi0_flux.energy) / np.shape(pi0_flux.energy)
    pim_wgts = pim_flux.norm * np.ones_like(pim_flux.energy) / np.shape(pim_flux.energy)
    brem_wgts = np.array(brem_flux.weight) / 100
    eta_wgts = 1e-2 * eta_flux.norm * np.ones_like(eta_flux.energy) / np.shape(eta_flux.energy)

    brem_energy_flux = brem_flux.energy
    pi0_energy_flux = pi0_flux.energy
    pim_energy_flux = pim_flux.energy
    eta_energy_flux = eta_flux.energy

    dm_energy_edges = np.linspace(0, 400, 100)
    dm_energy_bins = (dm_energy_edges[1:] + dm_energy_edges[:-1])/2

    pim_energy_flux = np.histogram(pim_energy_flux, weights=pim_wgts, bins=dm_energy_edges)[0]
    pi0_energy_flux = np.histogram(pi0_energy_flux, weights=pi0_wgts, bins=dm_energy_edges)[0]
    brem_energy_flux = np.histogram(brem_energy_flux, weights=brem_wgts, bins=dm_energy_edges)[0]
    eta_energy_flux = np.histogram(eta_energy_flux, weights=eta_wgts, bins=dm_energy_edges)[0]

    # NOTE: scale eta flux to match Dan's results
    eta_energy_flux *= 10

    if TIMING_CUT:
        total_flux = pi0_energy_flux + eta_energy_flux
    else:
        total_flux = pim_energy_flux + pi0_energy_flux + brem_energy_flux + eta_energy_flux

    signals = []
    for er in energy_bins_meV:
        signal = 0
        pe = pe_per_mev * er
        for e_chi, f_chi in zip(dm_energy_bins, total_flux):
            # total_flux is in [s^-1 MeV^-2]
            signal += dsigmadErSum(er, e_chi, m_chi, epsilon, mAs, Zs) * f_chi
        signals.append(signal * effE(pe))

    # bin_rescale = 20
    bin_rescale = 1
    return np.array(signals) * prefactor * bin_width / bin_rescale


# interaction_model: vector_ib2, vector_contact, vector_ib2
def dm_signal_gen_3body(m_chi, epsilon, interaction_model='vector_ib9', abd=(0,0,0)):
    dm_energy_bins, dm_flux = pip3_DM_events(m_chi*3, epsilon, abd=abd,
                            experiment=EXPERIMENT, interaction_model=interaction_model)

    dm_flux /= 4*np.pi * dist**2  # convert to [count meter^-2]
    dm_flux *= det_mass * atoms
    dm_flux *= meter_by_mev**2

    signals = []
    for er in energy_bins_meV:
        signal = 0
        pe = pe_per_mev * er
        for e_chi, f_chi in zip(dm_energy_bins, dm_flux):
            # total_flux is in [s^-1 MeV^-2]
            signal += dsigmadErSum(er, e_chi, m_chi, epsilon, mAs, Zs) * f_chi
        signals.append(signal * effE(pe))
    return np.array(signals) * bin_width


confidence_limit = 0.9
deltaChi2_limit = chi2.ppf(confidence_limit, len(bkg)) - len(bkg)
def deltaChi2(sig):
    return np.sum((sig)**2 / bkg)


# Convert confidence level to gaussian sigma (sigificance)
# 0.68->1; 0.95->2; 0.997->3
def cl2sig(cl):
    if cl > 1 or cl < 0:
        raise ValueError('CL must be between 0 and 1')
    return abs(norm.ppf((1-cl)/2))
sig_limit = cl2sig(confidence_limit)


# err = (statstic - limit) / limit
def test_err(signals, mass, eps, test='chi2', verbose=True):
    test_type = 'sig' if test == 't' else 'chi2'
    signal = np.sum(signals)
    if test == 't':
        bkg_sum = np.sum(bkg)
        # sig = (signal - bkg_sum) / np.sqrt(bkg_sum)
        sig = signal / np.sqrt(bkg_sum)  # the fixed t-test
        err = (sig - sig_limit) / sig_limit

    elif test == 'chi2':
        chi2 = deltaChi2(signals)
        err = (chi2 - deltaChi2_limit) / deltaChi2_limit

    if verbose:
        statstic = sig if test == 't' else chi2
        print(f'mass: {mass}, eps: {eps}, signal: {signal}, {test_type}: {statstic}, err: {err}\n')

    return err


def Grid_Search(mass, epsilon_array, save_file, test='chi2', lock=None, pi3=False):
    lower_bound, upper_bound = 0, 1

    for eps in epsilon_array:
        signals = dm_signal_gen(mass, eps)
        if pi3:
            signals += dm_signal_gen_3body(mass, eps, abd=(1e-6, 1e-6, 1e-6), interaction_model='vector_ib9')
        err = test_err(signals, mass, eps, test=test)
        if err > 0:
            lower_bound = eps
            break

    limit = str(mass) + " " + str(lower_bound) + " " + str(upper_bound) + '\n'
    print(f'\n\nlimit: {limit}\n\n')
    parallel_write(limit, save_file, lock=lock)


# Binary and grid mixing search
def Binary_Search(mass, epsilon_array, save_file, test='chi2', lock=None, pi3=False):
    while True:
        mid_index = int(len(epsilon_array)/2)
        eps = epsilon_array[mid_index]
        signals = dm_signal_gen(mass, eps)
        if pi3:
            signals += dm_signal_gen_3body(mass, eps, abd=(1e-6, 1e-6, 1e-6), interaction_model='vector_ib9')
        err = test_err(signals, mass, eps, test=test)

        # binary search for epsilon
        if np.abs(err) < 0.998:
            break
        else:
            err_bar = 0.0
            if err < 0:
                left = np.float((1-err_bar) * mid_index)
                epsilon_array = epsilon_array[int(left):]
            else:
                right = np.ceil((1+err_bar) * mid_index)
                epsilon_array = epsilon_array[:int(right)]

    Grid_Search(mass, epsilon_array, save_file, test=test, lock=lock, pi3=pi3)


def main_single(mass):
    eps_array = np.logspace(-6, -1, 200)
    Binary_Search(mass, eps_array, outputFile, lock=lock, test='chi2', pi3=False)


def testing():
    m_chi = 30
    eps = 1e-4
    signal = dm_signal_gen(m_chi, eps, test='chi2')
    abd = (1e-6, 1e-6, 1e-6)
    # signal3 = dm_signal_gen_3body(m_chi, eps, interaction_model='vector_ib9', abd=abd, test='chi2')
    # print(signal)
    print('DM  signal sum:', np.sum(signal))
    # print('DM (3body) signal sum:', np.sum(signal3))
    # print('ratio 3-body/pi0:', np.sum(signal3) / np.sum(signal))


def plot_elastic_DM():
    signals = dm_signal_gen(m_chi=25, epsilon=1e-4, test='chi2')
    plt.plot(energy_bins_meV, signals)
    plt.xlim(0, 0.04)
    plt.yscale('log')
    plt.show()


def main(resume=False):
    mass_array = np.logspace(np.log10(0.99), np.log10(184), 50) # m_chi
    # mass_array = np.logspace(np.log10(0.99), np.log10(10), 2) # m_chi

    if resume:
        mass_array = resume_from_last(outputFile)
        os.system(f'rm "{outputFile}.lock"')
    else:
        os.system(f'rm "{outputFile}" "{outputFile}.lock"')

    process_num = int(sys.argv[1])
    pool = mp.Pool(processes=process_num)
    pool.map(main_single, mass_array)
    os.system(f'rm "{outputFile}.lock"')


if __name__ == "__main__":
    lock = FileLock(outputFile + '.lock')

    # main_single(1)
    main()
    # testing()
    # plot_elastic_DM()
