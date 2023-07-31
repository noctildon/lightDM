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
import multiprocess as mp
from filelock import FileLock
import matplotlib.pyplot as plt
from exp_config import *
from DM_xsec import dsigmadEr_el
import warnings
warnings.filterwarnings("ignore")
global lock, outputFile

outputFile = "out/elasticDM/CCM_Ge_e_temp.txt"
TIMING_CUT = True # apply cuts or not
EXPERIMENT = 'CCM_Ge_e' # COHERENT_CsI, COHERENT_CsI_2018, CCM_Ge, CCM_Ge_e

mAs, _, Zs, _ = get_mA_Ji(EXPERIMENT)
prefactor, pot_rate_per_day, pim_rate, pion_rate, dist, secs, det_mass, atoms, pot_mu, pot_sigma = get_rates(EXPERIMENT)
pion_flux, brem_photons = get_pion_brem_flux(EXPERIMENT)
bkg, energy_edges, total_excess = get_bkg(EXPERIMENT)
energy_bins = (energy_edges[1:] + energy_edges[:-1])/2
HELM = False if EXPERIMENT == 'CCM_Ge_e' else True # use Helm form factor or not. set False for CCM_Ge_e


# Efficiency function
def eff_func(er):
    """
    Efficiency function (only effective for COHERENT)
    er: recoil energy [MeV]
    """
    if EXPERIMENT in ['COHERENT_CsI', 'COHERENT_CsI_2018']:
        return effE_MeV(er)
    return 1


def dsigmadErSum(er, echi, mchi, eps, mAs, Zs, helm=True):
    if len(mAs) != len(Zs):
        raise ValueError("mAs and Zs must have same length")
    s = 0
    for mA, Z in zip(mAs, Zs):
        s += dsigmadEr_el(er, echi, mchi, eps, mA, Z, helm)
    return s


def sigmaElastic(echi, mchi, eps, mA, Z, helm=True):
    """
    echi: dark matter energy
    mchi: dark matter mass
    eps: coupling
    mA: nuclear mass
    Z: atomic number
    return: elastic DM-nucleus total cross section [MeV^-2]
    """
    ermax_ = ermax(echi, mchi, dE=0, mA=mA)
    if ermax_ <= 0:
        return 0
    return quad(dsigmadEr_el, 0, ermax_, args=(echi, mchi, eps, mA, Z, helm))[0]


def sigmaElasticSum(echi, mchi, eps, mAs, Zs, helm=True):
    """
    return: total cross section [MeV^-2] summed over all nuclei
    """
    if len(mAs) != len(Zs):
        raise ValueError("mAs and Zs must have same length")
    s = 0
    for mA, Z in zip(mAs, Zs):
        s += sigmaElastic(echi, mchi, eps, mA, Z, helm)
    return s


@scale_cache(pos=1, power=4, base=1e-4)
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
    for idx, er in enumerate(energy_bins):
        signal = 0
        for e_chi, f_chi in zip(dm_energy_bins, total_flux):
            # total_flux is in [s^-1 MeV^-2]
            signal += dsigmadErSum(er, e_chi, m_chi, epsilon, mAs, Zs, HELM) * f_chi
        er_bin_width = energy_edges[idx+1] - energy_edges[idx]
        signals.append(signal * eff_func(er) * er_bin_width)

    return np.array(signals) * prefactor


def main_single(m_chi):
    eps_array = np.logspace(-6, -1, 150)
    crit_events = 2.3
    # Binary_Search(m_chi, eps_array, dm_signal_gen, bkg, crit_events, outputFile, test='nobkg', lock=lock)
    Grid_Search(m_chi, eps_array, dm_signal_gen, bkg, crit_events, outputFile, test='nobkg', lock=lock)


def plot_elastic_DM():
    signals = dm_signal_gen(m_chi=25, epsilon=1e-4, test='chi2')
    plt.plot(energy_bins, signals)
    plt.xlim(0, 0.04)
    plt.yscale('log')
    plt.show()


def main(resume=False):
    mass_array = np.logspace(np.log10(0.99), np.log10(184), 50) # m_chi

    if resume:
        mass_array = resume_from_last(outputFile, mass_array)
        os.system(f'rm "{outputFile}.lock"')
    else:
        os.system(f'rm "{outputFile}" "{outputFile}.lock"')

    pool = mp.Pool(processes=mp.cpu_count()-2)
    pool.map(main_single, mass_array)
    os.system(f'rm "{outputFile}.lock"')
    sort_file(outputFile, delimiter=' ')


if __name__ == "__main__":
    lock = FileLock(outputFile + '.lock')

    # main_single(1)
    main()
    # plot_elastic_DM()
