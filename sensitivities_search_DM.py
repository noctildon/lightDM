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
from scipy.stats import chi2
import multiprocess as mp
from filelock import FileLock
import matplotlib.pyplot as plt
from exp_config import *
from pip3body import pip3_DM_events
from rebin import rebin_bkg
from inel_nu import get_nu_signals
from DM_xsec import sigmaGTsum
import argparse
import warnings
warnings.filterwarnings("ignore")
global lock, outputFile
# from Anna_PLB.plot import cc_ratio
from BdNMC.DM_events import dm_flux_run as bdnmc_dm_flux


strength_rescale = get_strength_scale() # 0.16212
# strength_rescale = 1/11.5

DEBUG = True

if DEBUG == True:
    outputFile = "out/temp.txt"
    PARTICLE = 'fermion' # scalar or fermion
    EXPERIMENT = 'PIP2BD'  # COHERENT_NaI, CCM, CCM_CsI, PIP2BD
    process_num = 10
    BKG_free = False
else:
    parser = argparse.ArgumentParser(prog='DM sensitivity search',
        description="""Search for DM sensitivity
        Example: python3 sensitivities_search_DM.py -p=16 -outputFile='out/nai.txt' -particle='fermion' -exp='COHERENT_NaI' --bkgfree
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-p', type=int, default=mp.cpu_count()-2, help='The number of processes to use')
    parser.add_argument('-outputFile', help='The file to save the sensitivity')
    parser.add_argument('-particle', help='The DM particle type: fermion or scalar')
    parser.add_argument('-exp', help='The experiment: COHERENT_NaI, CCM, CCM_CsI, PIP2BD')
    parser.add_argument('--bkgfree', type=bool, default=False, help='If True, the background is set to be the inel nu bkg only', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    process_num = args.p
    outputFile = args.outputFile
    PARTICLE = args.particle
    EXPERIMENT = args.exp
    BKG_free = args.bkgfree


mAs, Jis, Zs, nucl_exes = get_mA_Ji(EXPERIMENT)
prefactor, pot_rate_per_day, pim_rate, pion_rate, dist, secs, det_mass, atoms, pot_mu, pot_sigma = get_rates(EXPERIMENT)
pion_flux, brem_photons = get_pion_brem_flux(EXPERIMENT)
bkg, energy_edges = get_bkg(EXPERIMENT, energy_cut=True)
energy_bins = (energy_edges[1:] + energy_edges[:-1])/2
bin_scale = 1

# rebin the bkg and add the inel nu bkg
bkg, energy_edges, bin_scale = rebin_bkg(bkg, energy_edges)
energy_bins = (energy_edges[1:] + energy_edges[:-1])/2
nu_bkg_prompt = get_nu_signals(EXPERIMENT, prompt=True, timing_cut=True, energy_bins=energy_bins)
nu_bkg_prompt *= strength_rescale

if BKG_free:
    bkg = nu_bkg_prompt
else:
    bkg += nu_bkg_prompt


# statistics test
confidence_limit = 0.9
deltaChi2_limit = chi2.ppf(confidence_limit, len(bkg)) - len(bkg)
sig_limit = cl2sig(confidence_limit)
def test_err(signals, mass, eps, test='chi2', verbose=True):
    """
    err = (statstic - limit) / limit
    """
    test_type = 'sig' if test == 't' else 'chi2'
    signal = np.sum(signals)
    if test == 't':
        bkg_sum = np.sum(bkg)
        # sig = (signal - bkg_sum) / np.sqrt(bkg_sum)
        sig = signal / np.sqrt(bkg_sum)  # the fixed t-test
        err = (sig - sig_limit) / sig_limit

    elif test == 'chi2':
        chi2 = deltaChi2(signals, bkg)
        err = (chi2 - deltaChi2_limit) / deltaChi2_limit

    if verbose:
        statstic = sig if test == 't' else chi2
        print(f'mass: {mass}, eps: {eps}, signal: {signal}, {test_type}: {statstic}, err: {err}\n')

    return err


def dm_flux_gen(m_chi, epsilon, fac='default'):
    """
    fac: 'default'=pyCEvNS built-in; 'bdnmc'=BdNMC external library
    """
    if fac == 'default':
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
        eta_energy_flux *= 10  # scale eta flux to match Dan's results
        total_flux = pim_energy_flux + pi0_energy_flux + brem_energy_flux + eta_energy_flux # [s^-1 MeV^2]

    elif fac == 'bdnmc':
        dm_flux, rdm_flux, dm_energy_bins = bdnmc_dm_flux(m_chi, epsilon, exp=EXPERIMENT)
        total_flux = dm_flux
        total_flux *= meter_by_mev**2 / 4*np.pi*dist**2  # [MeV^2]
        total_flux /= secs # [s^-1 MeV^2]

    return total_flux, dm_energy_bins


def dm_signal_gen(m_chi, epsilon, fac='default'):
    total_flux, dm_energy_bins = dm_flux_gen(m_chi, epsilon, fac=fac)

    signals = []
    for de in energy_bins: # the bkg gamma energy
        signal = 0
        for e_chi, f_chi in zip(dm_energy_bins, total_flux):
            # total_flux is in [s^-1 MeV^2]
            signal += sigmaGTsum(e_chi, m_chi, epsilon, mAs, Jis, nucl_exes, de, PARTICLE) * f_chi
        signals.append(signal)

    # normalize by bin width 20 and bin scale
    return np.array(signals) * prefactor / 20 / bin_scale * strength_rescale


# interaction_model: vector_ib2, vector_contact, vector_ib2
def dm_signal_gen_3body(m_chi, epsilon, interaction_model='vector_ib9', abd=(0,0,0)):
    signals = []
    dm_energy_bins, dm_flux = pip3_DM_events(m_chi*3, epsilon, abd=abd, experiment=EXPERIMENT,
                                interaction_model=interaction_model)

    dm_flux /= 4*np.pi * dist**2  # convert to [count meter^-2]
    dm_flux *= det_mass * atoms
    dm_flux *= meter_by_mev**2

    for de in energy_bins:
        signal = 0
        for e_chi, f_chi in zip(dm_energy_bins, dm_flux):
            signal += sigmaGTsum(e_chi, m_chi, epsilon, mAs, Jis, nucl_exes, de, PARTICLE) * f_chi
        signals.append(signal)
    return np.array(signals)


def Grid_Search(mass, epsilon_array, save_file, test='chi2', lock=None):
    lower_bound = 0
    upper_bound = 1

    for eps in epsilon_array:
        signals = dm_signal_gen(mass, eps)
        # signals += dm_signal_gen_3body(mass, eps, abd=(1e-6, 1e-6, 1e-6), interaction_model='vector_ib9')
        err = test_err(signals, mass, eps, test=test)
        if err > 0:
            lower_bound = eps
            break

    limit = str(mass) + " " + str(lower_bound) + " " + str(upper_bound) + '\n'
    print(f'\n\nlimit: {limit}\n\n')
    parallel_write(limit, save_file, lock=lock)


# Binary and grid mixing search
def Binary_Search(mass, epsilon_array, save_file, test='chi2', lock=None):
    while True:
        mid_index = int(len(epsilon_array)/2)
        eps = epsilon_array[mid_index]
        signals = dm_signal_gen(mass, eps)
        # signals += dm_signal_gen_3body(mass, eps, abd=(1e-6, 1e-6, 1e-6), interaction_model='vector_ib9')
        err = test_err(signals, mass, eps, test=test)

        # binary search for epsilon (was set to 1.8)
        if np.abs(err) < 0.99:
            break
        else:
            err_bar = 0.25
            if err < 0:
                left = np.float((1-err_bar) * mid_index)
                epsilon_array = epsilon_array[int(left):]
            else:
                right = np.ceil((1+err_bar) * mid_index)
                epsilon_array = epsilon_array[:int(right)]

    Grid_Search(mass, epsilon_array, save_file, test=test, lock=lock)


def main_single(mass):
    eps_array = np.logspace(-7, -2, 300)

    # Grid_Search(mass, eps_array, outputFile, lock=lock, test='chi2')
    Binary_Search(mass, eps_array, outputFile, lock=lock, test='chi2')


def point():
    mass = 1
    eps = 5e-5
    signals = dm_signal_gen(mass, eps)

    chi2 = deltaChi2(signals)
    print(f'mass: {mass}, eps: {eps}, chi2: {chi2}')
    print(f'deltaChi2_limit {deltaChi2_limit}')

    signal = np.sum(signals)
    bkgsum = np.sum(bkg)
    sig = (signal - bkgsum) / np.sqrt(bkgsum)
    print(f'signal: {signal}, bkg: {bkgsum}, sig: {sig}')

    plt.bar(energy_edges[:-1], height=signals, width=np.diff(energy_edges), align='edge', alpha=0.5, fc='None', ec='red', linewidth=2, label='signal')
    plt.bar(energy_edges[:-1], height=bkg, width=np.diff(energy_edges), align='edge', alpha=0.5, fc='None', ec='black', label='bkg')
    plt.yscale('log')
    plt.xlabel('Deexcitation photon energy [MeV]')
    plt.ylabel('Counts')
    plt.xlim(0, 60)
    # plt.xlim(0, 100)
    plt.ylim(1e-3, 1e5)
    plt.legend()
    plt.title(f'Exp: {EXPERIMENT}, mass: {mass} MeV, eps: {eps}')
    plt.tight_layout()
    plt.show()


def testing():
    m_chi = 3.058458393833251066e+01
    eps = 1.688203316700362703e-04

    signal = dm_signal_gen(m_chi, eps)
    print(EXPERIMENT)
    print(deltaChi2_limit)
    print(deltaChi2(signal, bkg))
    return

    title = r'$m_\chi=$'+f'{m_chi}MeV,' + r'$\epsilon=10^{-4}$'
    plt.figure(figsize=(8, 6))
    # plt.plot(energy_bins, signal_rescale, label='rescale')
    plt.plot(energy_bins, signal, label='no rescale')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Photon signal energy [MeV]', fontsize=18)
    plt.ylabel('Events ratio', fontsize=18)
    plt.legend(fontsize=18)
    plt.title(title, fontsize=22, loc='right')
    plt.tight_layout()
    plt.show()


def main():
    mass_array = np.logspace(np.log10(0.99), np.log10(184), 100) # m_chi

    os.system(f'rm "{outputFile}" "{outputFile}.lock"')
    pool = mp.Pool(processes=process_num)
    pool.map(main_single, mass_array)
    os.system(f'rm "{outputFile}.lock"')


if __name__ == "__main__":
    lock = FileLock(outputFile + '.lock')

    # main_single(1)
    # main()
    # point()
    testing()
