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
from inel_nu import get_nu_signals
from DM_xsec import sigmaGTsum, dsigmadErsum, sigmaGT_theta
import argparse
import warnings
warnings.filterwarnings("ignore")
global lock, outputFile
from BdNMC.DM_events import dm_flux_run as bdnmc_dm_flux


plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)


DEBUG = False
if DEBUG == True:
    outputFile = "out/temp.txt"
    PARTICLE = 'fermion' # scalar or fermion
    EXPERIMENT = 'CCM200'  # COHERENT_NaI, CCM120, CCM200, CCM_CsI, PIP2BD, SBND
    process_num = 10
    mass_ratio = 3  # m_A' / m_chi
    nu_BKG = False
    stats = 'chi2'  # chi2, t, nobkg
    crit_events = 30
    resume = False
else:
    parser = argparse.ArgumentParser(prog='DM sensitivity search',
        description="""Search for DM sensitivity
        Example: python3 DM.py -p=16 -outputFile='out/nai.txt' -particle='fermion' -exp='COHERENT_NaI' -stats='nobkg' -events=10 --nubkg
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-p', type=int, default=mp.cpu_count()-2, help='The number of processes to use')
    parser.add_argument('-outputFile', help='The file to save the sensitivity')
    parser.add_argument('-particle', help='The DM particle type: fermion or scalar')
    parser.add_argument('-exp', help='The experiment: COHERENT_NaI, CCM, CCM200, CCM_CsI, PIP2BD')
    parser.add_argument('-ratio', default=3, type=float, help='The mass ratio: m_A / m_chi')
    parser.add_argument('-stats', default='chi2', type=str, help='Statistics test: chi2, t, nobkg')
    parser.add_argument('-events', default=2.3, type=float, help='How many events to look for (only for nobkg test)')
    parser.add_argument('--nubkg', type=bool, default=False, help='If True, the background is set to be the inel nu bkg only', action=argparse.BooleanOptionalAction)
    parser.add_argument('-resume', type=bool, default=False, help='Resume from the last or not')
    args = parser.parse_args()

    process_num = args.p
    outputFile = args.outputFile
    PARTICLE = args.particle
    EXPERIMENT = args.exp
    mass_ratio = args.ratio
    nu_BKG = args.nubkg
    stats = args.stats
    crit_events = args.events
    resume = args.resume


mAs, Jis, Zs, nucl_exes = get_mA_Ji(EXPERIMENT)
prefactor, pot_rate_per_day, pim_rate, pion_rate, dist, secs, det_mass, atoms, pot_mu, pot_sigma = get_rates(EXPERIMENT)
pion_flux, brem_photons = get_pion_brem_flux(EXPERIMENT)
bkg, energy_edges, total_excess = get_bkg(EXPERIMENT, energy_cut=True)
energy_bins = (energy_edges[1:] + energy_edges[:-1])/2
q_bound = 200 if EXPERIMENT == 'SBND' else None

nu_bkg_prompt = get_nu_signals(EXPERIMENT, prompt=True, timing_cut=True, energy_bins=energy_bins)
thresdhold_dm_events = np.sqrt(np.sum(bkg - nu_bkg_prompt)) # the DM event number to search for
print(f'Experiment: {EXPERIMENT}, bkg: {np.sum(bkg)}, nu: {np.sum(nu_bkg_prompt)}, threshold DM: {thresdhold_dm_events}')
print(f'energy edges: {energy_edges}')
bkg = nu_bkg_prompt if nu_BKG else bkg


def dm_flux_gen(m_chi, m_med, epsilon, fac='default'):
    """
    fac: 'default'=pyCEvNS built-in; 'bdnmc'=BdNMC external library
    """
    if fac == 'default':
        # imiate eta flux by translating pi0 flux
        delta_m = 140 # MeV (mass translation)
        m_chi_eta = max(1, m_chi - delta_m) # at least 1 MeV
        m_med_eta = m_chi_eta * mass_ratio

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


@scale_cache(pos=2, power=4, base=1e-4)
def dm_signal_gen(m_chi, m_med, epsilon, fac='default'):
    total_flux, dm_energy_bins = dm_flux_gen(m_chi, m_med, epsilon, fac=fac)

    if EXPERIMENT == 'KARMEN':
        # just look at one single line
        eff = 0.2
        bgt, dE = 0.255, 15.1
        signal = 0
        for e_chi, f_chi in zip(dm_energy_bins, total_flux):
            signal += sigmaGT_theta(e_chi, m_chi, m_med, bgt, dE, epsilon, mAs[0], Jis[0], PARTICLE) * f_chi
        return np.array([signal*prefactor*eff])

    signals = []
    for de in energy_bins: # the bkg gamma energy
        signal = 0
        for e_chi, f_chi in zip(dm_energy_bins, total_flux):
            # total_flux is in [s^-1 MeV^2]
            signal += sigmaGTsum(e_chi, m_chi, m_med, epsilon, mAs, Jis, nucl_exes, de, PARTICLE, q_bound=q_bound) * f_chi
        signals.append(signal)
    return np.array(signals) * prefactor


def main_single(m_chi):
    m_med = m_chi * mass_ratio
    eps_array = np.logspace(-7, -2, 300)

    signal_fn = lambda m_chi, eps: dm_signal_gen(m_chi, m_med, eps)
    Binary_Search(m_chi, eps_array, signal_fn, bkg, crit_events, outputFile, test=stats, lock=lock)


def main(resume=False):
    mass_array = np.logspace(np.log10(0.99), np.log10(184), 100) # m_chi

    if resume:
        print('Resuming from last run...')
        mass_array = resume_from_last(outputFile, mass_array)
        os.system(f'rm "{outputFile}.lock"')
    else:
        os.system(f'rm "{outputFile}" "{outputFile}.lock"')
    pool = mp.Pool(processes=process_num)
    pool.map(main_single, mass_array)
    os.system(f'rm "{outputFile}.lock"')
    sort_file(outputFile, delimiter=' ')


def plot_excess():
    plt.figure(figsize=(8, 6))

    m_chi = 10
    eps = 4.74e-05
    m_med = m_chi * mass_ratio
    signals = dm_signal_gen(m_chi, m_med, eps)

    plot_hist(signals, energy_edges, fill=False, edgecolor='black', label=r'DM $m_\chi$={}MeV, $\epsilon$={}'.format(m_chi, latex_float(eps)))
    plot_hist(total_excess, energy_edges, fill=False, edgecolor='skyblue', label='excess')

    plt.ylabel('Count', fontsize=18)
    plt.xlabel('Energy [MeV]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Excess and DM events at CCM200 (3yr runs)', loc="right", fontsize=20)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    lock = FileLock(outputFile + '.lock')

    # plot_excess()
    # main_single(1)
    # dm_signal_test()
    main(resume)
