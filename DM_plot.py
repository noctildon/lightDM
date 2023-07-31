import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyCEvNS.events import *
from pyCEvNS.flux import *
from pyCEvNS.constants import *
from util import *
from inel_nu import get_nu_signals
from exp_config import *
from DM_xsec import sigmaGTsum, sigmaGT_theta
from pathlib import Path
home = str(Path.home())

colors = mcolors.TABLEAU_COLORS
color_tabs = list(colors.keys())
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)


# COHERENT_CsI, COHERENT_NaI, CCM
EXPERIMENT = 'COHERENT_NaI'
mAs, Jis, Zs, nucl_exes = get_mA_Ji(EXPERIMENT)
prefactor, pot_rate_per_day, pim_rate, pion_rate, dist, secs, det_mass, atoms, pot_mu, pot_sigma = get_rates(EXPERIMENT)
pion_flux, brem_photons = get_pion_brem_flux(EXPERIMENT)
bkg, energy_edges, total_excess = get_bkg(EXPERIMENT)


def plot_det_bkg():
    plt.figure(figsize=(8, 6))
    experiments = ['CCM120', 'CCM200'] # 2.25e+22, 5.28e+22

    for idx, experiment in enumerate(experiments):
        bkg, energy_edges, total_excess = get_bkg(experiment, energy_cut=False)
        plot_hist(bkg, energy_edges, fc=colors[color_tabs[idx]], alpha=.25, label=experiment)
        print(f'{experiment} bkg sum {np.sum(bkg)}')

    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'Count', fontsize=18)
    plt.xlabel('Energy [MeV]', fontsize=18)
    plt.xlim(1, 100)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Background (3 year run; no bkg reduction)', loc="right", fontsize=20)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_total_bkg():
    plt.figure(figsize=(8, 6))
    plot_hist(*get_bkg('CCM120')[:2], bin_density=True, fc='blue', alpha=.25, label='CCM120')
    plot_hist(*get_bkg('COHERENT_NaI')[:2], bin_density=True, fc='orangered', alpha=.25, label='NaI')

    # Inelastic nu bkg (prompt only)
    energy_bins = np.linspace(0, 100, 1000)
    bin_width = 0.1
    conv_width = 0.15
    signals_ccm = get_nu_signals('CCM120', energy_bins=energy_bins, timing_cut=True, prompt=True, conv_width=conv_width) / bin_width
    signals_nai = get_nu_signals('COHERENT_NaI', energy_bins=energy_bins, timing_cut=True, prompt=True, conv_width=conv_width) / bin_width

    plt.plot(energy_bins, signals_ccm, color='blue', label=r'CCM $\nu$')
    plt.plot(energy_bins, signals_nai, color='orangered', label=r'NaI $\nu$')

    # plt.yscale('log')
    plt.xlabel('Energy [MeV]', fontsize=18)
    plt.ylabel('Events / MeV', fontsize=18)
    plt.xlim(0, 20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Background', loc="right", fontsize=20)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_total_bkg_pip2bd():
    plt.figure(figsize=(8, 6))

    # Detector bkg
    bkg, energy_edges = get_bkg('PIP2BD')
    bkg_bin_width = (energy_edges[-1] - energy_edges[0]) / len(energy_edges)
    bkg_bin_rescale = 1 / bkg_bin_width / 0.1 # change the unit to per 0.1MeV
    plt.bar(x=energy_edges[:-1], height=bkg_bin_rescale*bkg, width=np.diff(energy_edges), align='edge', fc='green', alpha=.25, label='PIP2BD')

    # Inelastic nu bkg (prompt only)
    energy_bins = np.linspace(0, 100, 1000)
    bin_width = 0.1
    signals_pip2bd = get_nu_signals('PIP2BD', energy_bins=energy_bins, timing_cut=True, prompt=True) / bin_width
    plt.plot(energy_bins, signals_pip2bd, color='green', label=r'PIP2BD $\nu_\mu$')

    plt.xlabel('Energy [MeV]', fontsize=18)
    plt.ylabel('Events / 0.1 MeV', fontsize=18)
    plt.xlim(0, 20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Background', loc="right", fontsize=20)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.show()


def energy_flux_plot(m_chi, epsilon, mass_ratio=3):
    plt.figure(figsize=(8, 6))
    m_med = m_chi*mass_ratio

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

    # FIXME: Is simulate necessary?
    # brem_flux.simulate()
    # pim_flux.simulate()
    # pi0_flux.simulate()

    brem_wgts_const = 100 # FIXME: check this
    pi0_wgts = pi0_flux.norm * np.ones_like(pi0_flux.energy) / np.shape(pi0_flux.energy) / (meter_by_mev**2)
    pim_wgts = pim_flux.norm * np.ones_like(pim_flux.energy) / np.shape(pim_flux.energy) / (meter_by_mev**2)
    eta_wgts = 1e-2 * eta_flux.norm * np.ones_like(eta_flux.energy) / np.shape(eta_flux.energy) / (meter_by_mev**2)
    brem_wgts = np.array(brem_flux.weight) / meter_by_mev**2 / brem_wgts_const

    brem_energy_flux = brem_flux.energy
    pi0_energy_flux = pi0_flux.energy
    pim_energy_flux = pim_flux.energy
    eta_energy_flux = eta_flux.energy

    # FIXME: what unit is this flux?
    bin_width = 1 # MeV
    energy_edges = np.arange(m_chi, 300 + bin_width, bin_width)

    pim_energy_flux, pim_edges, _ = plt.hist(pim_energy_flux, weights=pim_wgts / bin_width, histtype='step', label=r'$\pi^-$', bins=energy_edges)
    pi0_energy_flux, pi0_edges, _ = plt.hist(pi0_energy_flux, weights=pi0_wgts / bin_width, histtype='step', label=r'$\pi^0$', bins=energy_edges)
    brem_energy_flux, brem_edges, _ = plt.hist(brem_energy_flux, weights=brem_wgts / bin_width, histtype='step', label='brem', bins=energy_edges)
    eta_energy_flux, eta_edges, _ = plt.hist(eta_energy_flux, weights=eta_wgts / bin_width, histtype='step', label=r'$\eta$', bins=energy_edges)

    # pip3 body decay (IB9)
    # e_bins, dm_flux = pip3_DM_events(m_aprime=m_med, epsilon=epsilon, interaction_model='vector_ib9', abd=(1e-6, 1e-6, 1e-6))
    # dm_flux /= 4*np.pi * dist**2 * 365 * 24 * 3600  # convert to [count meter^-2 s^-1]
    # plt.plot(e_bins, dm_flux, drawstyle='steps-mid', label=r'$\pi^+$')

    title = 'DM flux energy distribution '
    plt.yscale('log')
    plt.ylabel(r'Flux [m$^{-2}$ s$^{-1}$ MeV$^{-1}$]', fontsize=18)
    plt.xlabel('DM energy [MeV]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, loc="right", fontsize=20)
    plt.legend(fontsize=16, loc='upper right')
    plt.tight_layout()
    plt.show()


def timing_flux_plot(m_chi, epsilon, mass_ratio=3):
    m_med = m_chi*mass_ratio

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

    brem_wgts_const = 100
    pi0_wgts = pi0_flux.norm * np.ones_like(pi0_flux.energy) / np.shape(pi0_flux.energy) / (meter_by_mev**2)
    pim_wgts = pim_flux.norm * np.ones_like(pim_flux.energy) / np.shape(pim_flux.energy) / (meter_by_mev**2)
    eta_wgts = 1e-2 * eta_flux.norm * np.ones_like(eta_flux.energy) / np.shape(eta_flux.energy) / (meter_by_mev**2)
    brem_wgts = np.array(brem_flux.weight) / meter_by_mev**2 / brem_wgts_const

    brem_timing_flux = brem_flux.timing
    pi0_timing_flux = pi0_flux.timing
    pim_timing_flux = pim_flux.timing
    eta_timing_flux = eta_flux.timing

    bin_edges = np.linspace(0, 2, 200)

    pim_timing_flux = plt.hist(pim_timing_flux, weights=pim_wgts, histtype='step', label=r'$\pi^-$', bins=bin_edges)[0]
    pi0_timing_flux = plt.hist(pi0_timing_flux, weights=pi0_wgts, histtype='step', label=r'$\pi^0$', bins=bin_edges)[0]
    brem_timing_flux = plt.hist(brem_timing_flux, weights=brem_wgts, histtype='step', label='brem', bins=bin_edges)[0]
    eta_timing_flux = plt.hist(eta_timing_flux, weights=eta_wgts, histtype='step', label=r'$\eta$', bins=bin_edges)[0]

    plt.yscale('log')
    plt.ylabel(r'Flux [m$^{-2}$ s$^{-1}$]', fontsize=18)
    plt.xlabel('t [mus]', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim((-0.25, 2))
    plt.title('DM flux timing distribution', loc="right", fontsize=20)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def strength_plot(nucleus, label, color_idx=0, conv_width=False, show=True, exp=False):
    """
    nucleus: 'na23', 'ge76', 'ar40', 'ar39', 'cs133', 'I127', 'o16', 'n14', 'he4', 'c12'
    If exp is True, plot experimental data, otherwise plot bigstick data
    """
    if exp:
        title = 'Experiment Gamow-Teller strength'
        nucl = pd.read_csv(f'bigstick/exp data/{nucleus}.csv', skipinitialspace=True)
        nucl['B(M1)'].replace('', np.nan, inplace=True)
        nucl.dropna(subset=['B(M1)'], inplace=True)
        nucl = nucl[['energy [keV]', 'B(M1)', 'multipolarity']]
        energy = nucl['energy [keV]'].values * 1e-3  # keV to MeV
        strength = nucl['B(M1)'].values / 2.2993**2  # B(M1) to dimensionless B(GT)
    else:
        title = 'BIGSTICK Gamow-Teller strength'
        nucl_ex_path = f'bigstick/{nucleus}_s.res'
        nucl_ex = np.genfromtxt(nucl_ex_path, delimiter='   ')
        nucl_ex[:, 0] = nucl_ex[:, 0] - nucl_ex[0, 0]
        nucl_ex = nucl_ex[np.where(nucl_ex[:, 1] != 0)]
        energy = nucl_ex[:, 0]
        strength = nucl_ex[:, 1]

    if conv_width == 0:
        stem_plot(energy, strength, color=colors[color_tabs[color_idx]], label=label, stem_width=1.5, marker_size=3, alpha=0.5)
        plt.yscale('log')
    else:
        def totals(x):
            gau = 0
            for i in range(len(energy)):
                gau += gaussian(x, energy[i], conv_width) * strength[i]
            return gau

        xx = np.arange(0, 50, 0.01)   # for o16
        # xx = np.arange(0, 20, 0.01)   # for others
        yy = np.array([totals(x) for x in xx])
        plt.plot(xx, yy, color=colors[color_tabs[color_idx]], label=label)

    plt.title(title, fontsize=20, loc='right')
    plt.legend(fontsize=16)
    plt.ylabel('GT Strength', fontsize=18)
    plt.xlabel('Deexcitation energy [MeV]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if show:
        plt.show()


def strength_plots(nuclei, conv_width=0, exp=False):
    """
    nuclei = [nucleus]
    If exp is True, plot experimental data, otherwise plot bigstick data
    """
    plt.figure(figsize=(8, 6))
    for i, nucleus in enumerate(nuclei):
        label = nucleus.capitalize()
        strength_plot(nucleus, label, color_idx=i, conv_width=conv_width, show=False, exp=exp)
    plt.show()


def plot_DM_xsec(m_chi, eps=1e-4, mass_ratio=3):
    dm29 = np.genfromtxt('bigstick/c12_dm13.csv', delimiter=',') # BIGSTICK multipole
    # dm29[:, 1] *= np.sqrt(8)

    m_med = mass_ratio * m_chi
    mAs, Jis, Zs, _ = get_mA_Ji('c12')
    nucl_exes = [np.array([[15.1, 0.255]])] # KARMEN experiment
    e_chi_arr = np.arange(m_chi, 200, 1)

    def xsecGT(echi):
        s = 0
        for mA, Ji, nucl_ex in zip(mAs, Jis, nucl_exes):
            for dE, bgt in nucl_ex:
                s += sigmaGT_theta(echi, m_chi, m_med, bgt, dE, eps, mA, Ji, particle='fermion')
        return s

    signal_arr = [xsecGT(e_chi) for e_chi in e_chi_arr]
    signal_arr = np.array(signal_arr) * meter_by_mev**2 * 1e4

    plt.figure(figsize=(8, 6))
    plt.plot(e_chi_arr, signal_arr, label=r'KARMEN 15.1MeV $S_{\rm GT}=$0.255')
    plt.plot(dm29[:, 0], dm29[:, 1], label='BIGSTICK multipole')
    plt.ylabel(r'Inelastic cross section [cm$^2$]', fontsize=18)
    plt.xlabel('DM energy [MeV]', fontsize=18)
    plt.xlim(30, 75)
    plt.ylim(0, 3e-35)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.title(r"$^{12}$C $1^+$ state comparison. $m_\chi$=30MeV, $\epsilon=10^{-4}$", fontsize=22, loc='right')
    plt.tight_layout()
    plt.show()


def plot_conv_DM_xsec(e_chi, m_chi, eps=1e-4, mass_ratio=3):
    deex = np.arange(0, 20, 0.01)
    m_med = mass_ratio * m_chi
    signals = [sigmaGTsum(e_chi, m_chi, m_med, eps, mAs, Jis, nucl_exes, de, particle='fermion', conv_width=1) for de in deex]
    signals = np.array(signals) * meter_by_mev**2 * 1e4

    title = r'$E_\chi=${0}MeV, $m_\chi=${1}MeV, $\epsilon=$ {2}'.format(e_chi, m_chi, latex_float(eps, decimal=1))
    plt.figure(figsize=(8, 6))
    plt.plot(deex, signals, linewidth=2)
    plt.ylabel(r'Inelastic cross section [cm$^2$]', fontsize=18)
    plt.xlabel('Deexcitation energy [MeV]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, fontsize=22, loc='right')
    plt.tight_layout()
    plt.show()


def plot_conv_DM_rate(e_chi, m_chi, eps=1e-4, mass_ratio=3):
    deex = np.arange(0, 20, 0.01)
    m_med = mass_ratio * m_chi
    signals = [sigmaGTsum(e_chi, m_chi, m_med, eps, mAs, Jis, nucl_exes, de, particle='fermion') for de in deex]
    signals = np.array(signals) * meter_by_mev**2  # m^2

    flux = 1e2 # [m^-2 s^-1]
    signals *= flux
    signals *= det_mass * atoms # how many atoms
    signals *= 365*24*3600 # s^-1 to yr^-1

    print('sum', np.sum(signals))

    title = r'$E_\chi=$ {0}MeV, $m_\chi=$ {1}MeV, $\epsilon=$ {2}, $\phi=$ {3}$m^{{-2}} s^{{-1}}$'.format(e_chi, m_chi, latex_float(eps), latex_float(flux))
    plt.figure(figsize=(8, 6))
    plt.plot(deex, signals, label=title, linewidth=2)
    plt.ylabel(r'DM rate [yr$^{-1}$]', fontsize=18)
    plt.xlabel('Deexcitation energy [MeV]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, fontsize=22, loc='right')
    plt.tight_layout()
    plt.show()


def convert_NA64(particle):
    """
    Converts the new NA64 limits to compatible format
    It has columns: A' mass [MeV], epsilon and alpha_D=0.1
    """
    # alpha_D = 0.1
    alpha_D = 0.5
    if particle == 'fermion':
        file = np.genfromtxt('limits/NA64 Celentano/results_MvsEps_VFF_alphaD0p1.dat')
    elif particle == 'scalar':
        file = np.genfromtxt('limits/NA64 Celentano/results_MvsEps_VSS_alphaD0p1.dat')

    file[:, 0] /= 3
    file[:, 1] = file[:, 1]**2 * alpha_D * 3**(-4)
    return file


def plot_calculated(files, sort=False, x_axis='chi', y_axis='Y', mass_ratio=3):
    """
    Plot the calculated sensitivities
    files: [file], where file=[path, label, mass_ratio:optional]
    sort: sort the file by mass or not
    """
    for idx, file in enumerate(files):
        path = file[0]
        label = file[1]
        mass_ratio = file[2] if len(file) > 2 else 3
        if sort:
            sort_file(path)
        limits = np.genfromtxt(path)
        limits = limits[limits[:, 1] > 0]  # remove the zero lower limit
        mass_chi, eps = cleanLimitData(limits)
        tab = color_tabs[idx]
        color = colors[tab]

        splot = '-' if label in ['KARMEN'] else '--'
        plt.plot(x_convert(mass_chi, 'chi', x_axis, mass_ratio), y_convert(eps, 'eps', y_axis, mass_ratio), splot, color=color, label=label)
        # plt.plot(x_convert(mass_chi, 'chi', x_axis, mass_ratio), y_convert(eps, 'eps', y_axis, mass_ratio), '--', color=color, label=label)


def existing_constraints(model, x_axis='chi', y_axis='Y'):
    """
    model: 'fermion', 'scalar', 'CCM leptophobic', 'COHERENT leptophobic fermion', 'COHERENT leptophobic scalar'
    """

    if model == 'CCM leptophobic':
        # m_chi [MeV], alpha_B
        mass_ratio = 2.1
        ccm120 = np.genfromtxt('limits/2109.14146/CCM120.csv', delimiter=',')
        ccm120_s = np.genfromtxt('limits/2109.14146/CCM200 (CCM120 shielding).csv', delimiter=',')
        ccm200_s = np.genfromtxt('limits/2109.14146/CCM200 (proposed shielding).csv', delimiter=',')
        ccm200_u = np.genfromtxt('limits/2109.14146/CCM200 (underground ar).csv', delimiter=',')

        plt.plot(x_convert(ccm120[:, 0], 'chi', x_axis, mass_ratio), y_convert(ccm120[:, 1], 'alpha_B', y_axis, mass_ratio), color='yellowgreen', label='CCM120')
        plt.plot(x_convert(ccm120_s[:, 0], 'chi', x_axis, mass_ratio), y_convert(ccm120_s[:, 1], 'alpha_B', y_axis, mass_ratio), color='steelblue', label='CCM200 (CCM120 shielding)')
        plt.plot(x_convert(ccm200_s[:, 0], 'chi', x_axis, mass_ratio), y_convert(ccm200_s[:, 1], 'alpha_B', y_axis, mass_ratio), color='gold', label='CCM200 (proposed shielding)')
        plt.plot(x_convert(ccm200_u[:, 0], 'chi', x_axis, mass_ratio), y_convert(ccm200_u[:, 1], 'alpha_B', y_axis, mass_ratio), color='orchid', label='CCM200 (underground ar)')
        return
    elif model in ['COHERENT leptophobic fermion', 'COHERENT leptophobic scalar']:
        # m_A' [MeV], alpha_B
        mass_ratio = 2
        ccm = np.genfromtxt('limits/2205.12414/ccm.csv', delimiter=',')
        nPb = np.genfromtxt('limits/2205.12414/n-Pb scattering.csv', delimiter=',')
        anomalon = np.genfromtxt('limits/2205.12414/anomalon.csv', delimiter=',')
        miniboone = np.genfromtxt('limits/2205.12414/miniboone.csv', delimiter=',')
        na62 = np.genfromtxt('limits/2205.12414/NA62.csv', delimiter=',')
        csi = np.genfromtxt('limits/2205.12414/COHERENT CsI.csv', delimiter=',')
        s_1 = np.genfromtxt('limits/2205.12414/scalar eps=1e-1.csv', delimiter=',')
        s_3 = np.genfromtxt('limits/2205.12414/scalar eps=1e-3.csv', delimiter=',')
        s_5 = np.genfromtxt('limits/2205.12414/scalar eps=1e-5.csv', delimiter=',')
        f_2 = np.genfromtxt('limits/2205.12414/fermion eps=1e-2.csv', delimiter=',')
        f_4 = np.genfromtxt('limits/2205.12414/fermion eps=1e-4.csv', delimiter=',')
        f_6 = np.genfromtxt('limits/2205.12414/fermion eps=1e-6.csv', delimiter=',')

        # plt.plot(x_convert(ccm[:, 0], 'A', x_axis, mass_ratio), y_convert(ccm[:, 1], 'alpha_B', y_axis, mass_ratio), color='deepskyblue', label='CCM120')
        # plt.plot(x_convert(nPb[:, 0], 'A', x_axis, mass_ratio), y_convert(nPb[:, 1], 'alpha_B', y_axis, mass_ratio), color='deeppink', label='n-Pb scattering')
        # plt.plot(x_convert(anomalon[:, 0], 'A', x_axis, mass_ratio), y_convert(anomalon[:, 1], 'alpha_B', y_axis, mass_ratio), color='forestgreen', label='Anomalon')
        # plt.plot(x_convert(miniboone[:, 0], 'A', x_axis, mass_ratio), y_convert(miniboone[:, 1], 'alpha_B', y_axis, mass_ratio), color='orangered', label='MiniBooNE')
        # plt.plot(x_convert(na62[:, 0], 'A', x_axis, mass_ratio), y_convert(na62[:, 1], 'alpha_B', y_axis, mass_ratio), color='sienna', label='NA62')
        plt.plot(x_convert(csi[:, 0], 'A', x_axis, mass_ratio), y_convert(csi[:, 1], 'alpha_B', y_axis, mass_ratio), color='slateblue', label='COHERENT CsI')
        plt.fill_between(x_convert(csi[:, 0], 'A', x_axis, mass_ratio), y_convert(csi[:, 1], 'alpha_B', y_axis, mass_ratio), y_convert(1e-6, 'Y', y_axis, mass_ratio), color='#bbd5f2', alpha=0.3, label='Existing limits')

        return
        particle = model.split()[-1]
        if particle == 'scalar':
            plt.plot(x_convert(s_1[:, 0], 'A', x_axis, mass_ratio), y_convert(s_1[:, 1], 'alpha_B', y_axis, mass_ratio), '-.', color='black', label='scalar eps=1e-1')
            plt.plot(x_convert(s_3[:, 0], 'A', x_axis, mass_ratio), y_convert(s_3[:, 1], 'alpha_B', y_axis, mass_ratio), ':', color='black', label='scalar eps=1e-3')
            plt.plot(x_convert(s_5[:, 0], 'A', x_axis, mass_ratio), y_convert(s_5[:, 1], 'alpha_B', y_axis, mass_ratio), color='black', label='scalar eps=1e-5')
        elif particle == 'fermion':
            plt.plot(x_convert(f_2[:, 0], 'A', x_axis, mass_ratio), y_convert(f_2[:, 1], 'alpha_B', y_axis, mass_ratio), '-.', color='black', label='fermion eps=1e-2')
            plt.plot(x_convert(f_4[:, 0], 'A', x_axis, mass_ratio), y_convert(f_4[:, 1], 'alpha_B', y_axis, mass_ratio), ':', color='black', label='fermion eps=1e-4')
            plt.plot(x_convert(f_6[:, 0], 'A', x_axis, mass_ratio), y_convert(f_6[:, 1], 'alpha_B', y_axis, mass_ratio), color='black', label='fermion eps=1e-6')

    elif model == 'PIP2BD':
        # m_chi [GeV], Y. fig9 from 2203.08079
        par = np.genfromtxt('limits/2203.08079/PAR.csv', delimiter=',')
        c_par = np.genfromtxt('limits/2203.08079/C-PAR.csv', delimiter=',')
        rcs_sr = np.genfromtxt('limits/2203.08079/RCS-SR.csv', delimiter=',')

        # plt.plot(1e3*x_convert(par[:, 0], 'chi', x_axis), y_convert(par[:, 1], 'Y', y_axis), color='royalblue', label='PIP:PAR (elastic)')
        # plt.plot(1e3*x_convert(c_par[:, 0], 'chi', x_axis), y_convert(c_par[:, 1], 'Y', y_axis), color='orangered', label='PIP:C-PAR (elastic)')
        plt.plot(1e3*x_convert(rcs_sr[:, 0], 'chi', x_axis), y_convert(rcs_sr[:, 1], 'Y', y_axis), color='forestgreen', label='PIP:RCS-SR (elastic)')
        return

    # m_chi [MeV], Y
    ccm = interp(np.genfromtxt('limits/2110.11453/CCM.csv', delimiter=','))
    na64 = interp(np.genfromtxt('limits/2110.11453/NA64.csv', delimiter=','))
    miniboone_e = interp(np.genfromtxt('limits/2110.11453/miniboone e.csv', delimiter=','))
    miniboone_N = interp(np.genfromtxt('limits/2110.11453/miniboone N.csv', delimiter=','))
    e137 = interp(np.genfromtxt('limits/2110.11453/E137.csv', delimiter=','))
    lsnd = interp(np.genfromtxt('limits/2110.11453/LSND.csv', delimiter=','))
    coherent_csi = interp(np.genfromtxt('limits/2110.11453/purple solid.csv', delimiter=','))
    pseudo_dirac = np.genfromtxt('limits/2110.11453/pseudo-dirac.csv', delimiter=',')
    scalar_relic = np.genfromtxt('limits/2110.11453/scalar.csv', delimiter=',')
    # plt.plot(mass_chi_array, na64_2021, color='brown', label='NA64 (2021)')
    # plt.plot(mass_chi_array, na64, color='green', label='NA64')
    # plt.plot(mass_chi_array, miniboone_e, color='orangered', label='MiNiBooNE e/N')
    # plt.plot(mass_chi_array, miniboone_N, color='orangered')
    # plt.plot(mass_chi_array, e137, color='brown', label='E137')
    # plt.plot(mass_chi_array, lsnd, color='royalblue', label='LSND')

    mass_chi_array = np.arange(1, 300, 0.1)
    na64_2021 = interp(convert_NA64(model))
    min_sensitivity = minimize_arrays(ccm, na64, miniboone_e, miniboone_N, e137, lsnd, coherent_csi, na64_2021)
    # plt.plot(x_convert(mass_chi_array, 'chi', x_axis), y_convert(ccm, 'Y', y_axis), color='cyan', label='CCM (elastic)')
    plt.plot(x_convert(mass_chi_array, 'chi', x_axis), y_convert(coherent_csi, 'Y', y_axis), color='purple', label='COHERENT CsI (elastic)')
    plt.fill_between(x_convert(mass_chi_array, 'chi', x_axis), y_convert(min_sensitivity, 'Y', y_axis), y_convert(1e-6, 'Y', y_axis), color='#bbd5f2', alpha=0.3, label='Existing limits')

    if model == 'fermion':
        pseudo_dirac = interp(pseudo_dirac)
        plt.plot(x_convert(mass_chi_array, 'chi', x_axis), y_convert(pseudo_dirac, 'Y', y_axis), color='grey', label='Relic (pseudo Dirac)')
    elif model == 'scalar':
        scalar_relic = interp(scalar_relic)
        plt.plot(x_convert(mass_chi_array, 'chi', x_axis), y_convert(scalar_relic, 'Y', y_axis), color='grey', label='Scalar Relic')


def sensitivities_plots(files, sort=False, x_axis='chi', y_axis='Y', particle='fermion', save=False):
    """
    x_axis: 'chi', 'A'
    y_axis: 'Y', 'eps', 'alpha_B', 'gB'
    """
    plot_calculated(files, sort, x_axis, y_axis)
    if x_axis == 'chi':
        xlabel = r'$m_\chi$ [MeV]'
        xlims = (1, 300)
    elif x_axis == 'A':
        xlabel = r'$m_{A^\prime}$ [MeV]'
        xlims = (3, 900)

    if y_axis == 'Y':
        ylabel = r'$Y=\epsilon^2 \alpha_D  \left( \frac{m_\chi}{m_{A^\prime}} \right)^4 $'
        ylims = (2e-14, 1e-9)
    elif y_axis == 'eps':
        ylabel = r'$\epsilon$'
        ylims = (1e-6, 1e-2)
    elif y_axis == 'alpha_B':
        ylabel = r'$\alpha_B$'
        ylims = (5e-15, 2e-7)
    elif y_axis == 'gB':
        ylabel = r'$g_B = \epsilon e$'
        ylims = (1e-6, 2e-3)

    if particle == 'fermion':
        existing_constraints('fermion', x_axis, y_axis)
        # existing_constraints('COHERENT leptophobic fermion', x_axis, y_axis)
        title = "Fermionic DM"
    elif particle == 'scalar':
        existing_constraints('scalar', x_axis, y_axis)
        # existing_constraints('COHERENT leptophobic scalar', x_axis, y_axis)
        title = "Scalar DM"

    # existing_constraints('PIP2BD', x_axis, y_axis)
    # existing_constraints('CCM leptophobic', x_axis, y_axis)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.ylabel(ylabel, fontsize=24)
    plt.xlabel(xlabel, fontsize=24)
    plt.tick_params(axis='x', which='minor')
    plt.title(title, loc="right")
    plt.tight_layout()
    plt.legend(loc="lower right", framealpha=0.5, fontsize=8)

    # plt.annotate('COHERENT CsI', xy=(1.5, 3e-11), fontsize=12, color='purple')
    # plt.annotate('PIP2 el', xy=(1.5, 1.5e-12), fontsize=12, color='green')
    # plt.annotate('Ge', xy=(1.5, 4e-13), fontsize=12, color=colors[color_tabs[1]])
    # plt.annotate('CCM CsI', xy=(1, 1e-13), fontsize=12, color=colors[color_tabs[0]])
    # plt.annotate('PIP2 inel', xy=(6, 1e-13), fontsize=12, color='green')
    # plt.annotate('Relic', xy=(100, 4e-11), fontsize=12)

    if save:
        plt.savefig(f'{home}/sen_{particle}.pdf')
    plt.show()


def plot_pion():
    pion_flux = get_pion_brem_flux('CCM120')[0]
    pion_energy_flux_shielded = pion_flux.transpose()[2]

    pion_flux = get_pion_brem_flux('PIP2BD')[0]
    pion_energy_flux = pion_flux.transpose()[2]

    plt.figure(figsize=(8, 6))
    plt.hist(pion_energy_flux_shielded, bins=100, label='shielded', density=True, histtype='step')
    plt.hist(pion_energy_flux, bins=100, label='present', density=True, histtype='step')
    plt.xlabel('Pion kinetic energy [MeV]', fontsize=20)
    plt.ylabel('Probability density', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=18)
    plt.show()


def ccm_sbnd_comparison():
    # mass, total POT, distance, pion rate
    ccm = [7e3, 3*1.76e22, 23, 0.0633]
    sbnd = [112e3, 6.6e20, 110, 1]

    ccm_total = ccm[0] * ccm[1] / ccm[2]**2 * ccm[3]
    sbnd_total = sbnd[0] * sbnd[1] / sbnd[2]**2 * sbnd[3]
    ratio = sbnd_total / ccm_total
    print('Signal ratio SBND / CCM', ratio)
    print('Y ratio SBND / CCM', np.sqrt(ratio))


if __name__ == "__main__":
    # plot_pion()
    # plot_det_bkg()
    # plot_total_bkg()
    # plot_total_bkg_pip2bd()
    # strength_plot('c12', 'c12', color_idx=0, conv_width=1)
    # strength_plots(['c12'], conv_width=0)
    # timing_flux_plot(10, 1e-4)
    # energy_flux_plot(2, 1e-4)
    # plot_conv_DM_xsec(e_chi=75, m_chi=30)
    # plot_DM_xsec(m_chi=30)
    # plot_conv_DM_rate(e_chi=75, m_chi=30)

    fermions = [
        ['out/DM/ccm200_f_14e.txt', 'CCM'],
        # ['out/DM/nai_f_20e.txt', 'NaI'],
        ['out/DM/karmen_f_3.txt', 'KARMEN'],
        ['out/DM/pip2bd_f_nu.txt', 'PIP2BD'],
        ['out/DM/sbnd_f.txt', 'SBND (q $<$ 200MeV)'],
        ['out/DM/sbnd_f_noq.txt', 'SBND (free q)'],
    ]

    scalars = [
        ['out/DM/ccm200_s_14e.txt', 'CCM'],
        # ['out/DM/nai_s_20e.txt', 'NaI'],
        ['out/DM/karmen_s_3.txt', 'KARMEN'],
        ['out/DM/pip2bd_s_nu.txt', 'PIP2BD'],
        ['out/DM/sbnd_s.txt', 'SBND (q $<$ 200MeV)'],
        ['out/DM/sbnd_s_noq.txt', 'SBND (free q)'],
    ]

    sort = False
    # ccm_sbnd_comparison()
    # sensitivities_plots(fermions, particle='fermion', x_axis='chi', y_axis='Y', sort=sort)
    # sensitivities_plots(scalars, particle='scalar', x_axis='chi', y_axis='Y', sort=sort)

    # sensitivities_plots(fermions, particle='fermion', x_axis='A', y_axis='alpha_B')
    # sensitivities_plots(scalars, particle='scalar', x_axis='A', y_axis='alpha_B')
