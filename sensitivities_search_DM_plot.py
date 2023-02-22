import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyCEvNS.events import *
from pyCEvNS.flux import *
from pyCEvNS.constants import *
from util import *
from scipy import interpolate
from inel_nu import get_nu_signals
from exp_config import *
from DM_xsec import sigmaGTsum
# from pip3body import pip3_DM_events


strength_rescale = get_strength_scale() # 0.16212

colors = mcolors.TABLEAU_COLORS
color_tabs = list(colors.keys())
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)


# COHERENT_CsI, COHERENT_NaI, CCM
EXPERIMENT = 'CCM'
mAs, Jis, Zs, nucl_exes = get_mA_Ji(EXPERIMENT)
prefactor, pot_rate_per_day, pim_rate, pion_rate, dist, secs, det_mass, atoms, pot_mu, pot_sigma = get_rates(EXPERIMENT)
pion_flux, brem_photons = get_pion_brem_flux(EXPERIMENT)
bkg, energy_edges = get_bkg(EXPERIMENT)


def plot_det_bkg():
    plt.figure(figsize=(8, 6))
    bkg, energy_edges = get_bkg('CCM', energy_cut=False)
    plt.bar(x=energy_edges[:-1], height=bkg, width=np.diff(energy_edges), align='edge', fc='grey', alpha=.25, label='CCM')
    print('CCM bkg sum', np.sum(bkg))

    bkg, energy_edges = get_bkg('COHERENT_NaI', energy_cut=False)
    plt.bar(x=energy_edges[:-1], height=bkg, width=np.diff(energy_edges), align='edge', fc='cyan', alpha=.25, label='COHERENT NaI')
    print('coherent bkg sum', np.sum(bkg))

    plt.yscale('log')
    plt.ylabel(r'Count', fontsize=18)
    plt.xlabel('Energy [MeV]', fontsize=18)
    plt.xlim(0, 200)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Background energy distribution', loc="right", fontsize=20)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# detector bkg + inelastic nu bkg
def plot_total_bkg():
    plt.figure(figsize=(8, 6))

    # Detector bkg
    bkg, energy_edges = get_bkg('CCM')
    energy_edges[0] = 9 # move cut to 9 MeV
    bkg_bin_width = (energy_edges[-1] - energy_edges[0]) / len(energy_edges)
    bkg_bin_rescale = 1 / bkg_bin_width / 0.1 # change the unit to per 0.1MeV
    plt.bar(x=energy_edges[:-1], height=bkg_bin_rescale*bkg, width=np.diff(energy_edges), align='edge', fc='blue', alpha=.25, label='CCM')

    bkg, energy_edges = get_bkg('COHERENT_NaI')
    energy_edges[0] = 4 # move cut to 4 MeV
    bkg_bin_width = (energy_edges[-1] - energy_edges[0]) / len(energy_edges)
    bkg_bin_rescale = 1 / bkg_bin_width / 0.1 # change the unit to per 0.1MeV
    plt.bar(x=energy_edges[:-1], height=bkg_bin_rescale*bkg, width=np.diff(energy_edges), align='edge', fc='orangered', alpha=.25, label='NaI')

    # Inelastic nu bkg (prompt only)
    energy_bins = np.linspace(0, 100, 1000)
    bin_width = 0.1
    timing_cut = True
    signals_ccm = strength_rescale*get_nu_signals('CCM', energy_bins=energy_bins, timing_cut=timing_cut, prompt=True) / bin_width
    signals_nai = strength_rescale*get_nu_signals('COHERENT_NaI', energy_bins=energy_bins, timing_cut=timing_cut, prompt=True) / bin_width

    plt.plot(energy_bins, signals_ccm, color='blue', label=r'CCM $\nu_\mu$')
    plt.plot(energy_bins, signals_nai, color='orangered', label=r'NaI $\nu_\mu$')

    # plt.yscale('log')
    plt.xlabel('Energy [MeV]', fontsize=18)
    plt.ylabel('Events / 0.1 MeV', fontsize=18)
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
    timing_cut = True
    signals_pip2bd = strength_rescale*get_nu_signals('PIP2BD', energy_bins=energy_bins, timing_cut=timing_cut, prompt=True) / bin_width
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


def energy_flux_plot(m_chi, epsilon):
    plt.figure(figsize=(8, 6))
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
    # title += r'$m_\chi=10' +' MeV,' + r'\epsilon=10^{-4}$'
    plt.yscale('log')
    plt.ylabel(r'Flux [m$^{-2}$ s$^{-1}$ MeV$^{-1}$]', fontsize=18)
    plt.xlabel('DM energy [MeV]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, loc="right", fontsize=20)
    plt.legend(fontsize=16, loc='upper right')
    plt.tight_layout()
    plt.show()


def timing_flux_plot(m_chi, epsilon):
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
    plt.ylabel(r'Flux [$m^{-2} s^{-1}$]', fontsize=18)
    plt.xlabel('t [mus]', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim((-0.25, 2))
    plt.title('DM flux timing distribution', loc="right", fontsize=20)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


# nucleus = 'na23', 'ge76', 'ar40', 'cs133', 'I127', 'o16'
def strength_plot(nucleus, label, color_idx=0, convoluted=False):
    nucl_ex_path = f'bigstick/{nucleus}_s.res'
    nucl_ex = np.genfromtxt(nucl_ex_path, delimiter='   ')
    nucl_ex[:, 0] = nucl_ex[:, 0] - nucl_ex[0, 0]
    nucl_ex = nucl_ex[np.where(nucl_ex[:, 1] != 0)]

    bigstick_energy = nucl_ex[:, 0]
    bigstick_strength = nucl_ex[:, 1]

    if convoluted:
        width_keV = 150
        # width_keV = 1e3
        def totals(x):
            gau = 0
            for i in range(len(bigstick_energy)):
                gau += gaussian(x, bigstick_energy[i], width_keV*1e-3) * bigstick_strength[i]
            return gau

        # xx = np.arange(8, 50, 0.01)   # for o16
        xx = np.arange(0, 20, 0.01)   # for others
        yy = np.array([totals(x) for x in xx])
        plt.plot(xx, yy, color=colors[color_tabs[color_idx]], label=label)
    else:
        for x,y in zip(bigstick_energy, bigstick_strength):
            markerline, stemlines, baseline = plt.stem(x, y, linefmt='-')
            plt.setp(stemlines, linewidth=1)
            plt.setp(markerline, markersize=4, color=colors[color_tabs[0]])
        plt.setp(stemlines, linewidth=1, label=label)
        plt.yscale('log')

    plt.title(f'BIGSTICK Gamow-Teller strength', fontsize=20, loc='right')
    plt.legend(fontsize=16)
    plt.ylabel('GT Strength', fontsize=18)
    plt.xlabel('Deexcitation energy [MeV]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()


def strengths_plot(nuclei):
    """
    nuclei = [nucleus]
    """
    plt.figure(figsize=(8, 6))
    for i, nucleus in enumerate(nuclei):
        label = nucleus.capitalize()
        strength_plot(nucleus, label, color_idx=i, convoluted=True)

    plt.show()


def plot_conv_DM_xsec(e_chi, m_chi, eps=1e-4):
    deex = np.arange(0, 20, 0.01)
    signals = [sigmaGTsum(e_chi, m_chi, eps, mAs, Jis, nucl_exes, de) for de in deex]
    signals = np.array(signals) * meter_by_mev**2 * 1e4

    title = r'$E_{DM}=$'+f'{e_chi}MeV\n'+r'$m_{DM}=$'+f'{m_chi}MeV, '+r'$\epsilon=10^{-4}$'
    plt.plot(deex, signals, label=title, linewidth=2)
    plt.ylabel(r'Inelastic cross section [$cm^2$]', fontsize=18)
    plt.xlabel('Deexcitation energy [MeV]', fontsize=18)
    plt.legend(fontsize=16, loc='upper left')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title(title, fontsize=22, loc='right')
    plt.tight_layout()
    plt.show()


def plot_conv_DM_rate(e_chi, m_chi, eps=1e-4):
    deex = np.arange(0, 20, 0.01)
    signals = [sigmaGTsum(e_chi, m_chi, eps, mAs, Jis, nucl_exes, de) for de in deex]
    signals = np.array(signals) * meter_by_mev**2  # m^2

    signals *= 1e2  # flux [m^-2 s^-1]
    signals *= det_mass * atoms # how many atoms
    signals *= 365*24*3600 # s^-1 to yr^-1

    print('sum', np.sum(signals))

    title = r'$E_{DM}=$'+f'{e_chi}MeV,'+r'$m_{DM}=$'+f'{m_chi}MeV, '+r'$\epsilon=10^{-4}$'
    title += '\nFlux=' + r'$10^2 m^{-2} s^{-1}$'
    plt.plot(deex, signals, label=title, linewidth=2)
    plt.ylabel(r'DM rate [$yr^{-1}$]', fontsize=18)
    plt.xlabel('Deexcitation energy [MeV]', fontsize=18)
    # plt.legend(fontsize=16, loc='upper left')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title, fontsize=22, loc='right')
    plt.tight_layout()
    plt.show()


# interpolate and extrapolate
def interp(limits):
    f = interpolate.interp1d(limits[:, 0], limits[:, 1], fill_value='extrapolate')
    xx = np.arange(1, 300, 0.1)
    return f(xx)


def minimize_arrays(*arrs):
    min_arr = np.array([np.inf]* len(arrs[0]))
    for arr in arrs:
        min_arr = np.minimum(min_arr, arr)
    return min_arr


# files = [file]; file = [path, label]
def sensitivity_plot_fermion(files, sort=False):
    for idx, file in enumerate(files):
        path = file[0]
        label = file[1]
        if sort:
            sort_file(path)
        limits = np.genfromtxt(path)

        # remove the zero lower limit
        limits = limits[limits[:, 1] > 0]
        mass_chi, eps = cleanLimitData(limits)
        tab = color_tabs[idx]
        color = colors[tab]

        yy = eps**2 * alpha_D * 3**(-4)

        plt.plot(mass_chi, yy, '--', color=color, label=label)


    # read limits from paper
    ccm = interp(np.genfromtxt('limits/CCM.csv', delimiter=','))
    na64 = interp(np.genfromtxt('limits/NA64.csv', delimiter=','))
    miniboone_e = interp(np.genfromtxt('limits/miniboone e.csv', delimiter=','))
    miniboone_N = interp(np.genfromtxt('limits/miniboone N.csv', delimiter=','))
    e137 = interp(np.genfromtxt('limits/E137.csv', delimiter=','))
    lsnd = interp(np.genfromtxt('limits/LSND.csv', delimiter=','))
    coherent_csi = interp(np.genfromtxt('limits/purple solid.csv', delimiter=','))
    pseudo_dirac = interp(np.genfromtxt('limits/pseudo-dirac.csv', delimiter=','))

    min_sensitivity = minimize_arrays(ccm, na64, miniboone_e, miniboone_N, e137, lsnd, coherent_csi)
    mass_chi_array = np.arange(1, 300, 0.1)

    plt.plot(mass_chi_array, ccm, color='cyan', label='CCM (elastic)')
    plt.plot(mass_chi_array, coherent_csi, color='purple', label='COHERENT CsI (elastic)')
    plt.plot(mass_chi_array, pseudo_dirac, color='grey', label='Relic (pseudo Dirac)')
    # plt.plot(mass_chi_array, na64, color='green', label='NA64')
    # plt.plot(mass_chi_array, miniboone_e, color='orangered', label='MiNiBooNE e/N')
    # plt.plot(mass_chi_array, miniboone_N, color='orangered')
    # plt.plot(mass_chi_array, e137, color='brown', label='E137')
    # plt.plot(mass_chi_array, lsnd, color='royalblue', label='LSND')
    plt.fill_between(mass_chi_array, min_sensitivity, 1e-8, color='#bbd5f2', alpha=0.3, label='Existing limits')

    plt.legend(loc="lower right", framealpha=1, fontsize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(1,300)
    plt.ylim(2e-14,7e-9)
    plt.ylabel(r'$Y=\epsilon^2 \alpha_D  \left( \frac{m_\chi}{m_{A^\prime}} \right)^4 $', fontsize=24)
    plt.xlabel(r'$m_\chi$ [MeV]', fontsize=24)
    # plt.title("Fermionic DM search at 90\% CL via nucleus scattering", loc="right")
    plt.title("Fermionic DM", loc="right")
    plt.tick_params(axis='x', which='minor')
    plt.tight_layout()
    plt.show()


def sensitivity_plot_scalar(files, sort=False):
    for idx, file in enumerate(files):
        path = file[0]
        label = file[1]
        if sort:
            sort_file(path)
        limits = np.genfromtxt(path)

        # remove the zero lower limit
        limits = limits[limits[:, 1] > 0]
        mass_chi, eps = cleanLimitData(limits)
        tab = color_tabs[idx]
        color = colors[tab]

        yy = eps**2 * alpha_D * 3**(-4)
        plt.plot(mass_chi, yy, '--', color=color, label=label)

    # read limits from paper
    scalar = interp(np.genfromtxt('limits/scalar.csv', delimiter=','))
    ccm = interp(np.genfromtxt('limits/CCM.csv', delimiter=','))
    na64 = interp(np.genfromtxt('limits/NA64.csv', delimiter=','))
    miniboone_e = interp(np.genfromtxt('limits/miniboone e.csv', delimiter=','))
    miniboone_N = interp(np.genfromtxt('limits/miniboone N.csv', delimiter=','))
    e137 = interp(np.genfromtxt('limits/E137.csv', delimiter=','))
    lsnd = interp(np.genfromtxt('limits/LSND.csv', delimiter=','))
    coherent_csi = interp(np.genfromtxt('limits/purple solid.csv', delimiter=','))

    min_sensitivity = minimize_arrays(ccm, na64, miniboone_e, miniboone_N, e137, lsnd, coherent_csi)
    mass_chi_array = np.arange(1, 300, 0.1)

    plt.plot(mass_chi_array, ccm, color='cyan', label='CCM (elastic)')
    plt.plot(mass_chi_array, coherent_csi, color='purple', label='COHERENT CsI (elastic)')
    plt.plot(mass_chi_array, scalar, color='grey', label='Scalar Relic')
    # plt.plot(mass_chi_array, na64, color='green', label='NA64')
    # plt.plot(mass_chi_array, miniboone_e, color='orangered', label='MiNiBooNE e/N')
    # plt.plot(mass_chi_array, miniboone_N, color='orangered')
    # plt.plot(mass_chi_array, e137, color='brown', label='E137')
    # plt.plot(mass_chi_array, lsnd, color='royalblue', label='LSND')
    plt.fill_between(mass_chi_array, min_sensitivity, 1e-8, color='#bbd5f2', alpha=0.3, label='Existing limits')

    plt.legend(loc="lower right", framealpha=1, fontsize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(1,300)
    plt.ylim(2e-14,7e-9)
    plt.ylabel(r'$Y=\epsilon^2 \alpha_D  \left( \frac{m_\chi}{m_{A^\prime}} \right)^4 $', fontsize=24)
    plt.xlabel(r'$m_\chi$ [MeV]', fontsize=24)
    # plt.title("Scalar DM search at 90\% CL via nucleus scattering", loc="right")
    plt.title("Scalar DM", loc="right")
    plt.tick_params(axis='x', which='minor')
    plt.tight_layout()
    plt.show()


def plot_pion():
    pion_flux = get_pion_brem_flux('CCM')[0]
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


if __name__ == "__main__":
    # plot_pion()
    # plot_det_bkg()
    plot_total_bkg()
    # plot_total_bkg_pip2bd()
    # strength_plot('ar40', 'ar40', color_idx=0, convoluted=True)
    # strengths_plot(['ar40', 'na23', 'I127'])
    # timing_flux_plot(10, 1e-4)
    # energy_flux_plot(10, 1e-4)
    # plot_conv_DM_xsec(e_chi=75, m_chi=30)
    # plot_conv_DM_rate(e_chi=75, m_chi=30)

    fermions = [
        ['out/eta_fixed/nai_f_bkg=400.txt', 'NaI'],
        ['out/eta_fixed/ccm_f.txt', 'CCM'],
    ]
    fermions = [
        ['out/eta_fixed/rescale/nai_f_3yr.txt', 'NaI'],
        ['out/eta_fixed/rescale/ccm_f.txt', 'CCM'],
        ['out/eta_fixed/rescale/ccm_f_100t.txt', 'CCM (100t)'],
        ['out/eta_fixed/rescale/pip2bd_f.txt', 'PIP2BD'],
        ['out/eta_fixed/rescale/pip2bd_f_0bkg.txt', 'PIP2BD (bkg free)'],
    ]

    # sensitivity_plot_fermion(fermions)


    scalars = [
        ['out/eta_fixed/nai_s_bkg=400.txt', 'NaI'],
        ['out/eta_fixed/ccm_s.txt', 'CCM'],
    ]
    scalars = [
        ['out/eta_fixed/rescale/nai_s_3yr.txt', 'NaI'],
        ['out/eta_fixed/rescale/ccm_s.txt', 'CCM'],
        ['out/eta_fixed/rescale/ccm_s_100t.txt', 'CCM (100t)'],
        ['out/eta_fixed/rescale/pip2bd_s.txt', 'PIP2BD'],
        ['out/eta_fixed/rescale/pip2bd_s_0bkg.txt', 'PIP2BD (bkg free)'],
    ]

    # sensitivity_plot_scalar(scalars)