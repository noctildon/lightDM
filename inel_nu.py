"""
All in MeV scale
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from exp_config import *
from util import *
from scipy.integrate import quad
from pyCEvNS.constants import *

colors = mcolors.TABLEAU_COLORS
color_tabs = list(colors.keys())
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)


m_pi = 139.0   # pion mass [MeV]
m_mu = 105.6   # muon mass [MeV]
e_prompt = (m_pi**2 - m_mu**2) / (2*m_pi)


# delayed mu neutrino energy distribution
def dPdE_mu(enu):
    if 0 < enu < m_mu/2:
        return 64 * enu**2 * (3/4 - enu/m_mu) / m_mu**3
    return 0


# delayed e neutrino energy distribution
def dPdE_e(enu):
    if 0 < enu < m_mu/2:
        return 192 * enu**2 * (1/2 - enu/m_mu) / m_mu**3
    return 0


def dPdE_check():
    mu_sum = quad(lambda x: dPdE_mu(x), 0, m_mu/2)[0]
    e_sum = quad(lambda x: dPdE_e(x), 0, m_mu/2)[0]
    print("mu sum:", mu_sum)
    print("e sum:", e_sum)


def dPdE_plot():
    enu = np.linspace(0, m_mu/2, 100)
    plt.plot(enu, [dPdE_mu(e) for e in enu], label="mu")
    plt.plot(enu, [dPdE_e(e) for e in enu], label="e")
    plt.xlabel("Neutrino energy [MeV]")
    plt.ylabel("dP/dE")
    plt.legend()
    plt.show()


def xsec_GT(enu, dE, er, Ji, bgt):
    """
    enu: incoming neutrino energy
    dE: excitation energy
    er: recoil energy
    Ji: nucleus spin
    bgt: GT strength
    return: inelastic nu cross section
    """
    if enu < dE + er:
        return 0

    res = 2*Gf**2 * gA**2 / (np.pi * (2*Ji+1))
    res *= (enu - dE - er)**2 * bgt
    return res


def xsec_GT_sum(enu, dE, Jis, nucl_exes, width_keV=150):
    """
    enu: incoming neutrino energy
    dE: excitation energy = gamma energy = signal energy
    Jis: array of nuclear spins
    nucl_exes: array of nuclear excitation energies (should have same length as mA)
    width_keV: width of gaussian convolution
    return: inelastic nu cross section summed over all nuclei and lines
    """
    if len(Jis) != len(nucl_exes):
        raise ValueError("Jis and nucl_exes must have same length")

    er = 0 # ignorable recoil energy
    if enu <= dE + er:
        return 0
    s = 0

    for Ji, nucl_ex in zip(Jis, nucl_exes):
        gt_strength = conv_strength(dE, nucl_ex, width_keV=width_keV)
        s += xsec_GT(enu, dE, er, Ji, gt_strength)
    return s


def plot_conv_nu_xsec(enu, experiment):
    mAs, Jis, Zs, nucl_exes = get_mA_Ji(experiment)
    deex = np.arange(0, 20, 0.01)
    signals = [xsec_GT_sum(enu, de, Jis, nucl_exes) for de in deex]
    signals = np.array(signals) * meter_by_mev**2 * 1e4

    title = r'$E_\nu=$' + f'{enu}MeV'
    plt.plot(deex, signals, label=title, linewidth=2)
    plt.legend(fontsize=16)
    plt.ylabel(r'Inelastic cross section [$cm^2$]', fontsize=18)
    plt.xlabel('Deexcitation energy [MeV]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title(title, fontsize=22, loc='right')
    plt.tight_layout()
    plt.show()


# energy_bins = gamma energy bins
def inel_gen(prefactor, energy_bins, Jis, nucl_exes, prompt=False):
    signals = []
    for de in energy_bins:
        # delayed nu
        def delayed(enu):
            return xsec_GT_sum(enu, de, Jis, nucl_exes) * (dPdE_mu(enu) + dPdE_e(enu))
        signal = 0 if prompt else quad(delayed, 0, m_mu/2)[0]

        # prompt nu
        signal += xsec_GT_sum(e_prompt, de, Jis, nucl_exes)
        signals.append(signal)

    return np.array(signals) * prefactor


def get_nu_signals(experiment, prompt, timing_cut=True, energy_bins=None):
    mAs, Jis, Zs, nucl_exes = get_mA_Ji(experiment)
    prefactor = get_rates(experiment, specs='nu')[0]
    bin_width = (energy_bins[-1] - energy_bins[0]) / len(energy_bins)
    signals = inel_gen(prefactor, energy_bins, Jis, nucl_exes, prompt) * bin_width

    if timing_cut:
        # apply the timing cut: 0-0.22mus
        signals *= 5e-2

    return signals


# inelastic neutrino bkg
def inel_nu_bkg():
    energy_bins = np.linspace(0, 100, 1000)
    bin_width = energy_bins[1] - energy_bins[0]

    # prompt + delayed
    signals_ccm = get_nu_signals('CCM', energy_bins=energy_bins, prompt=False)
    signals_nai = get_nu_signals('COHERENT_NaI', energy_bins=energy_bins, prompt=False)
    # prompt only
    signals_ccm_p = get_nu_signals('CCM', energy_bins=energy_bins, prompt=True)
    signals_nai_p = get_nu_signals('COHERENT_NaI', energy_bins=energy_bins, prompt=True)

    print(np.sum(signals_ccm))
    print(np.sum(signals_nai))

    plt.figure(figsize=(8, 6))
    plt.plot(energy_bins, signals_ccm / bin_width, '-', color='blue', label='CCM (prompt + delayed)')
    plt.plot(energy_bins, signals_ccm_p / bin_width, '--', color='blue', label='CCM (prompt only)')
    plt.plot(energy_bins, signals_nai / bin_width, '-', color='orangered', label='NaI (prompt + delayed)')
    plt.plot(energy_bins, signals_nai_p / bin_width, '--', color='orangered', label='NaI (prompt only)')
    plt.legend()
    plt.title('Inelastic Neutrino Background', fontsize=20, loc='right')
    plt.xlabel('Deexcitation photon energy [MeV]', fontsize=18)
    # plt.ylabel('Events', fontsize=18)
    plt.ylabel('Events / 0.1 MeV', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, 20)
    plt.show()


if __name__ == "__main__":
    # dPdE_plot()
    inel_nu_bkg()
    # plot_conv_nu_xsec(30, 'CCM')
