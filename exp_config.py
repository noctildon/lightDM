from pyCEvNS.constants import *
from util import *
import numpy as np

# Constants
N_A = 6e23
mN = 939.565
Gf = 1.1664e-11 # Fermi constant (MeV^-2)
default_lifetime = 1e-4
alpha_D = .5
gA = 1.27


def get_bkg(experiment, energy_cut=True):
    if experiment in ['COHERENT_CsI', 'COHERENT_CsI_2018']:
        if experiment == 'COHERENT_CsI':
            # 2021 data release
            ac_bon = np.genfromtxt('Experiments/COHERENT2021/ac_bon.txt', delimiter=' ') # COHERENT CsI
        elif experiment == 'COHERENT_CsI_2018':
            # 2018 data release
            ac_bon = np.genfromtxt('Experiments/COHERENT/data_anticoincidence_beamOn.txt', delimiter=',')

        energy_bins = np.unique(ac_bon[:,0])
        bin_width = energy_bins[1] - energy_bins[0]
        energy_edges = np.linspace(energy_bins[0]-bin_width/2, energy_bins[-1]+bin_width/2, len(energy_bins)+1)

        energy_bins_number = len(energy_bins)
        timing_bins_number = int(ac_bon.shape[0] / energy_bins_number)

        bkg = []
        for i in range(energy_bins_number):
            # sume over timing bins
            events = 0
            for j in range(timing_bins_number):
                events += ac_bon[i*timing_bins_number +j, 2]
            bkg.append(events)
        bkg = np.array(bkg)
        # bkg += 1 # prevent div by 0

    elif experiment == 'COHERENT_NaI':
        energy_edges = np.genfromtxt('Experiments/COHERENT2021/coherentNaI energy edges.csv', delimiter=',')
        coherentNaI_photon = np.genfromtxt("Experiments/COHERENT2021/coherentNaI photon flux raw.csv", delimiter=",")
        bkg = np.ones_like(coherentNaI_photon[:, 0])
        bkg *= 400 / np.sum(bkg)

        if energy_cut:
            idx_lo = find_nearest(energy_edges, 5)  # lower window cut: 5MeV
            idx_hi = find_nearest(energy_edges, 16)  # upper window cut: 16MeV
            energy_edges = energy_edges[idx_lo:idx_hi+1]
            bkg = bkg[idx_lo:idx_hi]

    elif experiment == 'CCM_CsI':
        energy_edges = np.genfromtxt('Experiments/CCM/CCM bkg energy edges.csv', delimiter=',')
        idx_lo = find_nearest(energy_edges, 1)  # lower window cut: 1MeV
        idx_hi = find_nearest(energy_edges, 10)  # upper window cut: 10MeV

        energy_edges = energy_edges[idx_lo:idx_hi+1]
        bkg = np.genfromtxt('Experiments/CCM/CCM background.csv', delimiter=',')
        bkg = np.ones_like(bkg)
        bkg *= 10 / np.sum(bkg)
        bkg = bkg[idx_lo:idx_hi]


    elif experiment == 'CCM':
        energy_edges = np.genfromtxt('Experiments/CCM/CCM bkg energy edges.csv', delimiter=',')
        bkg = np.genfromtxt('Experiments/CCM/CCM background.csv', delimiter=',')
        bkg *= 1e-2 # background reduction

        # NOTE: POT scale up
        POT0 = 1.79e21
        POT = 2.25e22
        bkg *= POT / POT0

        if energy_cut:
            idx_lo = find_nearest(energy_edges, 10)  # lower window cut: 10MeV
            idx_hi = find_nearest(energy_edges, 20)  # upper window cut: 20MeV
            energy_edges = energy_edges[idx_lo:idx_hi+1]
            bkg = bkg[idx_lo:idx_hi]

    elif experiment == 'PIP2BD':
        # TODO: get better background
        energy_edges = np.genfromtxt('Experiments/CCM/CCM bkg energy edges.csv', delimiter=',')
        bkg = np.genfromtxt('Experiments/CCM/CCM background.csv', delimiter=',')
        bkg *= 1e-2 # background reduction

        POT0 = 1.79e21
        POT = 4.95e23
        bkg *= POT / POT0
        # bkg *= 100/7  # 7ton -> 100ton, but this seems not necessary

        if energy_cut:
            idx_lo = find_nearest(energy_edges, 10)  # lower window cut: 10MeV
            idx_hi = find_nearest(energy_edges, 20)  # upper window cut: 20MeV
            energy_edges = energy_edges[idx_lo:idx_hi+1]
            bkg = bkg[idx_lo:idx_hi]

    elif experiment == 'hyperK':
        # no strength rescale
        bin_width = 0.1
        energy_edges = np.arange(10, 50 + bin_width, bin_width) # 10-30MeV for background gamma
        bkg = np.genfromtxt('nus/nu_bkg_hyperK.csv', delimiter=' ')[:, 1]

    elif experiment == 'DUNE':
        # no strength rescale
        bin_width = 0.1
        energy_edges = np.arange(1, 20 + bin_width, bin_width) # 1-20MeV for background gamma
        bkg = np.genfromtxt('nus/nu_bkg_DUNE.csv', delimiter=' ')[:, 1]

    else:
        raise ValueError('Experiment not found. Must be one of COHERENT_CsI, COHERENT_NaI, CCM')

    return bkg, energy_edges


def get_nucl_strength(nucl):
    """
    nucl: 'ar40', 'na23', 'I127', 'cs133'
    return: 2d array [energy, strength] for the given nucleus
    """
    nucl_ex = np.genfromtxt(f'bigstick/{nucl}_s.res', delimiter='   ')
    nucl_ex[:, 0] = nucl_ex[:, 0] - nucl_ex[0, 0]  # reset the excitation energy
    nucl_ex = nucl_ex[nucl_ex[:,1] > 0]   # remove zero strength
    return nucl_ex


def get_mA_Ji(experiment):
    mA_na23 = 23*mN
    mA_ar40 = 40*mN
    mA_cs133 = 133*mN
    mA_I127 = 127*mN
    mA_o16 = 16*mN
    Ji_na23 = 1.5
    Ji_ar40 = 0
    Ji_cs133 = 3.5
    Ji_i127 = 2.5
    Ji_o16 = 0

    nucl_ex_ar40 = get_nucl_strength('ar40')
    nucl_ex_na23 = get_nucl_strength('na23')
    nucl_ex_cs133 = get_nucl_strength('cs133')
    nucl_ex_i127 = get_nucl_strength('I127')
    nucl_ex_o16 = get_nucl_strength('o16')

    # combine them
    nucl_exes_ar40 = [nucl_ex_ar40]
    nucl_exes_csi = [nucl_ex_cs133, nucl_ex_i127]
    nucl_exes_nai = [nucl_ex_na23, nucl_ex_i127]
    nucl_exes_o16 = [nucl_ex_o16]

    if experiment in ['COHERENT_CsI', 'COHERENT_CsI_2018', 'CCM_CsI']:
        mAs = [mA_cs133, mA_I127]
        Jis = [Ji_cs133, Ji_i127]
        Zs = [55, 53]
        return mAs, Jis, Zs, nucl_exes_csi

    elif experiment == 'COHERENT_NaI':
        mAs = [mA_na23, mA_I127]
        Jis = [Ji_na23, Ji_i127]
        Zs = [11, 53]
        return mAs, Jis, Zs, nucl_exes_nai

    elif experiment in ['CCM', 'DUNE', 'PIP2BD']:
        mAs = [mA_ar40]
        Jis = [Ji_ar40]
        Zs = [18]
        return mAs, Jis, Zs, nucl_exes_ar40

    elif experiment == 'hyperK':
        mAs = [mA_o16]
        Jis = [Ji_o16]
        Zs = [8]
        return mAs, Jis, Zs, nucl_exes_o16

    else:
        raise ValueError('Experiment not found. Must be one of COHERENT_CsI, COHERENT_NaI, CCM, hyperK')


def get_wimp_size(experiment):
    if experiment == 'hyperK':
        det_mass = 0.188e9  #  0.188M tons water [kg]
        det_num = det_mass * 1e3 * N_A / 16  # total number of O16

    if experiment == 'DUNE':
        det_mass = 40e6  #  40kton LAr [kg]
        det_num = det_mass * 1e3 * N_A / 40  # total number of Ar40

    return det_mass, det_num


def get_rates(experiment, specs='DM'):
    """
    experiment: 'COHERENT_CsI', 'COHERENT_NaI', 'CCM', 'CCM_CsI', 'PIP2BD'
    specs: 'DM', 'nu'
    """
    # how many atoms in the detector per kg
    AtomsCsI = 1e3 * N_A / (133+127)
    AtomsNaI = 1e3 * N_A / (23+127)
    AtomsLAr = 1e3 * N_A / 40

    pim_rate_coherent = 0.0457
    pim_rate_ccm = 0.0259
    pim_rate_pip2bd  = 0.2334719

    pion_rate_coherent = 0.1048
    pion_rate_ccm = 0.0633
    pion_rate_pip2bd  = 0.3221147

    yearsCsI = 1.61
    yearsCsI_2018 = 0.84
    yearsNaI = 3
    yearsCCM = 3
    yearsPIP2BD = 5
    yearsCCM_CsI = 3

    massCsI = 14.6 # kg
    massNaI = 3500
    massCCM = 7000
    # massCCM = 100e3  # future upgrade
    massPIP2BD = 100e3
    massCCM_CsI = 1e3 # 1ton CsI

    DistanceCsI = 19.3 # meter
    DistanceNaI = 22
    DistanceCCM = 20
    DistancePIP2BD = 15 # or 30

    # gaussian POT
    pot_mu_coherent = 0.7
    pot_mu_ccm = 0.145
    pot_mu_pip2bd = 0.15
    pot_sigma_coherent = 0.15
    pot_sigma_ccm = pot_mu_ccm / 2
    pot_sigma_pip2bd = pot_mu_pip2bd / 2

    # total POT
    POTCsI = 3.2e23
    POTCsI_2018 = yearsCsI_2018*1.5e23
    POTNaI = yearsNaI*6e23
    POTccm = yearsCCM*7.5e21
    POTccm_CsI = yearsCCM_CsI*1e22
    POTpip2bd = 4.95e23


    ########################## Derivation #########################################
    pot_rate_per_day_csi = int(POTCsI / yearsCsI / 365)
    pot_rate_per_day_csi_2018 = int(POTCsI_2018 / yearsCsI_2018 / 365)
    pot_rate_per_day_nai = int(POTNaI / yearsNaI / 365)
    pot_rate_per_day_ccm = int(POTccm / yearsCCM / 365)
    pot_rate_per_day_pip2bd = int(POTpip2bd / yearsPIP2BD / 365)

    secCsI = yearsCsI*365*24*60**2
    secCsI_2018 = yearsCsI_2018*365*24*60**2
    secNaI = yearsNaI*365*24*60**2
    secCCM = yearsCCM*365*24*60**2
    secCCM_CsI = yearsCCM_CsI*365*24*60**2
    secPIP2BD = yearsPIP2BD*365*24*60**2

    FluxCsI = POTCsI / (4 *np.pi* DistanceCsI**2 * secCsI) # m^-2 s^-1
    FluxCsI_2018 = POTCsI_2018 / (4 *np.pi* DistanceCsI**2 * secCsI_2018)
    FluxNaI = POTNaI / (4 *np.pi* DistanceNaI**2 * secNaI)
    FluxCCM = POTccm / (4 *np.pi* DistanceCCM**2 * secCCM)
    FluxCCM_CsI = POTccm_CsI / (4 *np.pi* DistanceCCM**2 * secCCM_CsI)
    FluxPIP2BD = POTpip2bd / (4 *np.pi* DistancePIP2BD**2 * secPIP2BD)

    ExposureCsI = massCsI *yearsCsI*365 # kg days
    ExposureCsI_2018 = massCsI *yearsCsI_2018*365 # kg days
    ExposureNaI = massNaI *yearsNaI*365
    ExposureCCM = massCCM *yearsCCM*365
    ExposureCCM_CsI = massCCM_CsI *yearsCCM_CsI*365
    ExposurePIP2BD = massPIP2BD *yearsPIP2BD*365
    ########################## Derivation #########################################


    if specs == 'nu':
        prefactorCsI = pion_rate_coherent * FluxCsI * ExposureCsI * AtomsCsI *24*60**2 * meter_by_mev**2
        prefactorCsI_2018 = pion_rate_coherent * FluxCsI_2018 * ExposureCsI_2018 * AtomsCsI *24*60**2 * meter_by_mev**2
        prefactorNaI = pion_rate_coherent * FluxNaI * ExposureNaI * AtomsNaI *24*60**2 * meter_by_mev**2
        prefactorccm = pion_rate_ccm * FluxCCM * ExposureCCM * AtomsLAr *24*60**2 * meter_by_mev**2
        prefactorccm_CsI = pion_rate_ccm * FluxCCM_CsI * ExposureCCM_CsI * AtomsCsI *24*60**2 * meter_by_mev**2
        prefactorPIP2BD = pion_rate_pip2bd * FluxPIP2BD * ExposurePIP2BD * AtomsLAr *24*60**2 * meter_by_mev**2

    elif specs == 'DM':
        prefactorCsI = secCsI * massCsI * AtomsCsI
        prefactorCsI_2018 = secCsI_2018 * massCsI * AtomsCsI
        prefactorNaI = secNaI * massNaI * AtomsNaI
        prefactorccm = secCCM * massCCM * AtomsLAr
        prefactorccm_CsI = secCCM_CsI * massCCM_CsI * AtomsCsI
        prefactorPIP2BD = secPIP2BD * massPIP2BD * AtomsLAr

    else:
        raise ValueError('specs must be one of DM, nu')

    if experiment == 'COHERENT_CsI':
        return prefactorCsI, pot_rate_per_day_csi, pim_rate_coherent, pion_rate_coherent, DistanceCsI, secCsI, massCsI, AtomsCsI, pot_mu_coherent, pot_sigma_coherent

    elif experiment == 'COHERENT_CsI_2018':
        return prefactorCsI_2018, pot_rate_per_day_csi_2018, pim_rate_coherent, pion_rate_coherent, DistanceCsI, secCsI_2018, massCsI, AtomsCsI, pot_mu_coherent, pot_sigma_coherent

    elif experiment == 'COHERENT_NaI':
        return prefactorNaI, pot_rate_per_day_nai, pim_rate_coherent, pion_rate_coherent, DistanceNaI, secNaI, massNaI, AtomsNaI, pot_mu_coherent, pot_sigma_coherent

    elif experiment == 'CCM':
        return prefactorccm, pot_rate_per_day_ccm, pim_rate_ccm, pion_rate_ccm, DistanceCCM, secCCM, massCCM, AtomsLAr, pot_mu_ccm, pot_sigma_ccm

    elif experiment == 'CCM_CsI':
        return prefactorccm_CsI, pot_rate_per_day_ccm, pim_rate_ccm, pion_rate_ccm, DistanceCCM, secCCM_CsI, massCCM_CsI, AtomsCsI, pot_mu_ccm, pot_sigma_ccm

    elif experiment == 'PIP2BD':
        return prefactorPIP2BD, pot_rate_per_day_pip2bd, pim_rate_pip2bd, pion_rate_pip2bd, DistancePIP2BD, secPIP2BD, massPIP2BD, AtomsLAr, pot_mu_pip2bd, pot_sigma_pip2bd

    else:
        raise ValueError('Experiment not found. Must be one of COHERENT_CsI, COHERENT_NaI, CCM, CCM_CsI')


def get_pion_brem_flux(experiment):
    if experiment in ['COHERENT_CsI', 'COHERENT_CsI_2018', 'COHERENT_NaI']:
        brem_photons_coherent = np.genfromtxt("Experiments/COHERENT/brem.txt")
        Pi0Info = np.genfromtxt("Experiments/COHERENT/Pi0_Info.txt")
        pion_flux = pion_info2flux(Pi0Info)

        return pion_flux, brem_photons_coherent

    elif experiment in ['CCM', 'CCM_CsI']:
        brem_photons_ccm = np.genfromtxt("Experiments/CCM/brem.txt")
        Pi0Info = np.genfromtxt("Experiments/CCM/Pi0_Info.txt")
        pion_flux = pion_info2flux(Pi0Info)

        return pion_flux, brem_photons_ccm

    elif experiment in ['PIP2BD']:
        # TODO: find the PIP2BD brem data
        brem_photons_ccm = np.genfromtxt("Experiments/CCM/brem.txt")
        pion_flux = pion_info2flux_shielded()

        return pion_flux, brem_photons_ccm

    else:
        raise ValueError('Experiment not found. Must be one of COHERENT_CsI, COHERENT_NaI, CCM')


def get_strength_scale():
    """
    return: rescale factor for the strength (as compared to the Anna's paper)
    """
    def get_strength_sum(nucleus, e_min, e_max):
        nucl_ex_path = f'bigstick/{nucleus}_s.res'
        nucl_ex = np.genfromtxt(nucl_ex_path, delimiter='   ')
        nucl_ex[:, 0] = nucl_ex[:, 0] - nucl_ex[0, 0]
        nucl_ex = nucl_ex[np.where(nucl_ex[:, 1] != 0)]
        bigstick_energy = nucl_ex[:, 0]
        bigstick_strength = nucl_ex[:, 1]

        strength_sum = 0
        for e, s in zip(bigstick_energy, bigstick_strength):
            if e_min < e < e_max:
                strength_sum += s
        return strength_sum

    # 4-13 MeV states for Ar40
    strength_anna = 0.651 # B(M1) dimensionful from https://doi.org/10.1016/j.physletb.2022.137576
    strength_our = get_strength_sum('ar40', 4, 13)  # B(GT) dimensionless
    strength_our *= 2.2993**2  # B(M1) dimensionful

    return strength_anna/strength_our
