import re
import itertools
from pyCEvNS.constants import *
from util import *
import numpy as np
import pandas as pd

# Constants
N_A = 6e23
mN = 939.565
Gf = 1.1664e-11 # Fermi constant (MeV^-2)
default_lifetime = 1e-4 # dark photon decay to DM lifetime [ms]
alpha_D = .5
gA = 1.27
experiments_df = pd.read_csv('Experiments/summary.csv', delimiter=',', skipinitialspace=True)
nuclei_df = pd.read_csv('bigstick/nuclei.csv', delimiter=',', skipinitialspace=True)


def experiment_dict(experiment):
    """
    Input an experiment name, output a dictionary consisting of the experiment's parameters
    """
    exp_dict = experiments_df[experiments_df['name'] == experiment].to_dict('records')[0]
    return exp_dict


def get_bkg(experiment, energy_cut=True, plot_mode=False):
    """
    return: the (reduced) total background in full run, and the energy edges
    """
    exp_dict = experiment_dict(experiment)
    years = float(exp_dict['years'])
    bkg_red = float(exp_dict['bkg red']) # < 1
    det_mass = float(exp_dict['det mass[kg]']) # kg

    if experiment in ['COHERENT_CsI', 'COHERENT_CsI_2018']:
        excess_per_year = 0

        if experiment == 'COHERENT_CsI':
            # 2021 data release
            # PE, mus, events
            ac_bon = np.genfromtxt('Experiments/COHERENT2021/ac_bon.txt', delimiter=' ')
        elif experiment == 'COHERENT_CsI_2018':
            # 2018 data release
            ac_bon = np.genfromtxt('Experiments/COHERENT/data_anticoincidence_beamOn.txt', delimiter=',')

        pe_per_mev = 0.0878 * 13.348 * 1e3
        energy_bins = np.unique(ac_bon[:,0])
        energy_bins /= pe_per_mev # convert PE to MeV
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
        yr = 1.61 if experiment == 'COHERENT_CsI' else 0.84
        bkg += 1 # prevent div by 0
        bkg_per_year = bkg / yr

    elif experiment in ['COHERENT_NaI', 'KARMEN']:
        excess_per_year = 0

        energy_edges = np.genfromtxt('Experiments/COHERENT2021/coherentNaI energy edges.csv', delimiter=',')
        coherentNaI_photon = np.genfromtxt("Experiments/COHERENT2021/coherentNaI photon flux raw.csv", delimiter=",")
        bkg = np.ones_like(coherentNaI_photon[:, 0])
        bkg *= 400 / np.sum(bkg)

        if energy_cut:
            # NaI: 1-10MeV, KARMEN (C12): 8-12MeV
            idx_lo = 1 if experiment == 'COHERENT_NaI' else 8
            idx_hi = 10 if experiment == 'COHERENT_NaI' else 12

            idx_lo = find_nearest(energy_edges, idx_lo)
            idx_hi = find_nearest(energy_edges, idx_hi)

            energy_edges = energy_edges[idx_lo:idx_hi+1]
            bkg = bkg[idx_lo:idx_hi]
        bkg_per_year = bkg / 3

    elif experiment == 'CCM_Ge':
        excess_per_year = 0

        energy_edges_keV = np.logspace(-2, 1, 100)
        energy_edges = energy_edges_keV * 1e-3 # convert to MeV
        bkg_per_year = np.ones_like(energy_edges) # bkg is not determined yet

    elif experiment == 'CCM_Ge_e':
        excess_per_year = 0

        # DM-e scattering
        energy_edges_keV = np.logspace(-3, 1, 100)
        energy_edges = energy_edges_keV * 1e-3 # convert to MeV
        bkg_per_year = np.ones_like(energy_edges) # bkg is not determined yet

    elif experiment == 'CCM_CsI':
        excess_per_year = 0

        energy_edges = np.genfromtxt('Experiments/CCM/CCM bkg energy edges.csv', delimiter=',')
        idx_lo = find_nearest(energy_edges, 1)
        idx_hi = find_nearest(energy_edges, 10)

        energy_edges = energy_edges[idx_lo:idx_hi+1]
        bkg = np.genfromtxt('Experiments/CCM/CCM background.csv', delimiter=',')
        bkg = np.ones_like(bkg)
        bkg *= 10 / np.sum(bkg)  # O(10) background
        bkg = bkg[idx_lo:idx_hi]
        bkg_per_year = bkg

    elif experiment in ['CCM120']:
        excess_per_year = 0

        energy_edges = np.genfromtxt('Experiments/CCM/CCM bkg energy edges.csv', delimiter=',')
        bkg = np.genfromtxt('Experiments/CCM/CCM background.csv', delimiter=',')
        bkg *= 2.25e22 / 1.79e21 # POT scale up

        if energy_cut:
            idx_lo = find_nearest(energy_edges, 9)
            idx_hi = find_nearest(energy_edges, 15)
            energy_edges = energy_edges[idx_lo:idx_hi+1]
            bkg = bkg[idx_lo:idx_hi]
        bkg_per_year = bkg / 3

        # use CCM120 to simulate PIP2BD
        # if experiment == 'PIP2BD':
        #     bkg_per_year *= 9.9e22 / 2.25e22 # POT scale up
        #     bkg_per_year *= 100/7  # mass scale up 7->100ton

    elif experiment in ['CCM200', 'PIP2BD']:
        excess_raw = np.genfromtxt('Experiments/CCM200/excess.csv', delimiter=',')
        energy_edges = np.genfromtxt('Experiments/CCM200/CCM bkg energy edges.csv', delimiter=',')
        bkg_raw = np.genfromtxt('Experiments/CCM200/CCM bkg.csv', delimiter=',')

        # convert from counts/PE to counts
        bkg = np.zeros(len(bkg_raw))
        for row in bkg_raw:
            idx = int(row[0])
            eventsperPE = row[1]
            binwidth = energy_edges[idx] - energy_edges[idx-1]
            bkg[idx-1] = eventsperPE * binwidth

        excess = np.zeros(len(excess_raw))
        for row in excess_raw:
            idx = int(row[0])
            eventsperPE = row[1]
            binwidth = energy_edges[idx] - energy_edges[idx-1]
            excess[idx-1] = eventsperPE * binwidth

        energy_edges *= 1/45  # convert to MeV (45PE = 1MeV)
        if energy_cut:
            idx_lo = find_nearest(energy_edges, 10)
            idx_hi = find_nearest(energy_edges, 14)

            energy_edges = energy_edges[idx_lo:idx_hi+1]
            bkg = bkg[idx_lo:idx_hi]
            excess = excess[idx_lo:idx_hi]

        bkg_per_year = bkg*12/1.5 # bkg was generated for 1.5 month, scale to 1 year
        excess_per_year = excess*12/1.5 # excess was generated for 1.5 month, scale to 1 year

        # use CCM200 to simulate PIP2BD
        if experiment == 'PIP2BD':
            bkg_per_year *= 9.9e22 / 1.76e21 # POT scale up
            bkg_per_year *= 100/7  # mass scale up 7->100ton

    elif experiment == 'SBND':
        excess_per_year = 0

        # NOTE: use CCM200 bkg to simulate SBND
        energy_edges = np.genfromtxt('Experiments/CCM200/CCM bkg energy edges.csv', delimiter=',')
        bkg_raw = np.genfromtxt('Experiments/CCM200/CCM bkg.csv', delimiter=',')

        # convert from counts/PE to counts
        bkg = np.zeros(len(bkg_raw))
        for row in bkg_raw:
            idx = int(row[0])
            eventsperPE = row[1]
            binwidth = energy_edges[idx] - energy_edges[idx-1]
            bkg[idx-1] = eventsperPE * binwidth

        energy_edges *= 1/45  # convert to MeV (45PE = 1MeV)
        if energy_cut:
            idx_lo = find_nearest(energy_edges, 10)
            idx_hi = find_nearest(energy_edges, 20)
            energy_edges = energy_edges[idx_lo:idx_hi+1]
            bkg = bkg[idx_lo:idx_hi]

        bkg_per_year = 10/np.sum(bkg) * bkg  # O(10) background

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    elif experiment in ['superK', 'hyperK']:
        excess_per_year = 0

        bin_width = 2
        if plot_mode:
            bin_width = 0.1
            energy_cut = False

        energy_edges = np.arange(10, 50 + bin_width, bin_width) # 10-30MeV for background gamma
        energy_bins = (energy_edges[1:] + energy_edges[:-1])/2

        bkg_file = np.genfromtxt('nus/nu_bkg_hyperK.csv', delimiter=' ') # generated for 188kton, bin_width=0.1
        bkg = bkg_file[:, 1] # s^-1
        bkg_energy_bins = bkg_file[:, 0]

        indices = [find_nearest(bkg_energy_bins, e) for e in energy_bins] # adjust the binning
        bkg_per_year = bkg[indices] * 365*24*3600
        bkg_per_year *= det_mass / 188e6 # scale the detector mass

        if energy_cut:
            idx_lo = find_nearest(energy_edges, 10)
            idx_hi = find_nearest(energy_edges, 40)
            energy_edges = energy_edges[idx_lo:idx_hi+1]
            bkg_per_year = bkg_per_year[idx_lo:idx_hi]

    elif experiment == 'DUNE':
        excess_per_year = 0

        bin_width = 2
        if plot_mode:
            bin_width = 0.1
            energy_cut = False

        energy_edges = np.arange(1, 20 + bin_width, bin_width) # 1-20MeV for background gamma
        energy_bins = (energy_edges[1:] + energy_edges[:-1])/2

        bkg_file = np.genfromtxt('nus/nu_bkg_DUNE.csv', delimiter=' ') # generated for 40kton, bin_width=0.1
        bkg = bkg_file[:, 1] # s^-1
        bkg_energy_bins = bkg_file[:, 0]

        indices = [find_nearest(bkg_energy_bins, e) for e in energy_bins] # adjust the binning
        bkg_per_year = bkg[indices] * 365*24*3600
        bkg_per_year *= det_mass / 40e6 # scale the detector mass

        if energy_cut:
            idx_lo = find_nearest(energy_edges, 10)
            idx_hi = find_nearest(energy_edges, 14.5)
            energy_edges = energy_edges[idx_lo:idx_hi+1]
            bkg_per_year = bkg_per_year[idx_lo:idx_hi]

    elif experiment == 'JUNO':
        excess_per_year = 0

        bin_width = 0.5
        if plot_mode:
            bin_width = 0.1
            energy_cut = False

        energy_edges = np.arange(1, 20 + bin_width, bin_width) # 1-20MeV for background gamma
        energy_bins = (energy_edges[1:] + energy_edges[:-1])/2

        bkg_file = np.genfromtxt('nus/nu_bkg_JUNO.csv', delimiter=' ') # generated for 20kton, bin_width=0.1
        bkg = bkg_file[:, 1] # s^-1
        bkg_energy_bins = bkg_file[:, 0]

        indices = [find_nearest(bkg_energy_bins, e) for e in energy_bins] # adjust the binning
        bkg_per_year = bkg[indices] * 365*24*3600
        bkg_per_year *= det_mass / 20e6 # scale the detector mass

        if energy_cut:
            idx_lo = find_nearest(energy_edges, 8)
            idx_hi = find_nearest(energy_edges, 14)
            energy_edges = energy_edges[idx_lo:idx_hi+1]
            bkg_per_year = bkg_per_year[idx_lo:idx_hi]

    else:
        raise ValueError('Experiment not found')

    total_bkg = bkg_per_year*bkg_red*years
    total_excess = excess_per_year*years
    return total_bkg, energy_edges, total_excess


def get_nucl_strength(nucl):
    """
    nucl: 'ar40', 'ar39', 'na23', 'I127', 'cs133', 'o16', 'he4', 'n14', 'c12', 'ge76'
    return: 2d array [energy, strength] for the given nucleus
    """
    nucl_ex = np.genfromtxt(f'bigstick/{nucl}_s.res', delimiter='   ')
    nucl_ex[:, 0] = nucl_ex[:, 0] - nucl_ex[0, 0]  # reset the excitation energy
    nucl_ex = nucl_ex[1:]                 # remove the ground state
    nucl_ex = nucl_ex[nucl_ex[:,1] > 0]   # remove zero strength
    return nucl_ex


def get_mA_Ji(experiment, e_range='all'):
    """
    experiment: COHERENT_CsI, COHERENT_CsI_2018, CCM_CsI, COHERENT_NaI, CCM, CCM200, PIP2BD
            OR superK, hyperK, DUNE, JUNO
            OR c12, n14, he4, ar39, ar40, o16, xe131, xe132, xe134
    e_range: energy range of the states to include, 'all' if include all states
    """
    if experiment == 'CCM_Ge_e':
        # DM-e scattering in Ge. sqrt(Z) to take care of the Z^2 factor in cross section
        return [me], None, [np.sqrt(32)], None

    mAs, Jis, Zs, nucl_exes = [], [], [], []
    if experiment in nuclei_df['nucleus'].to_list():
        nucl_dict = nuclei_df[nuclei_df['nucleus'] == experiment].to_dict('records')[0]
        mAs.append(nucl_dict['mass number']*mN)
        Jis.append(nucl_dict['nuclear spin'])
        Zs.append(nucl_dict['atomic number'])
        strength_scale = nucl_dict['strength scale']
        nucl_ex = get_nucl_strength(experiment) # [energy, strength]
        nucl_ex[:, 1] *= strength_scale # scale the strength
        nucl_exes.append(nucl_ex)
    else:
        exp_dict = experiment_dict(experiment)
        exp_dict['material'] = exp_dict['material'].split('+')

        for nucl in exp_dict['material']:
            mAs_, Jis_, Zs_, nucl_exes_ = get_mA_Ji(nucl)
            mAs += mAs_
            Jis += Jis_
            Zs += Zs_
            nucl_exes += nucl_exes_

    if e_range != 'all':
        for idx, nucl_ex in enumerate(nucl_exes):
            nucl_ex = nucl_ex[np.where(nucl_ex[:, 0] < e_range[1])]
            nucl_ex = nucl_ex[np.where(nucl_ex[:, 0] > e_range[0])]
            nucl_exes[idx] = nucl_ex

    return mAs, Jis, Zs, nucl_exes


def get_wimp_size(experiment):
    """
    experiment: superK, hyperK, DUNE
    """
    if experiment not in ['superK', 'hyperK', 'DUNE', 'JUNO']:
        raise ValueError('experiment must be superK, hyperK, DUNE, JUNO')

    exp_dict = experiment_dict(experiment)
    atoms = [re.findall(r'\d+', i) for i in exp_dict['material'].split('+')]
    atoms = list(itertools.chain(*atoms)) # flatten
    atoms = sum(map(int, atoms)) # str to int to sum
    atoms = 1e3 * N_A / atoms    # how many atoms in the detector per kg

    det_mass = float(exp_dict['det mass[kg]']) # detector mass [kg]
    det_num = det_mass * atoms # total number of molecules
    det_year = float(exp_dict['years'])
    return det_mass, det_num, det_year


def get_rates(experiment, specs='DM'):
    """
    experiment: 'COHERENT_CsI', 'COHERENT_CsI_2018', 'COHERENT_NaI', 'CCM120', 'CCM_Ge', 'CCM_Ge_e', 'CCM200', 'CCM_CsI', 'PIP2BD'
    specs: 'DM', 'nu'
    """
    exp_dict = experiment_dict(experiment)

    pot_mu = float(exp_dict['POT mu'])
    pot_sigma = float(exp_dict['POT sigma'])
    pim_rate = float(exp_dict['pim rate'])
    pion_rate = float(exp_dict['pion rate'])
    det_mass = float(exp_dict['det mass[kg]'])
    distance = float(exp_dict['distance[m]'])
    years = float(exp_dict['years'])
    pot_per_year = float(exp_dict['POT/year'])
    pot_per_day = int(pot_per_year / 365)
    secs = years*365*24*60**2
    flux = pot_per_day / (4*np.pi*distance**2 * 24*60**2) # m^-2 s^-1
    exposure = det_mass *years*365 # kg days

    atoms = [re.findall(r'\d+', i) for i in exp_dict['material'].split('+')]
    atoms = list(itertools.chain(*atoms)) # flatten
    atoms = sum(map(int, atoms)) # str to int to sum
    atoms = 1e3 * N_A / atoms    # how many atoms in the detector per kg

    if specs == 'nu':
        prefactor = pion_rate * flux * exposure * atoms *24*60**2 * meter_by_mev**2
    elif specs == 'DM':
        prefactor = secs * det_mass * atoms
    else:
        raise ValueError('specs must be nu or DM')

    return prefactor, pot_per_day, pim_rate, pion_rate, distance, secs, det_mass, atoms, pot_mu, pot_sigma


def get_pion_brem_flux(experiment):
    # the brem and Pi0_Info is simulated for 1e5 POT
    if experiment in ['COHERENT_CsI', 'COHERENT_CsI_2018', 'COHERENT_NaI']:
        brem_photons_coherent = np.genfromtxt("Experiments/COHERENT/brem.txt")
        Pi0Info = np.genfromtxt("Experiments/COHERENT/Pi0_Info.txt")
        pion_flux = pion_info2flux(Pi0Info)
        return pion_flux, brem_photons_coherent

    elif experiment in ['CCM120', 'CCM_CsI', 'CCM200', 'CCM_Ge', 'CCM_Ge_e']:
        brem_photons_ccm = np.genfromtxt("Experiments/CCM/brem.txt")
        Pi0Info = np.genfromtxt("Experiments/CCM/Pi0_Info.txt")
        pion_flux = pion_info2flux(Pi0Info)
        return pion_flux, brem_photons_ccm

    elif experiment in ['PIP2BD']:
        # TODO: find the PIP2BD brem data
        brem_photons_ccm = np.genfromtxt("Experiments/CCM/brem.txt")
        pion_flux = pion_info2flux_shielded()
        return pion_flux, brem_photons_ccm

    elif experiment in ['SBND']:
        brem_photons_ccm = np.genfromtxt("Experiments/CCM/brem.txt")
        pion_flux = pion_info2flux_BNB()
        return pion_flux, brem_photons_ccm

    elif experiment in ['KARMEN']:
        # return get_pion_brem_flux('COHERENT_NaI')
        return get_pion_brem_flux('CCM200')

    else:
        raise ValueError('Experiment not found')


def get_strength_scale(nucl='ar40'):
    """
    return: rescale factor for the strength (as compared to the literature)
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

    refs = {
        # nucleus: [strength (mu_N^2), e_min, e_max]
        'ar40': [0.651, 4, 13], # https://doi.org/10.1016/j.physletb.2022.137576
        'o16': [1.3636, 16, 28], # Magnetic Dipole Strength in O16, A. Arima (theoretical)
        'fe54': [1.59, 0, 10.6], # https://journals.aps.org/prc/abstract/10.1103/PhysRevC.101.064303
        'xe132': [0.658, 2.383, 3.9],
        'xe134': [0.491, 2.372, 3.943],
    }

    s, e_min, e_max = refs[nucl]
    strength_ref = s # B(M1) dimensionful (from reference)
    strength_our = get_strength_sum(nucl, e_min, e_max)  # B(GT) dimensionless (from bigstick)
    strength_our *= 2.2993**2  # B(M1) dimensionful

    return strength_ref/strength_our


if __name__ == "__main__":
    r_o = get_strength_scale('o16')
    r_ar = get_strength_scale('ar40')
    r_fe = get_strength_scale('fe54')
    r_xe132 = get_strength_scale('xe132')
    r_xe134 = get_strength_scale('xe134')

    print(f'o16 strength rescale: {r_o} (its actually theoretical)')
    print(f'ar40 strength rescale: {r_ar}')
    print(f'fe54 strength rescale: {r_fe}')
    print(f'xe132 strength rescale: {r_xe132}') # 0.6172424300769448
    print(f'xe134 strength rescale: {r_xe134}') # 0.677599612186958
