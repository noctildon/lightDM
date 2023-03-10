"""
CEvNS events
"""

from scipy.special import spherical_jn

from .constants import *
from .detectors import *
from .flux import *
from .helper import _poisson
from scipy.special import gamma


def formfsquared(q, rn=4.7, **kwargs):
    """
    form factor squared
    1810.05606
    :param q: momentum transfered
    :param rn: neutron skin radius
    :param kwargs: this is for legacy compatibility
    :return: form factor squared
    """
    r = rn * (10 ** -15) / meter_by_mev
    s = 0.9 * (10 ** -15) / meter_by_mev
    r0 = np.sqrt(5 / 3 * (r ** 2) - 5 * (s ** 2))
    return (3 * spherical_jn(1, q * r0) / (q * r0) * np.exp((-(q * s) ** 2) / 2)) ** 2


def eff_coherent(er):
    pe_per_mev = 0.0878 * 13.348 * 1000
    pe = er * pe_per_mev
    a = 0.6655
    k = 0.4942
    x0 = 10.8507
    f = a / (1 + np.exp(-k * (pe - x0)))
    if pe < 5:
        return 0
    if pe < 6:
        return 0.5 * f
    return f


def rates_nucleus(er, det: Detector, fx: Flux, efficiency=None, f=None, nsip=NSIparameters(), flavor='e',
                  op=oscillation_parameters(), ffs=formfsquared, q2=False, **kwargs):
    """
    calculating scattering rates per nucleus
    :param er: recoil energy
    :param det: detector
    :param fx: flux
    :param f: oscillation function
    :param efficiency: efficiency function
    :param flavor: flux flavor
    :param nsip: nsi parameters
    :param op: oscillation parameters
    :param ffs: custom formfactor sqared function
    :param q2: whether to include q^2 formfactor
    :return: scattering rates per nucleus
    """
    deno = 2 * np.sqrt(2) * gf * (2 * det.m * er + nsip.mz ** 2)
    # radiative corrections,
    # Barranco, 2005
    # is it necessary?
    rho = 1.0086
    knu = 0.9978
    lul = -0.0031
    ldl = -0.0025
    ldr = 7.5e-5
    lur = ldr / 2
    q2fact = 1.0
    if q2:
        q2fact = 2 * det.m * er
    if nsip.mz != 0:
        if flavor[0] == 'e':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * q2fact * nsip.gu['ee'] / deno + q2fact * nsip.gd['ee'] / deno) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  q2fact * nsip.gu['ee'] / deno + 2 * q2fact * nsip.gd['ee'] / deno)) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.gu['em'] / deno + nsip.gd['em'] / deno) +
                         0.5 * det.n * (nsip.gu['em'] / deno + 2 * nsip.gd['em'] / deno)) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.gu['et'] / deno + nsip.gd['et'] / deno) +
                         0.5 * det.n * (nsip.gu['et'] / deno + 2 * nsip.gd['et'] / deno)) ** 2
        elif flavor[0] == 'm':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * q2fact * nsip.gu['mm'] / deno + q2fact * nsip.gd['mm'] / deno) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  q2fact * nsip.gu['mm'] / deno + 2 * q2fact * nsip.gd['mm'] / deno)) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.gu['em'] / deno + nsip.gd['em'] / deno) +
                         0.5 * det.n * (nsip.gu['em'] / deno + 2 * nsip.gd['em'] / deno)) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.gu['mt'] / deno + nsip.gd['mt'] / deno) +
                         0.5 * det.n * (nsip.gu['mt'] / deno + 2 * nsip.gd['mt'] / deno)) ** 2
        elif flavor[0] == 't':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * q2fact * nsip.gu['tt'] / deno + q2fact * nsip.gd['tt'] / deno) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  q2fact * nsip.gu['tt'] / deno + 2 * q2fact * nsip.gd['tt'] / deno)) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.gu['et'] / deno + nsip.gd['et'] / deno) +
                         0.5 * det.n * (nsip.gu['et'] / deno + 2 * nsip.gd['et'] / deno)) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.gu['mt'] / deno + nsip.gd['mt'] / deno) +
                         0.5 * det.n * (nsip.gu['mt'] / deno + 2 * nsip.gd['mt'] / deno)) ** 2
        else:
            raise Exception('No such neutrino flavor!')
    else:
        if flavor[0] == 'e':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * q2fact * nsip.epu['ee'] + q2fact * nsip.epd['ee']) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  q2fact * nsip.epu['ee'] + 2 * q2fact * nsip.epd['ee'])) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.epu['em'] + nsip.epd['em']) +
                         0.5 * det.n * (nsip.epu['em'] + 2 * nsip.epd['em'])) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.epu['et'] + nsip.epd['et']) +
                         0.5 * det.n * (nsip.epu['et'] + 2 * nsip.epd['et'])) ** 2
        elif flavor[0] == 'm':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * q2fact * nsip.epu['mm'] + q2fact * nsip.epd['mm']) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  q2fact * nsip.epu['mm'] + 2 * q2fact * nsip.epd['mm'])) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.epu['em'] + nsip.epd['em']) +
                         0.5 * det.n * (nsip.epu['em'] + 2 * nsip.epd['em'])) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.epu['mt'] + nsip.epd['mt']) +
                         0.5 * det.n * (nsip.epu['mt'] + 2 * q2fact * nsip.epd['mt'])) ** 2
        elif flavor[0] == 't':
            qvs = (0.5 * det.z * (rho * (0.5 - 2 * knu * ssw) + 2 * lul + 2 * lur + ldl + ldr +
                                  2 * q2fact * nsip.epu['tt'] + q2fact * nsip.epd['tt']) +
                   0.5 * det.n * (-0.5 * rho + lul + lur + 2 * ldl + 2 * ldr +
                                  q2fact * nsip.epu['tt'] + 2 * q2fact * nsip.epd['tt'])) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.epu['et'] + nsip.epd['et']) +
                         0.5 * det.n * (nsip.epu['et'] + 2 * nsip.epd['et'])) ** 2 + \
                  np.abs(0.5 * det.z * (2 * nsip.epu['mt'] + nsip.epd['mt']) +
                         0.5 * det.n * (nsip.epu['mt'] + 2 * nsip.epd['mt'])) ** 2
        else:
            raise Exception('No such neutrino flavor!')
    if efficiency is not None:
        return np.dot(2 / np.pi * (gf ** 2) * (2 * fx.fint(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                                         2 * er * fx.fintinv(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                                         er * er * fx.fintinvs(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                                         det.m * er * fx.fintinvs(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)) *
                   det.m * qvs * ffs(np.sqrt(2 * det.m * er), **kwargs), det.frac) * efficiency(er)
    else:
        return np.dot(2 / np.pi * (gf ** 2) * (2 * fx.fint(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                                         2 * er * fx.fintinv(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                                         er * er * fx.fintinvs(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                                         det.m * er * fx.fintinvs(er, det.m, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)) *
                   det.m * qvs * ffs(np.sqrt(2 * det.m * er), **kwargs), det.frac)


def rates_electron(er, det: Detector, fx: Flux, efficiency=None, f=None, nsip=NSIparameters(), flavor='e',
                   op=oscillation_parameters(), **kwargs):
    """
    calculating electron scattering rates per nucleus
    :param er: recoil energy
    :param det: detector
    :param fx: flux
    :param f: oscillation function
    :param efficiency: efficiency function
    :param flavor: flux flavor
    :param nsip: nsi parameters
    :param op: oscillation parameters
    :return: scattering rates per nucleus
    """
    deno = 2 * np.sqrt(2) * gf * (2 * me * er + nsip.mz ** 2)
    if flavor[0] == 'e':
        epls = (0.5 + ssw + nsip.gel['ee'] / deno) ** 2 + (nsip.gel['em'] / deno) ** 2 + (nsip.gel['et'] / deno) ** 2
        eprs = (ssw + nsip.ger['ee'] / deno) ** 2 + (nsip.ger['em'] / deno) ** 2 + (nsip.ger['et'] / deno) ** 2
        eplr = (0.5 + ssw + nsip.gel['ee'] / deno) * (ssw + nsip.ger['ee'] / deno) + \
            0.5 * (np.real(nsip.gel['em'] / deno) * np.real(nsip.ger['em'] / deno) + np.imag(nsip.gel['em'] / deno) * np.imag(nsip.ger['em'] / deno)) + \
            0.5 * (np.real(nsip.gel['et'] / deno) * np.real(nsip.ger['et'] / deno) + np.imag(nsip.gel['et'] / deno) * np.imag(nsip.ger['et'] / deno))
    elif flavor[0] == 'm':
        epls = (-0.5 + ssw + nsip.gel['mm'] / deno) ** 2 + (nsip.gel['em'] / deno) ** 2 + (nsip.gel['mt'] / deno) ** 2
        eprs = (ssw + nsip.ger['mm'] / deno) ** 2 + (nsip.ger['em'] / deno) ** 2 + (nsip.ger['mt'] / deno) ** 2
        eplr = (-0.5 + ssw + nsip.gel['mm'] / deno) * (ssw + nsip.ger['mm'] / deno) + \
            0.5 * (np.real(nsip.gel['em'] / deno) * np.real(nsip.ger['em'] / deno) + np.imag(nsip.gel['em'] / deno) * np.imag(nsip.ger['em'] / deno)) + \
            0.5 * (np.real(nsip.gel['mt'] / deno) * np.real(nsip.ger['mt'] / deno) + np.imag(nsip.gel['mt'] / deno) * np.imag(nsip.ger['mt'] / deno))
    elif flavor[0] == 't':
        epls = (-0.5 + ssw + nsip.gel['tt'] / deno) ** 2 + (nsip.gel['mt'] / deno) ** 2 + (nsip.gel['et'] / deno) ** 2
        eprs = (ssw + nsip.ger['tt'] / deno) ** 2 + (nsip.ger['mt'] / deno) ** 2 + (nsip.ger['et'] / deno) ** 2
        eplr = (-0.5 + ssw + nsip.gel['tt'] / deno) * (ssw + nsip.ger['tt'] / deno) + \
            0.5 * (np.real(nsip.gel['et'] / deno) * np.real(nsip.ger['et'] / deno) + np.imag(nsip.gel['et'] / deno) * np.imag(nsip.ger['et'] / deno)) + \
            0.5 * (np.real(nsip.gel['mt'] / deno) * np.real(nsip.ger['mt'] / deno) + np.imag(nsip.gel['mt'] / deno) * np.imag(nsip.ger['mt'] / deno))
    else:
        raise Exception('No such neutrino flavor!')
    if flavor[-1] == 'r':
        temp = epls
        epls = eprs
        eprs = temp
    if efficiency is not None:
        return np.dot(2 / np.pi * (gf ** 2) * me * det.z *
                   (epls * fx.fint(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                    eprs * (fx.fint(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                            2 * er * fx.fintinv(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                            (er ** 2) * fx.fintinvs(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)) -
                    eplr * me * er * fx.fintinvs(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)), det.frac) * efficiency(er)
    else:
        return np.dot(2 / np.pi * (gf ** 2) * me * det.z *
                   (epls * fx.fint(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                    eprs * (fx.fint(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) -
                            2 * er * fx.fintinv(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs) +
                            (er ** 2) * fx.fintinvs(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)) -
                    eplr * me * er * fx.fintinvs(er, me, flavor=flavor, f=f, epsi=nsip, op=op, **kwargs)), det.frac)


def binned_events_nucleus(era, erb, expo, det: Detector, fx: Flux, nsip: NSIparameters, efficiency=None, f=None,
                          flavor='e', op=oscillation_parameters(), q2=False, **kwargs):
    """
    :return: number of nucleus recoil events in the bin [era, erb]
    """
    def rates(er):
        return rates_nucleus(er, det, fx, efficiency=efficiency, f=f, nsip=nsip, flavor=flavor, op=op, q2=q2, **kwargs)
    return quad(rates, era, erb)[0] * \
        expo * mev_per_kg * 24 * 60 * 60 / np.dot(det.m, det.frac)


def binned_events_electron(era, erb, expo, det: Detector, fx: Flux, nsip: NSIparameters, efficiency=None, f=None,
                           flavor='e', op=oscillation_parameters(), **kwargs):
    """
    :return: number of electron recoil events in the bin [era, erb]
    """
    def rates(er):
        return rates_electron(er, det, fx, efficiency=efficiency, f=f, nsip=nsip, flavor=flavor, op=op, **kwargs)
    return quad(rates, era, erb)[0] * \
        expo * mev_per_kg * 24 * 60 * 60 / np.dot(det.m, det.frac)


class NSIEventsGen:
    def __init__(self, flux: Flux, detector: Detector, expo: float, target='nucleus', nsi_param=NSIparameters(),
                 osci_param=oscillation_parameters(), osci_func=None, formfactsq=formfsquared, q2form=False, efficiency=None):
        self.flux = flux
        self.detector = detector
        self.expo = expo
        self.target = target
        self.nsi_param = nsi_param
        self.osci_param = osci_param
        self.formfactsq = formfactsq
        self.q2form = q2form
        self.efficiency = efficiency
        self.osci_func = osci_func

    def rates(self, er, flavor='e', **kwargs):
        if self.target == 'nucleus':
            return rates_nucleus(er, self.detector, self.flux, efficiency=self.efficiency, f=self.osci_func,
                                 nsip=self.nsi_param, flavor=flavor, op=self.osci_param, ffs=self.formfactsq, q2=self.q2form, **kwargs)
        elif self.target == 'electron':
            return rates_electron(er, self.detector, self.flux, efficiency=self.efficiency, f=self.osci_func,
                                  nsip=self.nsi_param, flavor=flavor, op=self.osci_param, **kwargs)
        else:
            raise Exception('Target should be either nucleus or electron!')

    def events(self, ea, eb, flavor='e', **kwargs):
        if self.target == 'nucleus':
            return binned_events_nucleus(ea, eb, self.expo, self.detector, self.flux, nsip=self.nsi_param, flavor=flavor,
                                         efficiency=self.efficiency, f=self.osci_func, op=self.osci_param, q2=self.q2form, **kwargs)
        elif self.target == 'electron':
            return binned_events_electron(ea, eb, self.expo, self.detector, self.flux, nsip=self.nsi_param,
                                          flavor=flavor, op=self.osci_param, efficiency=self.efficiency, **kwargs)
        else:
            return Exception('Target should be either nucleus or electron!')


def rates_dm_electron(er, det: Detector, fx: DMFlux, mediator_mass=None,
                      epsilon=None, efficiency=None, smear=False, **kwargs):
    if mediator_mass is None:
        mediator_mass = fx.dp_mass
    if epsilon is None:
        epsilon = fx.epsi_quark

    def rates(err):
        if err < det.er_min:
            return 0
        prefactor = epsilon**2 / (4*np.pi * (2*me*err+mediator_mass**2)**2)
        res = prefactor * np.dot(det.frac, det.z * (2*me*fx.fint2(err, me)
                                            - 2*me*err*fx.fint1(err, me)
                                            - err*(me**2-fx.dm_m**2)*fx.fint(err, me)
                                            + err**2*me*fx.fint(err, me)))
        if efficiency is not None:
            return res * efficiency(err)
        else:
            return res

    if not smear:
        return rates(er)
    else:
        def func(pep):
            pe_per_mev = 0.0878 * 13.348 * 1000
            return rates(pep/pe_per_mev) * _poisson(er*pe_per_mev, pep)
        return quad(func, 0, 60)[0]


def rates_dm_nucleus(er, det: Detector, fx: DMFlux, mediator_mass=None, alpha_D=None, kappa=None, efficiency=None, smear=False, scattering='elastic' ,**kwargs):
    """
    calculating dark matter scattering rates per nucleus
    :param er: recoil energy in MeV
    :param det: detector
    :param fx: dark matter flux
    :param mediator_mass: mediator mass in MeV
    :param epsilon: mediator to quark coupling multiply by mediator to dark matter coupling
    :param efficiency: efficiency function
    :param scattering: elastic or inelastic (GT)
    :return: dark matter scattering rates per nucleus
    """
    if mediator_mass is None:
        mediator_mass = fx.dp_mass
    if alpha_D is None:
        kappa = fx.epsi_quark

    ffsquared = HelmFormFactor(detector=det)

    def rates(err):
        if err < det.er_min:
            return 0

        if scattering == 'elastic':
            res = np.dot(det.frac, kAlpha * 4*np.pi*alpha_D*kappa**2 * det.z**2 *
                      (2*det.m*fx.fint2(err, det.m) - 2*det.m*err*fx.fint1(err, det.m) -
                       (det.m**2*err-fx.dm_m**2*err)*fx.fint(err, det.m) +
                        err**2*det.m*fx.fint(err, det.m)) / ((2*det.m*err+mediator_mass**2)**2) *
                      formfsquared(np.sqrt(2*err*det.m), **kwargs))

        # kAlpha * 4*np.pi*alpha_D*kappa**2 == e^2 gD^2 eps^2 / 4 pi
        elif scattering == 'inelastic' or scattering == 'GT':
            lldot = 0
            l33 = 0
            total_gt_strength = 0

            res = np.dot(det.frac, kAlpha * 4*np.pi*alpha_D*kappa**2 *
                      (2*det.m*fx.fint2(err, det.m) - 2*det.m*err*fx.fint1(err, det.m) -
                       (det.m**2*err-fx.dm_m**2*err)*fx.fint(err, det.m) +
                        err**2*det.m*fx.fint(err, det.m)) / ((2*det.m*err+mediator_mass**2)**2) )

        if efficiency is not None:
            return res * efficiency(err)
        else:
            return res

    def rates_scalar(err):
        if err < det.er_min:
            return 0
        res = np.dot(det.frac, kAlpha * np.pi*alpha_D*kappa**2 * det.z**2 *
                      (4*det.m*fx.fint2(err, det.m)
                       +(2*fx.dm_m**2 - 4*det.m*err) * fx.fint1(err, det.m) -
                       (det.m*err**2 - err*fx.dm_m**2 - 0.25*fx.dm_m**4 / det.m)*fx.fint(err, det.m)) / ((2*det.m*err+mediator_mass**2)**2) *
                      ffsquared(np.sqrt(2*err*det.m)))
        if efficiency is not None:
            return res * efficiency(err)
        else:
            return res

    if not smear:
        return rates(er)
    else:
        def func(pep):
            pe_per_mev = 0.0878 * 13.348 * 1000
            return rates(pep/pe_per_mev) * _poisson(er*pe_per_mev, pep)
        return quad(func, 0, 60)[0]


def binned_events_dm(era, erb, expo, det: Detector, fx: DMFlux, mediator_mass=None,
                     alpha_D=None, kappa=None, efficiency=None, smear=False, channel="nucleus", **kwargs):
    """
    :return: number of nucleus recoil events in the bin [era, erb]
    """
    def rN(er):
        return rates_dm_nucleus(er, det, fx, mediator_mass, alpha_D, kappa, efficiency, smear, **kwargs)
    def rN_GT(er):
        return rates_dm_nucleus(er, det, fx, mediator_mass, alpha_D, kappa, efficiency, smear, 'GT', **kwargs)
    def rE(er):
        return rates_dm_electron(er, det, fx, mediator_mass, epsilon, efficiency, smear, **kwargs)

    if channel == "electron":
        return quad(rE, era, erb)[0] * expo * mev_per_kg * 24 * 60 * 60 / np.dot(det.m, det.frac)
    elif channel == "nucleus":
        return quad(rN, era, erb)[0] * expo * mev_per_kg * 24 * 60 * 60 / np.dot(det.m, det.frac)
    elif channel == "nucleus-GT":
        return quad(rN_GT, era, erb)[0] * expo * mev_per_kg * 24 * 60 * 60 / np.dot(det.m, det.frac)
    else:
        return Exception('Channel should be either electron or nucleus!')


class DmEventsGen:
    """
    Dark matter events generator for COHERENT
    """
    def __init__(self, dark_photon_mass, life_time, dark_matter_mass, expo=4466, detector_type='csi',
                 detector_distance=19.3, pot_mu=0.75, pot_sigma=0.25, size=100000, smear=False, rn=4.7,
                 mono_energy=None):
        self.dp_mass = dark_photon_mass
        self.tau = life_time
        self.dm_mass = dark_matter_mass
        self.det_dist = detector_distance
        self.mu = pot_mu
        self.sigma = pot_sigma
        self.size = size
        self.det = Detector(detector_type)
        self.fx = None
        self.expo = expo
        self.smear = smear
        self.rn = rn
        self.e_chi = mono_energy
        if mono_energy is not None:
            self.fx = DMFlux(self.dp_mass, self.tau, 1, self.dm_mass, self.det_dist,
                             self.mu, self.sigma, self.size, self.e_chi)
        else:
            self.fx = DMFlux(self.dp_mass, self.tau, 1, self.dm_mass, self.det_dist,
                             self.mu, self.sigma, self.size)

    def generate_flux(self):
        self.fx = DMFlux(self.dp_mass, self.tau, 1, self.dm_mass, self.det_dist,
                         self.mu, self.sigma, self.size)

    def set_dark_photon_mass(self, dark_photon_mass):
        self.dp_mass = dark_photon_mass
        self.generate_flux()

    def set_life_time(self, life_time):
        self.tau = life_time
        self.generate_flux()

    def set_dark_matter_mass(self, dark_matter_mass):
        self.dm_mass = dark_matter_mass
        self.generate_flux()

    def set_detector_distance(self, detector_distance):
        self.det_dist = detector_distance
        self.generate_flux()

    def set_pot_mu(self, pot_mu):
        self.mu = pot_mu
        self.generate_flux()

    def set_pot_sigma(self, pot_sigma):
        self.sigma = pot_sigma
        self.generate_flux()

    def set_size(self, size):
        self.size = size
        self.generate_flux()

    def events(self, mediator_mass, alpha_D, kappa, energy_edges, timing_edges, channel="nucleus", efficiency=eff_coherent, efficiency_timing=None):
        """
        generate events according to the time and energy in measured data
        :param mediator_mass: mediator mass
        :param epsilon: mediator coupling to quark multiply by mediator coupling to dark matter
        :param n_meas: measured data
        :param energy_edges: energy bin edges [MeV]. Must be equally (linearly) spaced.
        :param timing_edges: time bin edges [microseconds]. Must be equally (linearly) spaced.
        :return: predicted number of event according to the time and energy in the measured data
        """
        energy_bins = (energy_edges[:-1] + energy_edges[1:]) / 2
        timing_bins = (timing_edges[:-1] + timing_edges[1:]) / 2
        n_meas = np.zeros((energy_bins.shape[0] * len(timing_bins), 2))
        n_dm = np.zeros(n_meas.shape[0])

        flat_index = 0
        for i in range(0, energy_bins.shape[0]):
            for j in range(0, timing_bins.shape[0]):
                n_meas[flat_index, 0] = energy_bins[i]
                n_meas[flat_index, 1] = timing_bins[j]
                flat_index += 1

        if len(self.fx.timing) == 0:
            return n_dm, n_meas

        t_hist = np.histogram(self.fx.timing, bins=timing_edges)
        probs = t_hist[0] / len(self.fx.timing) # time distribution PDF
        for i in range(n_meas.shape[0]):
            ti = np.where(timing_bins == n_meas[i,1])
            ei = np.where(energy_bins == n_meas[i,0])

            if efficiency_timing is not None:
                effT = efficiency_timing(timing_bins[ti])
            else:
                effT = 1

            n_dm[i] = binned_events_dm(energy_edges[ei[0]], energy_edges[ei[0]+1], self.expo,
                                       self.det, self.fx, mediator_mass, alpha_D, kappa, efficiency,
                                       self.smear, rn=self.rn, channel=channel) * probs[ti] * effT
        return n_dm, n_meas


class HelmFormFactor:
    """
    square of the form factor!
    """
    def __init__(self, rn=4.7, detector=None):
        self.rn = rn
        if detector is not None:
            self.rn = 4.7*((detector.n[0]+detector.z[0])/133)**(1/3)

    def __call__(self, q):
        r = self.rn * (10 ** -15) / meter_by_mev
        s = 0.9 * (10 ** -15) / meter_by_mev
        r0 = np.sqrt(5 / 3 * (r ** 2) - 5 * (s ** 2))
        return (3 * spherical_jn(1, q * r0) / (q * r0) * np.exp((-(q * s) ** 2) / 2)) ** 2

    def change_parameters(self, rn=None):
        self.rn = rn if rn is not None else self.rn


def _inv(ev):
    return 1/ev


def _invs(ev):
    return 1/ev**2



# Event generator base class.
# All other methods for detection rates should be children of this class.
class EventGenerator:
    def __init__(self, detector: Detector, flux, cross_section, efficiency=1.0, xedges=None,
                 yedges=None, zedges=None, exposure=1.0):
        self.det = detector
        self.xs = cross_section
        self.xedges = xedges
        self.yedges = yedges
        self.zedges = zedges
        if xedges is not None:
            self.xedges = (xedges[:-1] + xedges[1:])/2
        if yedges is not None:
            self.yedges = (yedges[:-1] + yedges[1:])/2
        if zedges is not None:
            self.zedges = (zedges[:-1] + zedges[1:])/2
        self.eff = efficiency
        self.exp = exposure  # must be in kg-days

    def scale(self):
        return self.exp * mev_per_kg * 24 * 3600 / np.dot(det.m, det.frac)

    def events1D(self):
        # rates is not a scalar, xedges not specified (raise error)

        try:
            if hasattr(self.xs, '__call__'):
                assert(xedges is not None)
                return quad(self.xs, xedges[0], xedges[-1])[0] * self.scale()
            else:
                return self.scale() * self.rates * self.eff
        except:
            raise Exception("You must supply xedges (e.g. [first, last]) bin boundaries.")



# Event generator base class for continuous rates function.
class EventGeneratorContinuous:
    def __init__(self, detector: Detector, rates, xedges=None, yedges=None, zedges=None,
                 efficiency=1.0, exposure=1.0):
        self.det = detector
        self.rates = rates
        self.rates_is_func = hasattr(self.rates, '__call__')
        self.xedges = xedges
        self.yedges = yedges
        self.zedges = zedges
        if xedges is not None:
            self.xedges = (xedges[:-1] + xedges[1:])/2
        if yedges is not None:
            self.yedges = (yedges[:-1] + yedges[1:])/2
        if zedges is not None:
            self.zedges = (zedges[:-1] + zedges[1:])/2
        self.eff = efficiency
        self.exp = exposure  # must be in kg-days

    def scale(self):
        return self.exp * mev_per_kg * 24 * 3600 / np.dot(det.m, det.frac)

    # 0-D event rate
    def count(self):
        # rates is not a scalar, xedges not specified (raise error)
        try:
            if self.rates_is_func:
                assert(xedges is not None)
                return quad(self.rates, xedges[0], xedges[-1])[0] * self.scale()
            else:
                return self.scale() * self.rates * self.eff
        except:
            raise Exception("You must supply xedges (e.g. [first, last]) bin boundaries.")

    # 1-D event rate
    def events1D(self):
        try:
            assert(self.rates_is_func)
        except:
            raise AttributeError("self.rates must be function with 1st argument integrable. \n"
                                 "Use count() to get the number of events with flat rate function.")
        return [self.scale()*self.eff*quad(self.rates, xedges[i], xedges[i+1])[0] \
                for i in range(len(self.xedges)-1)]

    # 2d events, e.g. energy, time
    def events2D(self):
        pass

    # 3d events, e.g. energy, time, theta
    def events3D(self):
        pass







class NeutrinoNucleusElasticVector:
    def __init__(self, nsi_parameters: NSIparameters, form_factor_square=HelmFormFactor()):
        self.nsi_parameters = nsi_parameters
        self.form_factor_square = form_factor_square

    def rates(self, er, flavor, flux: NeutrinoFlux, detector: Detector):
        rho = 1.0086
        knu = 0.9978
        lul = -0.0031
        ldl = -0.0025
        ldr = 7.5e-5
        lur = ldr / 2
        epu = self.nsi_parameters.eu()
        epd = self.nsi_parameters.ed()
        scale = 1
        if self.nsi_parameters.mz != 0:
            scale = self.nsi_parameters.mz**2 / (self.nsi_parameters.mz**2 + 2*detector.m*er)
        qvs = 0
        if flavor[0] == 'e':
            qvs = (0.5 * detector.z * (rho*(0.5 - 2*knu*ssw)+2*lul+2*lur+ldl+ldr+2*epu[0, 0]*scale+epd[0, 0]*scale) +
                   0.5*detector.n*(-0.5*rho + lul + lur + 2*ldl + 2*ldr + epu[0, 0]*scale + 2*epd[0, 0]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[0, 1]*scale + epd[0, 1]*scale) + 0.5*detector.n*(epu[0, 1]*scale + 2*epd[0, 1]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[0, 2]*scale + epd[0, 2]*scale) + 0.5*detector.n*(epu[0, 2]*scale + 2*epd[0, 2]*scale)) ** 2
        if flavor[0] == 'm':
            qvs = (0.5 * detector.z * (rho*(0.5 - 2*knu*ssw)+2*lul+2*lur+ldl+ldr+2*epu[1, 1]*scale+epd[1, 1]*scale) +
                   0.5*detector.n*(-0.5*rho + lul + lur + 2*ldl + 2*ldr + epu[1, 1]*scale + 2*epd[1, 1]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[1, 0]*scale + epd[1, 0]*scale) + 0.5*detector.n*(epu[1, 0]*scale + 2*epd[1, 0]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[1, 2]*scale + epd[1, 2]*scale) + 0.5*detector.n*(epu[1, 2]*scale + 2*epd[1, 2]*scale)) ** 2
        if flavor[0] == 't':
            qvs = (0.5 * detector.z * (rho*(0.5 - 2*knu*ssw)+2*lul+2*lur+ldl+ldr+2*epu[2, 2]*scale+epd[2, 2]*scale) +
                   0.5*detector.n*(-0.5*rho + lul + lur + 2*ldl + 2*ldr + epu[2, 2]*scale + 2*epd[2, 2]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[2, 0]*scale + epd[2, 0]*scale) + 0.5*detector.n*(epu[2, 0]*scale + 2*epd[2, 0]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[2, 1]*scale + epd[2, 1]*scale) + 0.5*detector.n*(epu[2, 1]*scale + 2*epd[2, 1]*scale)) ** 2
        fint = np.zeros(detector.iso)
        fintinv = np.zeros(detector.iso)
        fintinvs = np.zeros(detector.iso)
        emin = 0.5 * (np.sqrt(er ** 2 + 2 * er * detector.m) + er)
        for i in range(detector.iso):
            fint[i] = flux.integrate(emin[i], flux.ev_max, flavor)
            fintinv[i] = flux.integrate(emin[i], flux.ev_max, flavor, weight_function=_inv)
            fintinvs[i] = flux.integrate(emin[i], flux.ev_max, flavor, weight_function=_invs)
        res = np.dot(2 / np.pi * (gf ** 2) * (2 * fint - 2 * er * fintinv + er * er * fintinvs - detector.m * er * fintinvs) *
                     detector.m * qvs * self.form_factor_square(np.sqrt(2 * detector.m * er)), detector.frac)
        if detector.efficiency is not None:
            res *= detector.efficiency(er)
        return res

    def events(self, ea, eb, flavor, flux: NeutrinoFlux, detector: Detector, exposure):
        def func(er):
            return self.rates(er, flavor, flux, detector)
            # return effE(keV2PE(er)) * self.rates(er, flavor, flux, detector)
        return quad(func, ea, eb)[0] * exposure * mev_per_kg * 24 * 60 * 60 / np.dot(detector.m, detector.frac)

    def change_parameters(self):
        pass


class NeutrinoElectronElasticVector:
    def __init__(self, nsi_parameters: NSIparameters):
        self.nsi_parameters = nsi_parameters

    def rates(self, er, flavor, flux: NeutrinoFlux, detector: Detector):
        epel = self.nsi_parameters.eel()
        eper = self.nsi_parameters.eer()
        scale = 1
        if self.nsi_parameters.mz != 0:
            scale = self.nsi_parameters.mz**2 / (self.nsi_parameters.mz**2 + 2*me*er)
        epls = 0
        eprs = 0
        eplr = 0
        if flavor[0] == 'e':
            epls = (0.5 + ssw + epel[0, 0] * scale) ** 2 + np.abs(epel[0, 1] * scale) ** 2 + np.abs(epel[0, 2] * scale) ** 2
            eprs = (ssw + eper[0, 0] * scale) ** 2 + np.abs(eper[0, 1] * scale) ** 2 + np.abs(eper[0, 2] * scale) ** 2
            eplr = (0.5 + ssw + epel[0, 0] * scale) * (ssw + eper[0, 0] * scale) + \
                0.5 * (np.real(epel[0, 1] * scale) * np.real(eper[0, 1] * scale) +
                       np.imag(epel[0, 1] * scale) * np.imag(eper[0, 1] * scale)) + \
                0.5 * (np.real(epel[0, 2] * scale) * np.real(eper[0, 2] * scale) +
                       np.imag(epel[0, 2] * scale) * np.imag(eper[0, 2] * scale))
        elif flavor[0] == 'm':
            epls = (-0.5 + ssw + epel[1, 1] * scale) ** 2 + np.abs(epel[1, 0] * scale) ** 2 + np.abs(epel[1, 2] * scale) ** 2
            eprs = (ssw + eper[1, 1] * scale) ** 2 + np.abs(eper[1, 0] * scale) ** 2 + np.abs(eper[1, 2] * scale) ** 2
            eplr = (-0.5 + ssw + epel[1, 1] * scale) * (ssw + eper[1, 1] * scale) + \
                0.5 * (np.real(epel[1, 0] * scale) * np.real(eper[1, 0] * scale) +
                       np.imag(epel[1, 0] * scale) * np.imag(eper[1, 0] * scale)) + \
                0.5 * (np.real(epel[1, 2] * scale) * np.real(eper[1, 2] * scale) +
                       np.imag(epel[1, 2] * scale) * np.imag(eper[1, 2] * scale))
        elif flavor[0] == 't':
            epls = (-0.5 + ssw + epel[2, 2] * scale) ** 2 + np.abs(epel[2, 1] * scale) ** 2 + np.abs(epel[2, 0] * scale) ** 2
            eprs = (ssw + eper[2, 2] * scale) ** 2 + np.abs(eper[2, 1] * scale) ** 2 + np.abs(eper[2, 0] * scale) ** 2
            eplr = (-0.5 + ssw + epel[2, 2] * scale) * (ssw + eper[2, 2] * scale) + \
                0.5 * (np.real(epel[2, 0] * scale) * np.real(eper[2, 0] * scale) +
                       np.imag(epel[2, 0] * scale) * np.imag(eper[2, 0] * scale)) + \
                0.5 * (np.real(epel[2, 1] * scale) * np.real(eper[2, 1] * scale) +
                       np.imag(epel[2, 1] * scale) * np.imag(eper[2, 1] * scale))
        emin = 0.5 * (np.sqrt(er ** 2 + 2 * er * me) + er)
        fint = flux.integrate(emin, flux.ev_max, flavor)
        fintinv = flux.integrate(emin, flux.ev_max, flavor, weight_function=_inv)
        fintinvs = flux.integrate(emin, flux.ev_max, flavor, weight_function=_invs)
        if flavor[-1] == 'r':
            tmp = epls
            epls = eprs
            eprs = tmp
        res = np.dot(2 / np.pi * (gf ** 2) * me * detector.z *
                     (epls * fint + eprs * (fint - 2 * er * fintinv + (er ** 2) * fintinvs) - eplr * me * er * fintinvs), detector.frac)
        if detector.efficiency is not None:
            res *= detector.efficiency(er)
        return res

    def events(self, ea, eb, flavor, flux: NeutrinoFlux, detector: Detector, exposure):
        def func(er):
            return self.rates(er, flavor, flux, detector)
        return quad(func, ea, eb)[0] * exposure * mev_per_kg * 24 * 60 * 60 / np.dot(detector.m, detector.frac)

    def change_parameters(self):
        pass


# Charged Current Quasi-Elastic (CCQE) cross-section, assuming no CC NSI. Follows Bodek, Budd, Christy [1106.0340].
class NeutrinoNucleonCCQE:
    def __init__(self, flavor, flux: NeutrinoFlux):
        self.flavor = flavor
        self.flux = flux
        self.FastXS = np.vectorize(self.rates)

    def rates(self, ev, flavor='e', masq=axial_mass**2):
        m_lepton = me
        m_nucleon = massofn
        xi = 4.706  # Difference between proton and neutron magnetic moments.
        sign = -1

        if flavor == "mu" or flavor == "mubar":
            m_lepton = mmu
        if flavor == "tau" or flavor == "taubar":
            m_lepton = mtau

        if flavor == "ebar" or flavor == "mubar" or flavor == "taubar":
            sign = 1
            m_nucleon = massofp

        def dsigma(qsq):
            tau = qsq / (4 * m_nucleon ** 2)
            GD = (1 / (1 + qsq / 710000) ** 2)  # Dipole form factor with vector mass.
            TE = np.sqrt(1 + (6e-6 * qsq) * np.exp(-qsq / 350000))  # Transverse Enhancement of the magnetic dipole.

            FA = -1.267 / (1 + (qsq / masq)) ** 2  # Axial form factor.
            Fp = (2 * FA * (m_nucleon) ** 2) / (massofpi0 ** 2 + qsq)  # Pion dipole form factor (only relevant for low ev).
            F1 = GD * ((1 + xi * tau * TE) / (1 + tau))  # First nuclear form factor in dipole approximation.
            F2 = GD * (xi * TE - 1) / (1 + tau)  # Second nuclear form factor in dipole approximation.

            # A, B, and C are the vector, pseudoscalar, and axial vector terms, respectively.
            A = ((m_lepton ** 2 + qsq) / m_nucleon ** 2) * (
                    (1 + tau) * FA ** 2 - (1 - tau) * F1 ** 2 + tau * (1 - tau) * (F2) ** 2 + 4 * tau * F1 * F2
                    - 0.25 * ((m_lepton / m_nucleon) ** 2) * ((F1 + F2) ** 2 + (FA + 2 * Fp) ** 2
                                                              - 4 * (tau + 1) * Fp ** 2))
            B = 4 * tau * (F1 + F2) * FA
            C = 0.25 * (FA ** 2 + F1 ** 2 + tau * (F2) ** 2)

            return ((1 / (8 * np.pi)) * (gf * cabibbo * m_nucleon / ev) ** 2) * \
                   (A + sign * B * ((4 * m_nucleon * ev - qsq - m_lepton ** 2) / (m_nucleon) ** 2)
                    + C * ((4 * m_nucleon * ev - qsq - m_lepton ** 2) / (m_nucleon) ** 2) ** 2)

        sqts = np.sqrt(m_nucleon ** 2 + 2 * m_nucleon * ev)
        E_l = (sqts ** 2 + m_lepton ** 2 - m_nucleon ** 2) / (2 * sqts)
        if E_l ** 2 < m_lepton ** 2:
            return 0
        q2min = -m_lepton ** 2 + (sqts ** 2 - m_nucleon ** 2) / (sqts) * \
                (E_l - np.sqrt(E_l ** 2 - m_lepton ** 2))
        q2max = -m_lepton ** 2 + (sqts ** 2 - m_nucleon ** 2) / (sqts) * \
                (E_l + np.sqrt(E_l ** 2 - m_lepton ** 2))

        return quad(dsigma, q2min, q2max)[0]

    def events(self, eva, evb, detector: Detector, exposure):
        nucleons = detector.z  # convert the per-nucleon cross section into total cross section.
        if self.flavor == 'ebar' or self.flavor == 'mubar' or self.flavor == 'taubar':
            nucleons = detector.n

        return nucleons * self.flux.integrate(eva, evb, self.flavor, weight_function=self.FastXS) * \
               exposure * mev_per_kg * 24 *60 * 60 / np.dot(detector.m, detector.frac)

    def change_parameters(self):
        pass


class DMNucleusElasticVector:
    def __init__(self, epsilon_dm, epsilon_q, mediator_mass, form_factor_square=HelmFormFactor()):
        self.epsilon_dm = epsilon_dm
        self.epsilon_q = epsilon_q
        self.mediator_mass = mediator_mass
        self.form_factor_square = form_factor_square

    def rates(self, er, flux, detector: Detector):
        f0 = np.zeros(detector.iso)
        f1 = np.zeros(detector.iso)
        f2 = np.zeros(detector.iso)
        emin = 0.5 * (np.sqrt((er**2*detector.m+2*er*detector.m**2+2*er*flux.dm_m**2+4*detector.m*flux.dm_m**2)/detector.m) + er)
        for i in range(detector.iso):
            f0[i] = flux.integrate(emin[i], flux.ev_max, weight_function=flux.f0)
            f1[i] = flux.integrate(emin[i], flux.ev_max, weight_function=flux.f1)
            f2[i] = flux.integrate(emin[i], flux.ev_max, weight_function=flux.f2)
        res = np.dot(detector.frac, e_charge**4 * self.epsilon_dm**2 * self.epsilon_q**2 * detector.z**2 *
                      (2*detector.m*f2 - (er)*2*detector.m*f1 - (detector.m**2*er+flux.dm_m**2*er)*f0 + er**2*detector.m*f0) /
                          (4*np.pi*(2*detector.m*er+self.mediator_mass**2)**2) * self.form_factor_square(np.sqrt(2*detector.m*er)))
        if detector.efficiency is not None:
            res *= detector.efficiency(er)
        return res

    def events(self, ea, eb, flux, detector: Detector, exposure):
        def func(er):
            return self.rates(er, flux, detector)
        return quad(func, ea, eb)[0] * exposure * mev_per_kg * 24 * 60 * 60 / np.dot(detector.m, detector.frac)


class DMNucleusQuasiElasticVector:
    def __init__(self, epsilon_dm, epsilon_q, mediator_mass, dark_matter_mass, form_factor_square=HelmFormFactor()):
        self.epsilon_dm = epsilon_dm
        self.epsilon_q = epsilon_q
        self.mediator_mass = mediator_mass
        self.dm_mass = dark_matter_mass
        self.form_factor_square = form_factor_square
        self.gep0 = 1
        self.aep = [1, 3.253, 1.422, 0.08582, 0.3318, -0.09371, 0.01076]
        self.amp = [1, 3.104, 1.428, 0.1112, -0.006981, 0.0003705, -0.7063 * 1e-5]
        self.amn = [1, 3.043, 0.8548, 0.6806, -0.1287, 0.008912, 0]
        self.gmp0 = 2.793
        self.gmn0 = -1.913
        self.mun = -1.913
        self.aen = 0.942
        self.ben = 4.61
        self.mup = 2.793
        self.mv2 = 0.71

    def gepact(self, q2):
        return self.gep0/(1 + self.aep[1]*q2 + self.aep[2]*pow(q2,2) + self.aep[3]*pow(q2,3)+ self.aep[4]*pow(q2,4)+self.aep[5]*pow(q2,5)+ self.aep[6]*pow(q2,6))

    def dmp03(self, q2):
        return self.gmp0/(1 + self.amp[1]*q2 + self.amp[2]*pow(q2,2) + self.amp[3]*pow(q2,3)+ self.amp[4]*pow(q2,4)+self.amp[5]*pow(q2,5)+ self.amp[6]*pow(q2,6))

    def gen03(self, q2):
        return -self.mun * self.aen * q2/(4*massofn*massofn*1e-6) / (1 + self.ben*q2/(4 * massofn*massofn*1e-6)) / pow((1+ q2/self.mv2),2)

    def gmp03(self, q2):
        return self.gmp0 / (1 + self.amp[1]*q2 + self.amp[2]*pow(q2,2) + self.amp[3]*pow(q2,3)+ self.amp[4]*pow(q2,4)+self.amp[5]*pow(q2,5)+ self.amp[6]*pow(q2,6))

    def gep03(self, q2):
        return self.gmp03(q2)*self.gepact(6)/self.gmp03(6) if q2>= 6 else self.gepact(q2)

    def f2p(self, q2):
        return (self.gmp03(q2)-self.gep03(q2))/(1+q2/(4*pow(massofp*1e-3,2))) if q2<=10 else 0

    def f1p(self, q2):
        return (self.gep03(q2) + q2/(4*pow(massofp*1e-3,2)) * self.gmp03(q2))/(1+q2/(4 * pow(massofp*1e-3,2))) if q2<=10 else 0

    def aa(self, ev, er):
        return 2*massofp*ev*(ev-er)-self.dm_mass**2*er

    def bb(self, ev, er):
        return -er*((2*ev-er)**2-2*massofp*er-4*self.dm_mass**2)

    def cc(self, er):
        return -er*(massofp*er+2*self.dm_mass**2)

    def rates(self, er, flux, detector: Detector):
        emin = 0.5 * (np.sqrt((er**2*massofp+2*er*massofp**2+2*er*flux.dm_m**2+4*massofp*flux.dm_m**2)/massofp) + er)
        f0 = flux.integrate(emin, flux.ev_max, weight_function=flux.f0)
        f1 = flux.integrate(emin, flux.ev_max, weight_function=flux.f1)
        f2 = flux.integrate(emin, flux.ev_max, weight_function=flux.f2)
        ff1p = self.f1p(2*massofp*er*1e-6)
        ff2p = self.f2p(2*massofp*er*1e-6)
        res = np.dot(detector.frac, e_charge**4 * self.epsilon_dm**2 * self.epsilon_q**2 * detector.z**2 *
                     ((ff1p**2*2*massofp+ff2p**2*er)*f2 - (ff1p**2*2*massofp*er+er**2*ff2p**2)*f1 +
                      (-self.dm_mass**2*er*ff1p**2+0.25*ff2p**2*(er**2-2*massofp*er-4*self.dm_mass**2)+ff1p*ff2p*self.cc(er))*f0) /
                          (4*np.pi*(2*massofp*er+self.mediator_mass**2)**2) * self.form_factor_square(np.sqrt(2*detector.m*er)))
        if detector.efficiency is not None:
            res *= detector.efficiency(er)
        return res

    def events(self, ea, eb, flux, detector: Detector, exposure):
        def func(er):
            return self.rates(er, flux, detector)
        return quad(func, ea, eb)[0] * exposure * mev_per_kg * 24 * 60 * 60 / np.dot(detector.m, detector.frac)


class NeutrinoNucleusElasticScalar:
    def __init__(self, cu, cd, ms, form_factor_square=None):
        fpd = 0.0411
        fpu = 0.0208
        fnd = 0.0451
        fnu = 0.0189
        self.form_factor_square = form_factor_square
        self.ms = ms
        self.cp = massofp * (cu * fpu / massofu + cd * fpd / massofd)
        self.cn = massofn * (cu * fnu / massofu + cd * fnd / massofd)
        self.sm_interaction = NeutrinoNucleusElasticVector(NSIparameters())

    def rates(self, er, flavor, flux, detector):
        cn = detector.z * self.cp + detector.n * self.cn
        emin = 0.5 * (np.sqrt(er ** 2 + 2 * er * detector.m) + er)
        fintinvs = np.zeros(detector.iso)
        for i in range(detector.iso):
            fintinvs[i] = flux.integrate(emin[i], flux.ev_max, flavor, weight_function=_invs)
        if self.form_factor_square is not None:
            res = np.dot(1 / (4*np.pi) * detector.m**2 * cn**2 * er / (2*detector.m*er+self.ms**2)**2 * fintinvs * self.form_factor_square(np.sqrt(2*detector.m*er)), detector.frac)
        else:
             res = np.dot(1 / (4*np.pi) * detector.m**2 * cn**2 * er / (2*detector.m*er+self.ms**2)**2 * fintinvs, detector.frac)
        if detector.efficiency is not None:
            res *= detector.efficiency(er)
        return self.sm_interaction.rates(er, flavor, flux, detector) + res

    def events(self, ea, eb, flavor, flux: NeutrinoFlux, detector: Detector, exposure):
        def func(er):
            return self.rates(er, flavor, flux, detector)
        return quad(func, ea, eb)[0] * exposure * mev_per_kg * 24 * 60 * 60 / np.dot(detector.m, detector.frac)


class NeutrinoNucleusElascticVectorQ2:
    def __init__(self, nsi_parameters: NSIparameters, lmd, form_factor_square=HelmFormFactor()):
        self.nsi_parameters = nsi_parameters
        self.form_factor_square = form_factor_square
        self.lmd = lmd

    def rates(self, er, flavor, flux: NeutrinoFlux, detector: Detector):
        rho = 1.0086
        knu = 0.9978
        lul = -0.0031
        ldl = -0.0025
        ldr = 7.5e-5
        lur = ldr / 2
        epu = self.nsi_parameters.eu()
        epd = self.nsi_parameters.ed()
        scale = 1
        if self.nsi_parameters.mz != 0:
            scale = self.nsi_parameters.mz**2 / (self.nsi_parameters.mz**2 + 2*detector.m*er)
        qvs = 0
        if flavor[0] == 'e':
            qvs = (0.5 * detector.z * (rho*(0.5 - 2*knu*ssw)+2*lul+2*lur+ldl+ldr+2*epu[0, 0]*scale+epd[0, 0]*scale) +
                   0.5*detector.n*(-0.5*rho + lul + lur + 2*ldl + 2*ldr + epu[0, 0]*scale + 2*epd[0, 0]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[0, 1]*scale + epd[0, 1]*scale) + 0.5*detector.n*(epu[0, 1]*scale + 2*epd[0, 1]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[0, 2]*scale + epd[0, 2]*scale) + 0.5*detector.n*(epu[0, 2]*scale + 2*epd[0, 2]*scale)) ** 2
        if flavor[0] == 'm':
            qvs = (0.5 * detector.z * (rho*(0.5 - 2*knu*ssw)+2*lul+2*lur+ldl+ldr+2*epu[1, 1]*scale+epd[1, 1]*scale) +
                   0.5*detector.n*(-0.5*rho + lul + lur + 2*ldl + 2*ldr + epu[1, 1]*scale + 2*epd[1, 1]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[1, 0]*scale + epd[1, 0]*scale) + 0.5*detector.n*(epu[1, 0]*scale + 2*epd[1, 0]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[1, 2]*scale + epd[1, 2]*scale) + 0.5*detector.n*(epu[1, 2]*scale + 2*epd[1, 2]*scale)) ** 2
        if flavor[0] == 't':
            qvs = (0.5 * detector.z * (rho*(0.5 - 2*knu*ssw)+2*lul+2*lur+ldl+ldr+2*epu[2, 2]*scale+epd[2, 2]*scale) +
                   0.5*detector.n*(-0.5*rho + lul + lur + 2*ldl + 2*ldr + epu[2, 2]*scale + 2*epd[2, 2]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[2, 0]*scale + epd[2, 0]*scale) + 0.5*detector.n*(epu[2, 0]*scale + 2*epd[2, 0]*scale)) ** 2 + \
                np.abs(0.5*detector.z*(2*epu[2, 1]*scale + epd[2, 1]*scale) + 0.5*detector.n*(epu[2, 1]*scale + 2*epd[2, 1]*scale)) ** 2
        fint = np.zeros(detector.iso)
        fintinv = np.zeros(detector.iso)
        fintinvs = np.zeros(detector.iso)
        emin = 0.5 * (np.sqrt(er ** 2 + 2 * er * detector.m) + er)
        for i in range(detector.iso):
            fint[i] = flux.integrate(emin[i], flux.ev_max, flavor)
            fintinv[i] = flux.integrate(emin[i], flux.ev_max, flavor, weight_function=_inv)
            fintinvs[i] = flux.integrate(emin[i], flux.ev_max, flavor, weight_function=_invs)
        res = np.dot(2 / np.pi * (gf ** 2) * (2 * fint - 2 * er * fintinv + er * er * fintinvs - detector.m * er * fintinvs) *
                     detector.m * qvs * self.form_factor_square(np.sqrt(2 * detector.m * er)) * 2*detector.m*er/self.lmd**2, detector.frac)
        if detector.efficiency is not None:
            res *= detector.efficiency(er)
        return res

    def events(self, ea, eb, flavor, flux: NeutrinoFlux, detector: Detector, exposure):
        def func(er):
            return self.rates(er, flavor, flux, detector)
        return quad(func, ea, eb)[0] * exposure * mev_per_kg * 24 * 60 * 60 / np.dot(detector.m, detector.frac)

    def change_parameters(self):
        pass
