"""
flux related class and functions
"""

from scipy.integrate import quad
import pandas as pd
from numbers import Number

from .helper import LinearInterp, polar_to_cartesian, lorentz_boost, lorentz_matrix, _gaussian
from .oscillation import survival_solar
from .parameters import *


def _invs(ev):
    return 1/ev**2



class FluxBaseContinuous:
    def __init__(self, ev, flux, norm=1):
        self.norm = norm
        self.ev = ev
        self.fx = flux
        self.ev_min = self.ev[0]
        self.ev_max = self.ev[-1]
        self.binw = self.ev[1:] - self.ev[:-1]
        self.precalc = {None: self.binw*(self.fx[1:]+self.fx[:-1])/2}

    def __call__(self, ev):
        if ev == self.ev_min:
            return self.fx[0] * self.norm
        if ev == self.ev_max:
            return self.fx[-1] * self.norm
        if self.ev_min < ev < self.ev_max:
            idx = self.ev.searchsorted(ev)
            l1 = ev - self.ev[idx-1]
            l2 = self.ev[idx] - ev
            h1 = self.fx[idx-1]
            h2 = self.fx[idx]
            return (l1*h2 + l2*h1) / (l1 + l2) * self.norm
        return 0

    def integrate(self, ea, eb, weight_function=None):
        if eb <= ea:
            return 0
        res = 0
        if weight_function not in self.precalc:
            weighted = weight_function(self.ev)*self.fx
            self.precalc[weight_function] = self.binw * (weighted[1:]+weighted[:-1]) / 2
        eb = min(eb, self.ev_max)
        ea = max(ea, self.ev_min)
        idxmin = self.ev.searchsorted(ea, side='right')
        idxmax = self.ev.searchsorted(eb, side='left')
        if idxmin == idxmax:
            l1 = ea - self.ev[idxmin - 1]
            l2 = self.ev[idxmin] - ea
            h1 = self.fx[idxmin - 1] * weight_function(self.ev[idxmin - 1]) \
                if weight_function is not None else self.fx[idxmin - 1]
            h2 = self.fx[idxmin] * weight_function(self.ev[idxmin]) \
                if weight_function is not None else self.fx[idxmin]
            ha = (l1*h2+l2*h1)/(l1+l2)
            l1 = eb - self.ev[idxmax - 1]
            l2 = self.ev[idxmax] - eb
            hb = (l1*h2+l2*h1)/(l1+l2)
            return (ha + hb) * (eb - ea) / 2 * self.norm
        res += np.sum(self.precalc[weight_function][idxmin:idxmax-1])
        l1 = ea - self.ev[idxmin-1]
        l2 = self.ev[idxmin] - ea
        h1 = self.fx[idxmin-1]*weight_function(self.ev[idxmin-1]) \
            if weight_function is not None else self.fx[idxmin-1]
        h2 = self.fx[idxmin]*weight_function(self.ev[idxmin]) \
            if weight_function is not None else self.fx[idxmin]
        res += ((l1*h2+l2*h1)/(l1+l2)+h2)*l2/2
        l1 = eb - self.ev[idxmax - 1]
        l2 = self.ev[idxmax] - eb
        h1 = self.fx[idxmax - 1] * weight_function(self.ev[idxmax - 1]) \
            if weight_function is not None else self.fx[idxmax-1]
        h2 = self.fx[idxmax] * weight_function(self.ev[idxmax]) \
            if weight_function is not None else self.fx[idxmax]
        res += ((l1 * h2 + l2 * h1) / (l1 + l2) + h1) * l1 / 2
        return res * self.norm



class Flux:
    """
    flux class,
    flux at source
    """
    def __init__(self, fl_name, delimiter=',', fl_unc=0):
        """
        initializing flux, can take in user provided flux
        restrictions: user provided data must have 7 columns,
        first column is neutrino energy in MeV,
        other columns are neutrino flux in cm^2/s/MeV, they are enu, munu, taunu, enubar, munubar, taunubar
        :param fl_name: name of the flux or path to the file or array of neutrino flux
        :param delimiter: delimiter of the input file, default is ','
        :param fl_unc: uncertainty of flux
        """
        if isinstance(fl_name, str):
            self.fl_name = fl_name.lower()
        else:
            self.fl_name = 'default'
        if self.fl_name == 'reactor':
            self.evMin = 0.0
            self.evMax = 30  # MeV
            self.flUn = 0.02
            fpers = 3.0921 * (10 ** 16)  # antineutrinos per fission
            nuperf = 6.14102
            self.__nuflux1m = nuperf * fpers / (4 * np.pi) * (meter_by_mev ** 2)
        elif self.fl_name in ['sns', 'prompt', 'delayed']:
            self.evMin = 0
            self.evMax = 52  # MeV
            self.flUn = 0.1
            self.__norm = 1.13 * (10 ** 11) * (meter_by_mev ** 2)
        elif self.fl_name in ['solar', 'b8', 'f17', 'n13', 'o15', 'pp', 'hep']:
            f = np.genfromtxt(pkg_resources.resource_filename(__name__, 'data/' + self.fl_name + '.csv'), delimiter=',')
            self.flUn = 0
            self.evMin = f[0, 0]
            self.evMax = f[-1, 0]
            self.__nue = LinearInterp(f[:, 0], f[:, 1] * ((100 * meter_by_mev) ** 2))
        else:
            if isinstance(fl_name, np.ndarray):
                f = fl_name
            else:
                f = np.genfromtxt(fl_name, delimiter=delimiter)
            self.evMin = np.amin(f[:, 0])
            self.evMax = np.amax(f[:, 0])
            self.flUn = fl_unc
            self.__nue = LinearInterp(f[:, 0], f[:, 1] * ((100 * meter_by_mev) ** 2))
            self.__numu = LinearInterp(f[:, 0], f[:, 2] * ((100 * meter_by_mev) ** 2))
            self.__nutau = LinearInterp(f[:, 0], f[:, 3] * ((100 * meter_by_mev) ** 2))
            self.__nuebar = LinearInterp(f[:, 0], f[:, 4] * ((100 * meter_by_mev) ** 2))
            self.__numubar = LinearInterp(f[:, 0], f[:, 5] * ((100 * meter_by_mev) ** 2))
            self.__nutaubar = LinearInterp(f[:, 0], f[:, 6] * ((100 * meter_by_mev) ** 2))

    def flux(self, ev, flavor='e', f=None, **kwargs):
        """
        differential neutrino flux at the detector, unit MeV^-3*s^-1
        :param ev: nuetrino energy
        :param flavor: nuetrino flavor
        :param f: function that convolves with neutrino flux, typically neutrino oscillation,
                the first argument must be neutrino energy,
                the last two arguments must be input flavor nui and out put flavor nuf
        :param kwargs: parameters with keys that goes into function f
        :return: neutrino flux
        """
        if self.fl_name == 'reactor':
            # Phys.Rev.D39, 11 Vogel
            # 5.323608902707208 = Integrate[Exp[.870 - .16*e - .091*e^2], {e, 0, 10}]
            # reactor neutrino is actually anti-neutrino, this may cause problem when doing electron scattering
            if flavor == 'ebar':
                if f is not None:
                    return np.exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * \
                           f(ev, nui='ebar', nuf=flavor, **kwargs)
                return np.exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * self.__nuflux1m
            elif flavor[-1] == 'r':
                if f is not None:
                    return np.exp(0.87 - 0.16 * ev - 0.091 * (ev ** 2)) / 5.323608902707208 * \
                           f(ev, nui='ebar', nuf=flavor, **kwargs)
                return 0
            else:
                return 0
        elif self.fl_name in ['sns', 'delayed']:
            if flavor[-1] != 'r':
                if f is not None:
                    return (3 * ((ev / (2 / 3 * 52)) ** 2) - 2 * ((ev / (2 / 3 * 52)) ** 3)) / 29.25 * self.__norm * \
                           f(ev, nui='e', nuf=flavor, **kwargs)
                return (3 * ((ev / (2 / 3 * 52)) ** 2) - 2 * ((ev / (2 / 3 * 52)) ** 3)) / 29.25 * self.__norm \
                    if flavor == 'e' else 0
            else:
                if f is not None:
                    return (3 * ((ev / 52) ** 2) - 2 * ((ev / 52) ** 3)) / 26 * self.__norm * \
                           f(ev, nui='mubar', nuf=flavor, **kwargs)
                return (3 * ((ev / 52) ** 2) - 2 * ((ev / 52) ** 3)) / 26 * self.__norm if flavor == 'mubar' else 0
        elif self.fl_name == 'prompt':
            return 0
        elif self.fl_name in ['solar', 'b8', 'f17', 'n13', 'o15', 'pp', 'hep']:
            if flavor[-1] != 'r':
                if f is None:
                    f = survival_solar
                return self.__nue(ev) * f(ev, nui='e', nuf=flavor, **kwargs)
            return 0
        else:
            if flavor[-1] != 'r':
                if f is None:
                    if flavor == 'e':
                        return self.__nue(ev)
                    elif flavor == 'mu':
                        return self.__numu(ev)
                    elif flavor == 'tau':
                        return self.__nutau(ev)
                    else:
                        return 0
                return self.__nue(ev) * f(ev, nui='e', nuf=flavor, **kwargs) + \
                    self.__numu(ev) * f(ev, nui='mu', nuf=flavor, **kwargs) + \
                    self.__nutau(ev) * f(ev, nui='tau', nuf=flavor, **kwargs)
            else:
                if f is None:
                    if flavor == 'ebar':
                        return self.__nuebar(ev)
                    elif flavor == 'mubar':
                        return self.__numubar(ev)
                    elif flavor == 'taubar':
                        return self.__nutaubar(ev)
                    else:
                        return 0
                return self.__nuebar(ev) * f(ev, nui='ebar', nuf=flavor, **kwargs) + \
                    self.__numubar(ev) * f(ev, nui='mubar', nuf=flavor, **kwargs) + \
                    self.__nutaubar(ev) * f(ev, nui='taubar', nuf=flavor, **kwargs)

    def fint(self, er, m, flavor='e', f=None, **kwargs):
        """
        flux integration over the range that can produce a recoil energy er
        :param er: recoil energy
        :param m: mass of the target, it can be an array
        :param flavor: neutrino flavor
        :param f: function that convolves with neutrino flux, typically neutrino oscillation,
                the first argument must be neutrino energy,
                the last two arguments must be input flavor nui and out put flavor nuf
        :param kwargs: parameters with keys that goes into function f
        :return: the result of integration, it can be an array
        """
        emin = 0.5 * (np.sqrt(er ** 2 + 2 * er * m) + er)

        def fx(ev):
            return self.flux(ev, flavor, f, **kwargs)

        if not isinstance(emin, np.ndarray):
            res = quad(fx, emin, self.evMax)[0]  # no need to check range, because outside evMin and evMax are 0
            if self.fl_name == 'solar':
                if f is None:
                    f = survival_solar
                # pep
                res += 1.44e8 * ((100 * meter_by_mev) ** 2) * f(1.439, nui='e', nuf=flavor, **kwargs) \
                    if emin < 1.439 else 0
                # be7
                res += 5e9 * ((100 * meter_by_mev) ** 2) * f(0.8613, nui='e', nuf=flavor, **kwargs) \
                    if emin < 0.8613 else 0
            elif self.fl_name in ['sns', 'prompt']:
                if f is None and flavor == 'mu':
                    # prompt neutrino
                    res += self.__norm if emin <= 29 else 0
                elif f is not None and flavor[-1] != 'r':
                    res += self.__norm * f(29, nui='mu', nuf=flavor, **kwargs) if emin <= 29 else 0
        else:
            res = np.zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(fx, emin[i], self.evMax)[0]
                if self.fl_name == 'solar':
                    if f is None:
                        f = survival_solar
                    # pep
                    res[i] += 1.44e8 * ((100 * meter_by_mev) ** 2) * f(1.439, nui='e', nuf=flavor, **kwargs) \
                        if emin[i] < 1.439 else 0
                    # be7
                    res[i] += 5e9 * ((100 * meter_by_mev) ** 2) * f(0.8613, nui='e', nuf=flavor, **kwargs) \
                        if emin[i] < 0.8613 else 0
                elif self.fl_name in ['sns', 'prompt']:
                    if f is None and flavor == 'mu':
                        # prompt neutrino
                        res[i] += self.__norm if emin[i] <= 29 else 0
                    elif f is not None and flavor[-1] != 'r':
                        res[i] += self.__norm * f(29, nui='mu', nuf=flavor, **kwargs) if emin[i] <= 29 else 0
        return res

    def fintinv(self, er, m, flavor='e', f=None, **kwargs):
        """
        flux/ev integration over the range that can produce a recoil energy er
        :param er: recoil energy
        :param m: mass of the target, it can be an array
        :param flavor: neutrino flavor
        :param f: function that convolves with neutrino flux, typically neutrino oscillation,
                the first argument must be neutrino energy,
                the last two arguments must be input flavor nui and out put flavor nuf
        :param kwargs: parameters with keys that goes into function f
        :return: the result of integration, it can be an array
        """
        emin = 0.5 * (np.sqrt(er ** 2 + 2 * er * m) + er)

        def finv(ev):
            """
            flux/ev
            """
            return self.flux(ev, flavor, f, **kwargs) / ev

        if not isinstance(emin, np.ndarray):
            res = quad(finv, emin, self.evMax)[0]
            if self.fl_name == 'solar':
                if f is None:
                    f = survival_solar
                # pep
                res += 1.44e8 * ((100 * meter_by_mev) ** 2) * f(1.439, nui='e', nuf=flavor, **kwargs) / 1.439 \
                    if emin < 1.439 else 0
                # be7
                res += 5e9 * ((100 * meter_by_mev) ** 2) * f(0.8613, nui='e', nuf=flavor, **kwargs) / 0.8613 \
                    if emin < 0.8613 else 0
            elif self.fl_name in ['sns', 'prompt']:
                if f is None and flavor == 'mu':
                    # prompt neutrino
                    res += self.__norm / 29 if emin <= 29 else 0
                elif f is not None and flavor[-1] != 'r':
                    res += self.__norm / 29 * f(29, nui='mu', nuf=flavor, **kwargs) if emin <= 29 else 0
        else:
            res = np.zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(finv, emin[i], self.evMax)[0]
                if self.fl_name == 'solar':
                    if f is None:
                        f = survival_solar
                    # pep
                    res[i] += 1.44e8 * ((100 * meter_by_mev) ** 2) * f(1.439, nui='e', nuf=flavor, **kwargs) / \
                        1.439 if emin[i] < 1.439 else 0
                    # be7
                    res[i] += 5e9 * ((100 * meter_by_mev) ** 2) * f(0.8613, nui='e', nuf=flavor, **kwargs) / \
                        0.8613 if emin[i] < 0.8613 else 0
                elif self.fl_name in ['sns', 'prompt']:
                    if f is None and flavor == 'mu':
                        # prompt neutrino
                        res[i] += self.__norm / 29 if emin[i] <= 29 else 0
                    elif f is not None and flavor[-1] != 'r':
                        res[i] += self.__norm / 29 * f(29, nui='mu', nuf=flavor, **kwargs) \
                            if emin[i] <= 29 else 0
        return res

    def fintinvs(self, er, m, flavor='e', f=None, **kwargs):
        """
        flux/ev^2 integration over the range that can produce a recoil energy er
        :param er: recoil energy
        :param m: mass of the target, it can be an array
        :param flavor: neutrino flavor
        :param f: function that convolves with neutrino flux, typically neutrino oscillation,
                the first argument must be neutrino energy,
                the last two arguments must be input flavor nui and out put flavor nuf
        :param kwargs: parameters with keys that goes into function f
        :return: the result of integration, it can be an array
        """
        emin = 0.5 * (np.sqrt(er ** 2 + 2 * er * m) + er)

        def finvs(ev):
            """
            flux/ev^2
            """
            return self.flux(ev, flavor, f, **kwargs) / (ev ** 2)

        if not isinstance(emin, np.ndarray):
            res = quad(finvs, emin, self.evMax)[0]
            if self.fl_name == 'solar':
                if f is None:
                    f = survival_solar
                # pep
                res += 1.44e8 * ((100 * meter_by_mev) ** 2) * f(1.439, nui='e', nuf=flavor, **kwargs) / 1.439**2\
                    if emin < 1.439 else 0
                # be7
                res += 5e9 * ((100 * meter_by_mev) ** 2) * f(0.8613, nui='e', nuf=flavor, **kwargs) / 0.8613**2 \
                    if emin < 0.8613 else 0
            elif self.fl_name in ['sns', 'prompt']:
                if f is None and flavor == 'mu':
                    # prompt neutrino
                    res += self.__norm / 29**2 if emin <= 29 else 0
                elif f is not None and flavor[-1] != 'r':
                    res += self.__norm / 29**2 * f(29, nui='mu', nuf=flavor, **kwargs) if emin <= 29 else 0
        else:
            res = np.zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(finvs, emin[i], self.evMax)[0]
                if self.fl_name == 'solar':
                    if f is None:
                        f = survival_solar
                    # pep
                    res[i] += 1.44e8 * ((100 * meter_by_mev) ** 2) * f(1.439, nui='e', nuf=flavor, **kwargs) / \
                        1.439**2 if emin[i] < 1.439 else 0
                    # be7
                    res[i] += 5e9 * ((100 * meter_by_mev) ** 2) * f(0.8613, nui='e', nuf=flavor, **kwargs) / \
                        0.8613**2 if emin[i] < 0.8613 else 0
                elif self.fl_name in ['sns', 'prompt']:
                    if f is None and flavor == 'mu':
                        # prompt neutrino
                        res[i] += self.__norm / 29**2 if emin[i] <= 29 else 0
                    elif f is not None and flavor[-1] != 'r':
                        res[i] += self.__norm / 29**2 * f(29, nui='mu', nuf=flavor, **kwargs) \
                            if emin[i] <= 29 else 0
        return res



class NeutrinoFluxFactory:
    def __init__(self):
        self.flux_list = ['solar', 'solar_b8', 'solar_f17', 'solar_hep', 'solar_n13', 'solar_o15', 'solar_pp',
                          'solar_pep', 'solar_be7', 'coherent', 'coherent_prompt', 'coherent_delayed',
                          'far_beam_nu', 'far_beam_nubar', 'atmospheric','jsns_prompt', 'jsns_delayed', 'jsns_prompt_continuous',
                          'near_beam_nu', 'near_beam_nubar',]

    def print_available(self):
        print(self.flux_list)

    def interp_flux(self, nrg, data):
        return np.interp(nrg, data[:,0], data[:,1])

    def get(self, flux_name, sigma_prompt=0 ,**kwargs):
        if flux_name not in self.flux_list:
            print('flux name not in current list: ', self.flux_list)
            raise Exception('flux not found.')
        if flux_name in ['solar_b8', 'solar_f17', 'solar_hep', 'solar_n13', 'solar_o15', 'solar_pp']:
            f = np.genfromtxt(pkg_resources.resource_filename(__name__, 'data/' + flux_name[6:] + '.csv'), delimiter=',')
            return NeutrinoFlux(continuous_fluxes={'ev': f[:, 0], 'e': f[:, 1]})
        if flux_name == 'solar':
            f = np.genfromtxt(pkg_resources.resource_filename(__name__, 'data/' + flux_name + '.csv'), delimiter=',')
            return NeutrinoFlux(continuous_fluxes={'ev': f[:, 0], 'e': f[:, 1]}, delta_fluxes={'e': [(1.439, 1.44e8), (0.8613, 5e9)]})
        if flux_name == 'pep':
            return NeutrinoFlux(delta_fluxes={'e': [(1.439, 1.44e8), ]})
        if flux_name == 'be7':
            return NeutrinoFlux(delta_fluxes={'e': [(0.8613, 5e9), ]})
        if flux_name == 'coherent':
            def de(evv):
                return (3 * ((evv / (2 / 3 * 52)) ** 2) - 2 * ((evv / (2 / 3 * 52)) ** 3)) / 29.25
            def dmubar(evv):
                return (3 * ((evv / 52) ** 2) - 2 * ((evv / 52) ** 3)) / 26
            ev = np.linspace(0.001, 52, 100)
            return NeutrinoFlux(continuous_fluxes={'ev': ev, 'e': de(ev), 'mubar': dmubar(ev)},
                                delta_fluxes={'mu': [(29, 1)]}, norm=1.13 * (10 ** 7)) ## default unit is /(cm^2*s)
        if flux_name == 'coherent_delayed':
            def de(evv):
                return (3 * ((evv / (2 / 3 * 52)) ** 2) - 2 * ((evv / (2 / 3 * 52)) ** 3)) / 29.25
            def dmubar(evv):
                return (3 * ((evv / 52) ** 2) - 2 * ((evv / 52) ** 3)) / 26
            ev = np.linspace(0.001, 52, kwargs['npoints'] if 'npoints' in kwargs else 100)
            return NeutrinoFlux(continuous_fluxes={'ev': ev, 'e': de(ev), 'mubar': dmubar(ev)}, norm=1.13 * (10 ** 7))
        if flux_name == 'coherent_prompt':
            if sigma_prompt == 0:
                return NeutrinoFlux(delta_fluxes={'mu': [(29, 1)]}, norm=1.13 * (10 ** 7))
            elif isinstance(sigma_prompt,Number) and sigma_prompt != 0:
                def dmu(evv):
                    return _gaussian(evv, 29, sigma_prompt)
                ev = np.linspace(0.001, 52, kwargs['npoints'] if 'npoints' in kwargs else 100)
                return NeutrinoFlux(continuous_fluxes={'ev': ev, 'mu': dmu(ev)}, norm=1.13 * (10 ** 7))
            else:
                raise ValueError('sigma_prompt must be a number')
        if flux_name == 'jsns':
            nu_e = np.genfromtxt(pkg_resources.resource_filename(__name__, "data/jsns2/jsns_nu_e.txt"), delimiter=',')
            nu_mu = np.genfromtxt(pkg_resources.resource_filename(__name__, "data/jsns2/jsns_nu_mu_nodelta.txt"), delimiter=',')
            nubar_mu = np.genfromtxt(pkg_resources.resource_filename(__name__, "data/jsns2/jsns_nubar_mu.txt"), delimiter=',')

            norm_nu_e = quad(self.interp_flux, 0, 300, args=(nu_e,))[0]
            norm_nu_mu = quad(self.interp_flux, 0, 300, args=(nu_mu,))[0]
            norm_nubar_mu = quad(self.interp_flux, 0, 300, args=(nubar_mu,))[0]

            def numuPDF(energy):
                return self.interp_flux(energy, nu_mu) / norm_nu_mu

            def nuePDF(energy):
                return self.interp_flux(energy, nu_e) / norm_nu_e

            def nubarmuPDF(energy):
                return self.interp_flux(energy, nubar_mu) / norm_nubar_mu

            edges = np.arange(0, 302, 2) # energy bin edges
            ev = (edges[:-1] + edges[1:]) / 2

            return NeutrinoFlux(continuous_fluxes={'ev': ev, 'e': nuePDF(ev), 'mubar': nubarmuPDF(ev), 'mu': numuPDF(ev)},
                                delta_fluxes={'mu': [(29, 1),(236, 0.013)]}, norm=4.9 * (10 ** 7)) ## default unit is /(cm^2*s)
        if flux_name == 'jsns_delayed':
            nu_e = np.genfromtxt(pkg_resources.resource_filename(__name__, "data/jsns2/jsns_nu_e.txt"), delimiter=',')
            nubar_mu = np.genfromtxt(pkg_resources.resource_filename(__name__, "data/jsns2/jsns_nubar_mu.txt"), delimiter=',')

            norm_nu_e = quad(self.interp_flux, 0, 300, args=(nu_e,))[0]
            norm_nubar_mu = quad(self.interp_flux, 0, 300, args=(nubar_mu,))[0]

            def nuePDF(energy):
                return self.interp_flux(energy, nu_e) / norm_nu_e

            def nubarmuPDF(energy):
                return self.interp_flux(energy, nubar_mu) / norm_nubar_mu

            edges = np.arange(0, 302, 2) # energy bin edges
            ev = (edges[:-1] + edges[1:]) / 2
            return NeutrinoFlux(continuous_fluxes={'ev': ev, 'e': nuePDF(ev), 'mubar': nubarmuPDF(ev)}, norm=3 * (10 ** 7))
        if flux_name == 'jsns_prompt':
            return NeutrinoFlux(delta_fluxes={'mu': [(29, 1),(236, 0.013)]}, norm=1.85 * (10 ** 7))
        if flux_name == 'jsns_prompt_continuous':
            nu_mu = np.genfromtxt(pkg_resources.resource_filename(__name__, "data/jsns2/jsns_nu_mu_nodelta.txt"), delimiter=',')
            norm_nu_mu = quad(self.interp_flux, 0, 300, args=(nu_mu,))[0]

            def numuPDF(energy):
                return self.interp_flux(energy, nu_mu) / norm_nu_mu

            edges = np.arange(0, 302, 2) # energy bin edges
            ev = (edges[:-1] + edges[1:]) / 2

            return NeutrinoFlux(continuous_fluxes={'ev': ev, 'mu': numuPDF(ev)},
                                norm=1.85 * (10 ** 4))
        if flux_name == 'far_beam_nu':
            far_beam_txt = 'data/dune_beam_fd_nu_flux_120GeVoptimized.txt'
            f_beam = np.genfromtxt(pkg_resources.resource_filename(__name__, far_beam_txt), delimiter=',')
            nu = {'ev': f_beam[:, 0],
                  'e': f_beam[:, 1],
                  'mu': f_beam[:, 2],
                  'ebar': f_beam[:, 4],
                  'mubar': f_beam[:, 5]}
            return NeutrinoFlux(continuous_fluxes=nu)
        if flux_name == 'far_beam_nubar':
            far_beam_txt = 'data/dune_beam_fd_antinu_flux_120GeVoptimized.txt'
            f_beam = np.genfromtxt(pkg_resources.resource_filename(__name__, far_beam_txt), delimiter=',')
            nu = {'ev': f_beam[:, 0],
                  'e': f_beam[:, 1],
                  'mu': f_beam[:, 2],
                  'ebar': f_beam[:, 4],
                  'mubar': f_beam[:, 5]}
            return NeutrinoFlux(continuous_fluxes=nu)
        if flux_name == 'near_beam_nu':
            far_beam_txt = 'data/dune_beam_nd_nu_flux_120GeVoptimized.txt'
            f_beam = np.genfromtxt(pkg_resources.resource_filename(__name__, far_beam_txt))
            nu = {'ev': f_beam[:, 0],
                  'e': f_beam[:, 1],
                  'mu': f_beam[:, 2],
                  'ebar': f_beam[:, 4],
                  'mubar': f_beam[:, 5]}
            return NeutrinoFlux(continuous_fluxes=nu)
        if flux_name == 'near_beam_nubar':
            far_beam_txt = 'data/dune_beam_nd_antinu_flux_120GeVoptimized.txt'
            f_beam = np.genfromtxt(pkg_resources.resource_filename(__name__, far_beam_txt))
            nu = {'ev': f_beam[:, 0],
                  'e': f_beam[:, 1],
                  'mu': f_beam[:, 2],
                  'ebar': f_beam[:, 4],
                  'mubar': f_beam[:, 5]}
            return NeutrinoFlux(continuous_fluxes=nu)
        if flux_name == 'atmospheric':
            if 'zenith' not in kwargs:
                raise Exception('please specify zenith angle')
            zen = np.round(kwargs['zenith'], decimals=3)
            zen_list = np.round(np.linspace(-0.975, 0.975, 40), decimals=3)
            if zen not in zen_list:
                print('available choice of zenith angle: ', zen_list)
                raise Exception('zenith angle not available')
            idx = (0.975 - zen) / 0.05 * 61
            f_atmos = np.genfromtxt(pkg_resources.resource_filename(__name__, 'data/atmos.txt'), delimiter=',')
            nu = {'ev': f_atmos[int(round(idx)):int(round(idx))+61, 0],
                  'e': f_atmos[int(round(idx)):int(round(idx))+61, 2],
                  'mu': f_atmos[int(round(idx)):int(round(idx))+61, 3],
                  'ebar': f_atmos[int(round(idx)):int(round(idx))+61, 5],
                  'mubar': f_atmos[int(round(idx)):int(round(idx))+61, 6]}
            return NeutrinoFlux(continuous_fluxes=nu)



class NeutrinoFlux:
    def __init__(self, continuous_fluxes=None, delta_fluxes=None, norm=1):
        self.norm = norm * ((100 * meter_by_mev) ** 2)
        self.ev_min = None
        self.ev_max = None
        if continuous_fluxes is None:
            self.nu = None
        elif isinstance(continuous_fluxes, dict):
            self.ev = continuous_fluxes['ev']
            sorted_idx = np.argsort(self.ev)
            self.ev = self.ev[sorted_idx]
            self.ev_min = self.ev[0]
            self.ev_max = self.ev[-1]
            if self.ev_min == 0:
                raise Exception('flux with neutrino energy equal to zeros is not supported. '
                                'please consider using a small value for your lower bound.')
            self.nu = {'e': continuous_fluxes['e'][sorted_idx] if 'e' in continuous_fluxes else None,
                       'mu': continuous_fluxes['mu'][sorted_idx] if 'mu' in continuous_fluxes else None,
                       'tau': continuous_fluxes['tau'][sorted_idx] if 'tau' in continuous_fluxes else None,
                       'ebar': continuous_fluxes['ebar'][sorted_idx] if 'ebar' in continuous_fluxes else None,
                       'mubar': continuous_fluxes['mubar'][sorted_idx] if 'mubar' in continuous_fluxes else None,
                       'taubar': continuous_fluxes['taubar'][sorted_idx] if 'taubar' in continuous_fluxes else None}
            self.binw = self.ev[1:] - self.ev[:-1]
            self.precalc = {None: {flr: self.binw*(flx[1:]+flx[:-1])/2 if flx is not None else None for flr, flx in self.nu.items()}}
        else:
            raise Exception('only support dict as input.')
        if delta_fluxes is None:
            self.delta_nu = None
        elif isinstance(delta_fluxes, dict):
            self.delta_nu = {'e': delta_fluxes['e'] if 'e' in delta_fluxes else None,
                             'mu': delta_fluxes['mu'] if 'mu' in delta_fluxes else None,
                             'tau': delta_fluxes['tau'] if 'tau' in delta_fluxes else None,
                             'ebar': delta_fluxes['ebar'] if 'ebar' in delta_fluxes else None,
                             'mubar': delta_fluxes['mubar'] if 'mubar' in delta_fluxes else None,
                             'taubar': delta_fluxes['taubar'] if 'taubar' in delta_fluxes else None}
            for flavor in self.delta_nu: # grab the maximum energy of the delta fluxes
                if self.delta_nu[flavor] is None:
                    continue
                energies = [self.delta_nu[flavor][i][0] for i in range(len(self.delta_nu[flavor]))]
                if self.ev_max is None or max(energies) > self.ev_max:
                    self.ev_max = max(energies)

        else:
            raise Exception("'delta_fluxes' must be a dictionary of a list of tuples! e.g. {'e': [(12, 4), (14, 15)], ...}")

    def __call__(self, ev, flavor):
        if self.nu is None or self.nu[flavor] is None:
            return 0
        if ev == self.ev_min:
            return self.nu[flavor][0] * self.norm
        if ev == self.ev_max:
            return self.nu[flavor][-1] * self.norm
        if self.ev_min < ev < self.ev_max:
            idx = self.ev.searchsorted(ev)
            l1 = ev - self.ev[idx - 1]
            l2 = self.ev[idx] - ev
            h1 = self.nu[flavor][idx - 1]
            h2 = self.nu[flavor][idx]
            return (l1*h2+l2*h1)/(l1+l2) * self.norm
        return 0

    def integrate(self, ea, eb, flavor, weight_function=None):
        """
        Please avoid using lambda as your weight_function!!!
        :param ea:
        :param eb:
        :param flavor:
        :param weight_function:
        :return:
        """
        if eb <= ea:
            return 0
        res = 0
        if self.delta_nu is not None and self.delta_nu[flavor] is not None:
            for deltas in self.delta_nu[flavor]:
                if ea < deltas[0] <= eb:  # self.ev_max should be included with <=
                    res += deltas[1] if weight_function is None else deltas[1]*weight_function(deltas[0])
        if self.nu is not None and self.nu[flavor] is not None:
            if weight_function not in self.precalc:
                weight = weight_function(self.ev)
                self.precalc[weight_function] = {flr: self.binw*((flx*weight)[1:]+(flx*weight)[:-1])/2
                                                 if flx is not None else None for flr, flx in self.nu.items()}
            eb = min(eb, self.ev_max)
            ea = max(ea, self.ev_min)
            idxmin = self.ev.searchsorted(ea, side='right')
            idxmax = self.ev.searchsorted(eb, side='left')
            if idxmin == idxmax:
                l1 = ea - self.ev[idxmin - 1]
                l2 = self.ev[idxmin] - ea
                h1 = self.nu[flavor][idxmin - 1] * weight_function(self.ev[idxmin - 1]) \
                    if weight_function is not None else self.nu[flavor][idxmin - 1]
                h2 = self.nu[flavor][idxmin] * weight_function(self.ev[idxmin]) \
                    if weight_function is not None else self.nu[flavor][idxmin]
                ha = (l1*h2+l2*h1)/(l1+l2)
                l1 = eb - self.ev[idxmax - 1]
                l2 = self.ev[idxmax] - eb
                hb = (l1*h2+l2*h1)/(l1+l2)
                return (ha + hb) * (eb - ea) / 2 * self.norm
            res += np.sum(self.precalc[weight_function][flavor][idxmin:idxmax-1])
            l1 = ea - self.ev[idxmin-1]
            l2 = self.ev[idxmin] - ea
            h1 = self.nu[flavor][idxmin-1]*weight_function(self.ev[idxmin-1]) \
                if weight_function is not None else self.nu[flavor][idxmin-1]
            h2 = self.nu[flavor][idxmin]*weight_function(self.ev[idxmin]) \
                if weight_function is not None else self.nu[flavor][idxmin]
            res += ((l1*h2+l2*h1)/(l1+l2)+h2)*l2/2
            l1 = eb - self.ev[idxmax - 1]
            l2 = self.ev[idxmax] - eb
            h1 = self.nu[flavor][idxmax - 1] * weight_function(self.ev[idxmax - 1]) \
                if weight_function is not None else self.nu[flavor][idxmax-1]
            h2 = self.nu[flavor][idxmax] * weight_function(self.ev[idxmax]) \
                if weight_function is not None else self.nu[flavor][idxmax]
            res += ((l1 * h2 + l2 * h1) / (l1 + l2) + h1) * l1 / 2
        return res * self.norm

    def change_parameters(self):
        pass



class DMFlux:
    """
    Dark matter flux at COHERENT
    """
    def __init__(self, dark_photon_mass, life_time, coupling_quark, dark_matter_mass,
                 detector_distance=19.3, pot_mu=0.75, pot_sigma=0.25, size=100000, mono_energy=None):
        """
        initialize and generate flux
        :param dark_photon_mass: dark photon mass
        :param life_time: life time of dark photon in rest frame, unit in micro second
        :param coupling_quark: dark photon coupling to quarks
        :param dark_matter_mass: mass of dark matter, unit in MeV
        :param detector_distance: distance from the detector to the Hg target
        :param pot_mu: mean of guassian distribution of proton on target, unit in micro second
        :param pot_sigma: std of guassian distribution of proton on target, unit in micro second
        :param size: size of sampling dark photons
        """
        self.dp_m = dark_photon_mass
        self.dm_m = dark_matter_mass
        self.epsi_quark = coupling_quark
        self.det_dist = detector_distance / meter_by_mev
        self.dp_life = life_time * 1e-6 * c_light / meter_by_mev
        self.pot_mu = pot_mu * 1e-6 * c_light / meter_by_mev
        self.pot_sigma = pot_sigma * 1e-6 * c_light / meter_by_mev
        if mono_energy is None:
            self.timing, self.energy = self._generate(size)
        else:
            self.timing, self.energy = self._mono_flux(mono_energy, pot_mu)

        self.ed_min = self.energy.min()
        self.ed_max = self.energy.max()
        self.dm_norm = self.epsi_quark**2*0.23*1e20 / (4*np.pi*(detector_distance**2)*24*3600) * (meter_by_mev**2) * \
            self.timing.shape[0] * 2 / size

    def _generate(self, size=1000000):
        """
        generate dark matter flux at COHERENT
        :param size: size of sampling dark photons
        :return: time and energy histogram of dark matter
        """
        dp_m = self.dp_m
        dp_e = ((massofpi+massofp)**2 - massofn**2 + dp_m**2)/(2*(massofpi+massofp))
        dp_p = np.sqrt(dp_e ** 2 - dp_m ** 2)
        dp_v = dp_p / dp_e
        gamma = dp_e / dp_m
        tau = self.dp_life * gamma
        tf = np.random.normal(self.pot_mu, self.pot_sigma, size)  # POT
        t = np.random.exponential(tau, size)  # life time of each dark photon
        cs = np.random.uniform(-1, 1, size)  # direction of each dark photon
        # in rest frame
        estar = dp_m / 2
        pstar = np.sqrt(estar ** 2 - self.dm_m ** 2)
        pstarx = pstar * cs
        pstary = pstar * np.sqrt(1 - cs ** 2)
        # boost to lab frame
        elab = gamma * (estar + dp_v * pstarx)
        plabx = gamma * (pstarx + dp_v * estar)
        plaby = pstary
        vx = plabx / elab
        vy = plaby / elab
        timing = []
        energy = []
        for i in range(size):
            a = vx[i] ** 2 + vy[i] ** 2
            b = 2 * vx[i] * t[i] * dp_v
            cc = dp_v ** 2 * t[i] ** 2 - self.det_dist ** 2
            if b ** 2 - 4 * a * cc >= 0:
                if (-b - np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a) > 0:
                    timing.append((-b - np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a) + t[i] + tf[i])
                    energy.append(elab[i])
                if (-b + np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a) > 0:
                    timing.append((-b + np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a) + t[i] + tf[i])
                    energy.append(elab[i])
        return np.array(timing) / c_light * meter_by_mev * 1e6, np.array(energy)

    def _mono_flux(self, e_chi, t_trig, size=1000):
        return np.random.normal(loc=t_trig, scale=0.01*t_trig, size=size), np.random.normal(loc=e_chi, scale=0.005*e_chi, size=size)

    def flux(self, ev):
        """
        dark matter flux
        :param ev: dark matter energy
        :return: dark matter flux
        """
        return 1/(self.ed_max-self.ed_min)*self.dm_norm if self.ed_min <= ev <= self.ed_max else 0

    def fint(self, er, m, **kwargs):
        """
        flux/(ex^2-mx^2) integration
        :param er: recoil energy in MeV
        :param m: target nucleus mass in MeV
        :param kwargs: other argument
        :return: flux/(ex^2-mx^2) integration
        """
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_m**2+4*m*self.dm_m**2)/m) + er)
        emin = 0.0 * emin

        def integrand(ex):
            return self.flux(ex)/(ex**2 - self.dm_m**2)

        if not isinstance(emin, np.ndarray):
            res = quad(integrand, emin, self.ed_max)[0]
        else:
            res = np.zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(integrand, emin[i], self.ed_max)[0]
        return res

    def fint1(self, er, m, **kwargs):
        """
        flux*ex/(ex^2-mx^2) integration
        :param er: recoil energy in MeV
        :param m: target nucleus mass in MeV
        :param kwargs: other argument
        :return: flux*ex/(ex^2-mx^2) integration
        """
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_m**2+4*m*self.dm_m**2)/m) + er)
        emin = 0.0 * emin

        def integrand(ex):
            return self.flux(ex) * ex / (ex ** 2 - self.dm_m ** 2)

        if not isinstance(emin, np.ndarray):
            res = quad(integrand, emin, self.ed_max)[0]
        else:
            res = np.zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(integrand, emin[i], self.ed_max)[0]
        return res

    def fint2(self, er, m, **kwargs):
        """
        flux*ex^2/(ex^2-mx^2) integration
        :param er: recoil energy in MeV
        :param m: target nucleus mass in MeV
        :param kwargs: other argument
        :return: flux*ex^2/(ex^2-mx^2) integration
        """
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_m**2+4*m*self.dm_m**2)/m) + er)
        emin = 0.0 * emin

        def integrand(ex):
            return self.flux(ex) * ex**2 / (ex ** 2 - self.dm_m ** 2)

        if not isinstance(emin, np.ndarray):
            res = quad(integrand, emin, self.ed_max)[0]
        else:
            res = np.zeros_like(emin)
            for i in range(emin.shape[0]):
                res[i] = quad(integrand, emin[i], self.ed_max)[0]
        return res



class DMFluxIsoPhoton(FluxBaseContinuous):
    def __init__(self, photon_distribution, dark_photon_mass, coupling, dark_matter_mass, life_time=0.001,
                 detector_distance=19.3, pot_rate=5e20, pot_sample=100000, brem_suppress=True,
                 pot_mu=0.7, pot_sigma=0.15, sampling_size=100, nbins=20, verbose=False):
        self.nbins = nbins
        self.photon_flux = photon_distribution
        self.dp_m = dark_photon_mass
        self.dm_m = dark_matter_mass
        self.epsilon = coupling
        self.life_time = life_time  # input in mus, internal in s
        self.det_dist = detector_distance  # meters
        self.pot_rate = pot_rate  # the number of POT/day in the experiment
        self.pot_mu = pot_mu
        self.pot_sigma = pot_sigma
        self.pot_sample = pot_sample  # the number of POT in photon_distribution
        self.time = []
        self.energy = []
        self.weight = []
        self.norm = 1
        self.sampling_size = sampling_size
        self.supp = brem_suppress  # add phase space suppression
        self.verbose = verbose
        for photon_events in photon_distribution:
            if self.verbose:
                print("getting photons from E =", photon_events[0], "Size =", photon_events[1])
            self._generate_single(photon_events, self.sampling_size)

        normalization = self.epsilon ** 2 * (self.pot_rate / self.pot_sample) \
                        / (4 * np.pi * (self.det_dist ** 2) * 24 * 3600) * (meter_by_mev**2)
        self.norm = normalization
        self.weight = [x * self.norm for x in self.weight]
        self.timing = np.array(self.time) * 1e6
        hist, bin_edges = np.histogram(self.energy, bins=nbins, weights=self.weight, density=True)
        super().__init__((bin_edges[:-1] + bin_edges[1:]) / 2, hist, norm=np.sum(self.weight))

    def getScaledWeights(self):
        wgt = self.weight
        wgt = [x * self.norm * 24 * 3600 / (meter_by_mev**2) for x in wgt]
        return wgt

    def simulate(self):
        self.time = []
        self.energy = []
        self.weight = []
        normalization = self.epsilon ** 2 * (self.pot_rate / self.pot_sample) \
                        / (4 * np.pi * (self.det_dist ** 2) * 24 * 3600) * (meter_by_mev**2)
        self.norm = normalization

        for photon_events in self.photon_flux:
            if self.verbose:
                print("getting photons from E =", photon_events[0], "Size =", photon_events[1])
            self._generate_single(photon_events, self.sampling_size)
        self.weight = [x * self.norm for x in self.weight]
        self.timing = np.array(self.time) * 1e6
        hist, bin_edges = np.histogram(self.energy, bins=self.nbins, weights=self.weight, density=True)
        super().__init__((bin_edges[:-1] + bin_edges[1:]) / 2, hist, norm=np.sum(self.weight))

    def _generate_single(self, photon_events, nsamples):
        # Initiate photon position, energy and momentum.
        if photon_events[0]**2 < self.dp_m**2:
            return
        dp_m = self.dp_m
        dp_e = photon_events[0]
        dp_p = np.sqrt(dp_e ** 2 - self.dp_m ** 2)
        dp_momentum = np.array([dp_e, 0, 0, dp_p])


        # dark photon to dark matter
        dm_m = self.dm_m
        dm_e = self.dp_m / 2
        dm_p = np.sqrt(dm_e ** 2 - dm_m ** 2)

        # Directional sampling.
        dp_wgt = photon_events[1] / nsamples  # Event weight
        # Brem suppression
        if self.supp == True:
            el_e = 1.0773*dp_e + 13.716  # most likely electron energy that produced this dark photon
            supp_fact = min(1, 1154 * np.exp(-24.42 * np.power(dp_m/el_e, 0.3174)))
            dp_wgt *= supp_fact

        ## optimize
        #pos = np.zeros(3) ## optimize
        t = np.random.normal(self.pot_mu * 1e-6, self.pot_sigma * 1e-6, nsamples)
        t_dp = np.random.exponential(1e-6 * self.life_time * dp_momentum[0] / dp_m, nsamples)
        t += t_dp
        csd = np.random.uniform(-1, 1, nsamples)
        phid = np.random.uniform(0, 2 * np.pi, nsamples)
        boost_matr = lorentz_matrix(np.array([-dp_momentum[1] / dp_momentum[0],
                                              -dp_momentum[2] / dp_momentum[0],
                                              -dp_momentum[3] / dp_momentum[0]]))
        pos_z = c_light * t_dp * dp_momentum[3] / dp_momentum[0]  # position is along z by construction
        for i in range(nsamples):
            dm_momentum = np.array([dm_e, dm_p * np.sqrt(1 - csd[i] ** 2) * np.cos(phid[i]),
                                    dm_p * np.sqrt(1 - csd[i] ** 2) * np.sin(phid[i]), dm_p * csd[i]])
            dm_momentum = boost_matr @ dm_momentum

            # dark matter arrives at detector, assuming azimuthal symmetric
            # append the time and energy spectrum of the DM.
            # DM particle 1
            v = dm_momentum[1:] / dm_momentum[0] * c_light
            a = v[0]*v[0] + v[1]*v[1] + v[2]*v[2] #np.sum(v ** 2)
            b = 2*v[2]*pos_z[i]  # dot product is along z by construction
            c = pos_z[i]**2 - self.det_dist ** 2
            if b ** 2 - 4 * a * c >= 0:
                t_dm = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                if t_dm >= 0:
                    if self.verbose:
                        print("adding weight", dp_wgt)
                    self.time.append(t[i]+t_dm)
                    self.energy.append(dm_momentum[0])
                    self.weight.append(dp_wgt)
                t_dm = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                if t_dm >= 0:
                    if self.verbose:
                        print("adding weight", dp_wgt)
                    self.time.append(t[i]+t_dm)
                    self.energy.append(dm_momentum[0])
                    self.weight.append(dp_wgt)
            # DM particle 2
            v = (dp_momentum - dm_momentum)[1:] / (dp_momentum - dm_momentum)[0] * c_light
            a = v[0]*v[0] + v[1]*v[1] + v[2]*v[2] #np.sum(v ** 2)
            b = b = 2*v[2]*pos_z[i]
            c = pos_z[i]**2 - self.det_dist ** 2
            if b ** 2 - 4 * a * c >= 0:
                t_dm = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                if t_dm >= 0:
                    if self.verbose:
                        print("adding weight", dp_wgt)
                    self.time.append(t[i]+t_dm)
                    self.energy.append((dp_momentum - dm_momentum)[0])
                    self.weight.append(dp_wgt)
                t_dm = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                if t_dm >= 0:
                    if self.verbose:
                        print("adding weight", dp_wgt)
                    self.time.append(t[i]+t_dm)
                    self.energy.append((dp_momentum - dm_momentum)[0])
                    self.weight.append(dp_wgt)


    def fint(self, er, m):
        if np.isscalar(m):
            m = np.array([m])
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_m**2+4*m*self.dm_m**2)/m) + er)
        res = np.zeros_like(emin)
        for i in range(emin.shape[0]):
            res[i] = self.integrate(emin[i], self.ev_max, weight_function=self.f0)
        return res

    def fint1(self, er, m):
        if np.isscalar(m):
            m = np.array([m])
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_m**2+4*m*self.dm_m**2)/m) + er)
        res = np.zeros_like(emin)
        for i in range(emin.shape[0]):
            res[i] = self.integrate(emin[i], self.ev_max, weight_function=self.f1)
        return res

    def fint2(self, er, m):
        if np.isscalar(m):
            m = np.array([m])
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_m**2+4*m*self.dm_m**2)/m) + er)
        res = np.zeros_like(emin)
        for i in range(emin.shape[0]):
            res[i] = self.integrate(emin[i], self.ev_max, weight_function=self.f2)
        return res

    def f0(self, ev):
        return 1/(ev**2 - self.dm_m**2)

    def f1(self, ev):
        return ev/(ev**2 - self.dm_m**2)

    def f2(self, ev):
        return ev**2 / (ev**2 - self.dm_m**2)



class DMFluxFromPiMinusAbsorption:
    r"""
    Dark matter flux from pi^- + p -> A^\prime + n -> \chi + \chi + n
    """
    def __init__(self, dark_photon_mass, coupling_quark, dark_matter_mass, life_time=0.001,
                 detector_distance=19.3, pot_rate=5e20, pot_mu=0.7, pot_sigma=0.15, pion_rate=18324/500000,
                 sampling_size=100000):
        """
        initialize and generate flux
        default values are COHERENT experiment values
        :param dark_photon_mass: dark photon mass
        :param life_time: life time of dark photon in rest frame, unit in micro second
        :param coupling_quark: dark photon coupling to quarks divided by electron charge
        :param dark_matter_mass: mass of dark matter, unit in MeV
        :param detector_distance: distance from the detector to the target
        :param pot_rate: proton on target rate, unit POT/day
        :param pot_mu: mean of guassian distribution of proton on target, unit in micro second
        :param pot_sigma: std of guassian distribution of proton on target, unit in micro second
        :param pion_rate: pi^- production rate
        :param sampling_size: size of sampling dark photons
        """
        self.dp_m = dark_photon_mass
        self.dm_m = dark_matter_mass
        self.epsi_quark = coupling_quark
        self.det_dist = detector_distance / meter_by_mev
        self.life_time = life_time  # input in mus, internal in s
        self.pot_mu = pot_mu
        self.pot_sigma = pot_sigma
        self.pot_rate = pot_rate
        self.pion_rate = pion_rate
        self.sampling_size = sampling_size
        self.timing = []
        self.energy = []
        self.ed_min = None
        self.ed_max = None
        self.norm = None
        self.simulate()
        self.ev_min = self.ed_min
        self.ev_max = self.ed_max

    def get_lifetime(self, g, m):
        return ((16 * np.pi ** 2) / ((g ** 2) * m)) * mev_per_hz

    def simulate(self):
        """
        generate dark matter flux
        """
        # First check that the dp mass is less than the pi- mass.
        if self.dp_m > massofpi:
            self.norm = 0.0
            return
        dp_m = self.dp_m
        dp_e = ((massofpi + massofp) ** 2 - massofn ** 2 + dp_m ** 2) / (2 * (massofpi + massofp))
        dp_p = np.sqrt(dp_e ** 2 - dp_m ** 2)
        dp_v = dp_p / dp_e
        gamma = dp_e / dp_m
        tau = (self.life_time * 1e-6 * c_light / meter_by_mev) * gamma
        tf = np.random.normal(self.pot_mu * 1e-6 * c_light / meter_by_mev,
                              self.pot_sigma * 1e-6 * c_light / meter_by_mev,
                              self.sampling_size)  # POT
        t = np.random.exponential(tau, self.sampling_size)  # life time of each dark photon
        cs = np.random.uniform(-1, 1, self.sampling_size)  # direction of each dark photon
        # in rest frame
        estar = dp_m / 2
        pstar = np.sqrt(estar ** 2 - self.dm_m ** 2)
        pstarx = pstar * cs
        pstary = pstar * np.sqrt(1 - cs ** 2)
        # boost to lab frame
        elab = gamma * (estar + dp_v * pstarx)
        plabx = gamma * (pstarx + dp_v * estar)
        plaby = pstary
        vx = plabx / elab
        vy = plaby / elab
        timing = []
        energy = []
        for i in range(self.sampling_size):
            a = vx[i] ** 2 + vy[i] ** 2
            b = 2 * vx[i] * t[i] * dp_v
            cc = dp_v ** 2 * t[i] ** 2 - self.det_dist ** 2
            if b ** 2 - 4 * a * cc >= 0:
                if (-b - np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a) > 0:
                    timing.append((-b - np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a) + t[i] + tf[i])
                    energy.append(elab[i])
                if (-b + np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a) > 0:
                    timing.append((-b + np.sqrt(b ** 2 - 4 * a * cc)) / (2 * a) + t[i] + tf[i])
                    energy.append(elab[i])
        self.timing = np.array(timing) / c_light * meter_by_mev * 1e6
        self.energy = np.array(energy)
        self.ed_min = min(energy)
        self.ed_max = max(energy)
        self.ev_min = self.ed_min
        self.ev_max = self.ed_max
        self.norm = self.epsi_quark ** 2 * self.pot_rate * self.pion_rate / (4 * np.pi * (self.det_dist ** 2) * 24 * 3600) * \
                    self.timing.shape[0] * 2 / self.sampling_size

    def __call__(self, ev):
        """
        dark matter flux, the spectrum is flat because of isotropic
        :param ev: dark matter energy
        :return: dark matter flux
        """
        return 1 / (self.ed_max - self.ed_min) * self.norm if self.ed_min <= ev <= self.ed_max else 0

    def integrate(self, ea, eb, weight_function=None):
        """
        adaptive quadrature can achieve almost linear time on simple weight function, no need to do precalculation
        :param ea: lowerbound
        :param eb: upperbound
        :param weight_function: weight function
        :return: integration of the flux, weighted by the weight function
        """
        if eb <= ea:
            return 0
        eb = min(eb, self.ed_max)
        ea = max(ea, self.ed_min)
        if weight_function is None:
            return (eb - ea) / (self.ed_max - self.ed_min) * self.norm
        return quad(weight_function, ea, eb, epsrel=1e-3)[0] / (self.ed_max - self.ed_min) * self.norm

    def change_parameters(self, dark_photon_mass=None, life_time=None, coupling_quark=None, dark_matter_mass=None,
                          detector_distance=None, pot_rate=None, pot_mu=None, pot_sigma=None, pion_rate=None, sampling_size=None):
        self.dp_m = dark_photon_mass if dark_photon_mass is not None else self.dp_m
        self.dp_life = life_time * 1e-6 * c_light / meter_by_mev if life_time is not None else self.dp_life
        self.epsi_quark = coupling_quark if coupling_quark is not None else self.epsi_quark
        self.dm_m = dark_matter_mass if dark_matter_mass is not None else self.dm_m
        self.det_dist = detector_distance / meter_by_mev if detector_distance is not None else self.det_dist
        self.pot_rate = pot_rate if pot_rate is not None else self.pot_rate
        self.pot_mu = pot_mu * 1e-6 * c_light / meter_by_mev if pot_mu is not None else self.pot_mu
        self.pot_sigma = pot_sigma * 1e-6 * c_light / meter_by_mev if pot_sigma is not None else self.pot_sigma
        self.pion_rate = self.pion_rate if pion_rate is not None else self.pion_rate
        self.sampling_size = sampling_size if sampling_size is not None else self.sampling_size
        self.simulate()

    def fint(self, er, m):
        if np.isscalar(m):
            m = np.array([m])
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_m**2+4*m*self.dm_m**2)/m) + er)
        res = np.zeros_like(emin)
        for i in range(emin.shape[0]):
            res[i] = self.integrate(emin[i], self.ev_max, weight_function=self.f0)
        return res

    def fint1(self, er, m):
        if np.isscalar(m):
            m = np.array([m])
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_m**2+4*m*self.dm_m**2)/m) + er)
        res = np.zeros_like(emin)
        for i in range(emin.shape[0]):
            res[i] = self.integrate(emin[i], self.ev_max, weight_function=self.f1)
        return res

    def fint2(self, er, m):
        if np.isscalar(m):
            m = np.array([m])
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_m**2+4*m*self.dm_m**2)/m) + er)
        res = np.zeros_like(emin)
        for i in range(emin.shape[0]):
            res[i] = self.integrate(emin[i], self.ev_max, weight_function=self.f2)
        return res

    def f0(self, ev):
        return 1/(ev**2 - self.dm_m**2)

    def f1(self, ev):
        return ev/(ev**2 - self.dm_m**2)

    def f2(self, ev):
        return ev**2 / (ev**2 - self.dm_m**2)



class DMFluxFromPi0Decay(FluxBaseContinuous):
    """
    z direction is the direction of the beam
    """
    def __init__(self, pi0_distribution, dark_photon_mass, coupling_quark, dark_matter_mass, meson_mass=massofpi0, life_time=0.001,
                 detector_distance=19.3, detector_direction=0, detector_width=0.1, pot_rate=5e20, pot_mu=0.7,
                 pot_sigma=0.15, pion_rate=52935/500000, nbins=20):
        self.pi0_distribution = pi0_distribution
        self.dp_m = dark_photon_mass
        self.life_time = life_time      # input in mus, internal in s
        self.epsilon = coupling_quark
        self.dm_m = dark_matter_mass
        self.meson_mass = meson_mass
        self.det_dist = detector_distance
        self.det_direc = detector_direction
        self.det_width = detector_width
        self.pot_rate = pot_rate
        self.pot_mu = pot_mu
        self.pot_sigma = pot_sigma
        self.pion_rate = pion_rate
        self.time = []
        self.energy = []
        self.nbins = nbins
        self.dm_m = dark_matter_mass
        for pi0_events in pi0_distribution:  # must be in the form [azimuth, cos(zenith), kinetic energy]
            self._generate_single(pi0_events)
        self.timing = np.array(self.time)*1e6
        hist, bin_edges = np.histogram(self.energy, bins=nbins, density=True)
        ps_factor = np.heaviside(self.meson_mass - self.dp_m, 0.0) * 2 * self.epsilon**2 * (1 - (self.dp_m / self.meson_mass)**2)**3
        super().__init__((bin_edges[:-1]+bin_edges[1:])/2, hist,
                         norm=ps_factor*pot_rate*pion_rate*len(self.time)/len(pi0_distribution)/
                              (2*np.pi*(min(1.0, detector_direction+detector_width/2)-max(-1.0, detector_direction-detector_width/2))*detector_distance**2*24*3600)
                              *(meter_by_mev**2))
    def get_lifetime(self, g, m):
        return ((16 * np.pi ** 2) / ((g ** 2) * m)) * mev_per_hz

    def simulate(self):
        self.time = []
        self.energy = []
        for pi0_events in self.pi0_distribution:  # must be in the form [azimuth, cos(zenith), kinetic energy]
            self._generate_single(pi0_events)
        self.timing = np.array(self.time)*1e6
        hist, bin_edges = np.histogram(self.energy, bins=self.nbins, density=True)
        ps_factor = np.heaviside(self.meson_mass - self.dp_m, 0.0) * 2 * self.epsilon**2 * (1 - (self.dp_m / self.meson_mass)**2)**3
        norm = ps_factor * self.pot_rate * self.pion_rate * \
            len(self.time)/len(self.pi0_distribution)/ \
                (2*np.pi*(min(1.0, self.det_direc+self.det_width/2)-max(-1.0, self.det_direc-self.det_width/2))*self.det_dist**2*24*3600)*(meter_by_mev**2)
        super().__init__((bin_edges[:-1]+bin_edges[1:])/2, hist, norm=norm)

    def _generate_single(self, pi0_events):
        if self.dp_m > self.meson_mass:
            return
        pos = np.zeros(3)
        t = 0
        t += np.random.normal(self.pot_mu * 1e-6, self.pot_sigma * 1e-6)
        pi_e = self.meson_mass + pi0_events[2]
        pi_p = np.sqrt(pi_e**2 - self.meson_mass**2)
        pi_v = pi_p / pi_e
        t_pi = np.random.exponential(8.4e-17*pi_e/self.meson_mass)
        pos += pi_v * polar_to_cartesian(pi0_events[:2]) * t_pi * c_light
        t += t_pi
        # pi0 to dark photon
        dp_m = self.dp_m
        dp_e = (self.meson_mass**2 + dp_m**2)/(2*self.meson_mass)
        dp_p = (self.meson_mass**2 - dp_m**2)/(2*self.meson_mass)
        cs = np.random.uniform(-1, 1)
        phi = np.random.uniform(0, 2*np.pi)
        dp_momentum = np.array([dp_e, dp_p*np.sqrt(1-cs**2)*np.cos(phi), dp_p*np.sqrt(1-cs**2)*np.sin(phi), dp_p*cs])
        dp_momentum = lorentz_boost(dp_momentum, -pi_v*polar_to_cartesian(pi0_events[:2]))
        t_dp = np.random.exponential((self.life_time*1e-6)*dp_momentum[0]/dp_m)
        pos += c_light*t_dp*np.array([dp_momentum[1]/dp_momentum[0], dp_momentum[2]/dp_momentum[0], dp_momentum[3]/dp_momentum[0]])
        t += t_dp
        # dark photon to dark matter
        dm_m = self.dm_m
        dm_e = dp_m / 2
        dm_p = np.sqrt(dm_e**2 - dm_m**2)
        csd = np.random.uniform(-1, 1)
        phid = np.random.uniform(0, 2*np.pi)
        dm_momentum = np.array([dm_e, dm_p*np.sqrt(1-csd**2)*np.cos(phid), dm_p*np.sqrt(1-csd**2)*np.sin(phid), dm_p*csd])
        dm_momentum = lorentz_boost(dm_momentum, np.array([-dp_momentum[1]/dp_momentum[0],
                                                           -dp_momentum[2]/dp_momentum[0],
                                                           -dp_momentum[3]/dp_momentum[0]]))
        # dark matter arrives at detector, assuming azimuthal symmetric
        v = dm_momentum[1:]/dm_momentum[0]*c_light
        a = np.sum(v**2)
        b = 2*np.sum(v*pos) #2 * v[2] * (c_light * dp_p / dp_e) * t_dp
        c = np.sum(pos**2) - self.det_dist**2
        if b**2 - 4*a*c >= 0:
            t_dm = (-b+np.sqrt(b**2-4*a*c))/(2*a)
            if t_dm >= 0: #and self.det_direc-self.det_width/2 <= (pos[2]+v[2]*t_dm)/np.sqrt(np.sum((v*t_dm + pos)**2)) <= self.det_direc+self.det_width/2:
                self.time.append(t+t_dm)
                self.energy.append(dm_momentum[0])
            t_dm = (-b-np.sqrt(b**2-4*a*c))/(2*a)
            if t_dm >= 0: #and self.det_direc-self.det_width/2 <= (pos[2]+v[2]*t_dm)/np.sqrt(np.sum((v*t_dm + pos)**2)) <= self.det_direc+self.det_width/2:
                self.time.append(t+t_dm)
                self.energy.append(dm_momentum[0])
        v = (dp_momentum-dm_momentum)[1:]/(dp_momentum-dm_momentum)[0]*c_light
        a = np.sum(v**2)
        b = 2*np.sum(v*pos)
        c = np.sum(pos**2) - self.det_dist**2
        if b**2 - 4*a*c >= 0:
            t_dm = (-b+np.sqrt(b**2-4*a*c))/(2*a)
            if t_dm >= 0: #and self.det_direc-self.det_width/2 <= (pos[2]+v[2]*t_dm)/np.sqrt(np.sum((v*t_dm + pos)**2)) <= self.det_direc+self.det_width/2:
                self.time.append(t+t_dm)
                self.energy.append((dp_momentum-dm_momentum)[0])
            t_dm = (-b-np.sqrt(b**2-4*a*c))/(2*a)
            if t_dm >= 0: #and self.det_direc-self.det_width/2 <= (pos[2]+v[2]*t_dm)/np.sqrt(np.sum((v*t_dm + pos)**2)) <= self.det_direc+self.det_width/2:
                self.time.append(t+t_dm)
                self.energy.append((dp_momentum-dm_momentum)[0])

    def to_pandas(self):
        return pd.DataFrame({'time': self.time, 'energy': self.energy})

    def fint(self, er, m):
        if np.isscalar(m):
            m = np.array([m])
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_m**2+4*m*self.dm_m**2)/m) + er)
        res = np.zeros_like(emin)
        for i in range(emin.shape[0]):
            res[i] = self.integrate(emin[i], self.ev_max, weight_function=self.f0)
        return res

    def fint1(self, er, m):
        if np.isscalar(m):
            m = np.array([m])
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_m**2+4*m*self.dm_m**2)/m) + er)
        res = np.zeros_like(emin)
        for i in range(emin.shape[0]):
            res[i] = self.integrate(emin[i], self.ev_max, weight_function=self.f1)
        return res

    def fint2(self, er, m):
        if np.isscalar(m):
            m = np.array([m])
        emin = 0.5 * (np.sqrt((er**2*m+2*er*m**2+2*er*self.dm_m**2+4*m*self.dm_m**2)/m) + er)
        res = np.zeros_like(emin)
        for i in range(emin.shape[0]):
            res[i] = self.integrate(emin[i], self.ev_max, weight_function=self.f2)
        return res

    def f0(self, ev):
        return 1/(ev**2 - self.dm_m**2)

    def f1(self, ev):
        return ev/(ev**2 - self.dm_m**2)

    def f2(self, ev):
        return ev**2 / (ev**2 - self.dm_m**2)

