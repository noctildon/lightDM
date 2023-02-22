from pyCEvNS.flux import *

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.pylab import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# todo: write real unit test for this
# brem suppression
def brem_supp(r):
    supp = 1154 * np.exp(-24.42 * np.power(r, 0.3174))
    is_valid = (1 > supp)
    return is_valid * supp + 1 * np.invert(is_valid)

r_vals = np.linspace(0, 1, 1000)
supp_vals = brem_supp(r_vals)
plt.plot(r_vals, supp_vals, label=r"$f = 1154 \exp(-24.42 x^{0.3174})$")
plt.yscale('log')
plt.xscale('log')
plt.xlim((0.019, 1))
plt.title(r"$e^\pm \to e^\pm X$", loc="right", fontsize=15)
plt.xlabel(r"$x = m_X / E_e$", fontsize=15)
plt.ylabel(r"$f$", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()
plt.clf()


from pyCEvNS.events import rates_dm_electron
from pyCEvNS.detectors import *
lar = Detector('jsns_scintillator')
brem_photons = np.genfromtxt("data/jsns/brem.txt")

dmflux_25 = DMFlux(dark_photon_mass=75, life_time=0.001, coupling_quark=1, dark_matter_mass=25)
dmflux_2 = DMFlux(dark_photon_mass=75, life_time=0.001, coupling_quark=1, dark_matter_mass=2)
def diffxs(er, fx):
    return rates_dm_electron(er, lar, fx, mediator_mass=75,
                             epsilon=1, efficiency=None, smear=False)


ers = np.logspace(-3,3, 1000)
rates_25 = [diffxs(er, dmflux_25) for er in ers]
rates_2 = [diffxs(er, dmflux_2) for er in ers]
plt.plot(ers, rates_25, label=r"$m_\chi = 25$ MeV")
plt.plot(ers, rates_2, label=r"$m_\chi = 2$ MeV")
plt.title(r"$m_V = 25$ MeV")
plt.ylabel(r"Rate $\propto \frac{d\sigma}{dE_{r,e}}$ [a.u.]")
plt.xlabel(r"$E_r$ [MeVee]")
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.show()
plt.clf()


brem_photons = np.genfromtxt("data/coherent/brem.txt")
isoflux = DMFluxIsoPhoton(brem_photons, 75, 1, 25)

isoflux.simulate()
print(isoflux.fx)
plt.hist2d(isoflux.energy,isoflux.time)
plt.xlabel(r"$E_\chi$ [MeV]")
plt.ylabel(r"$t$ [s]")
plt.show()
plt.clf()

isoflux.life_time = 1
isoflux.coupling = 0.5
isoflux.simulate()
print(isoflux.fx)
plt.hist2d(isoflux.energy,isoflux.time)
plt.xlabel(r"$E_\chi$ [MeV]")
plt.ylabel(r"$t$ [s]")
plt.show()
plt.clf()


