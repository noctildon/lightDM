import sys
sys.path.append("../")

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import pyCEvNS.axion as alp
from pyCEvNS.axion import primakoff_scattering_xs, IsotropicAxionFromPrimakoff, primakoff_prod_quant, primakoff_production_cdf


isoprim = IsotropicAxionFromPrimakoff(photon_rates=[[1,1]], axion_mass=0.1, axion_coupling=2.7e-8)

print(r"testing a + Z $\to \gamma$ + Z")
mass_array = np.linspace(0, 1.2, 100)
xs_array = np.zeros_like(mass_array)
for i, m in enumerate(mass_array):
    xs_array[i] = primakoff_scattering_xs(1.115, 32, ma=m, g=2.7e-8)


mev2_to_barn = 0.00257
plt.plot(mass_array, 1e-24 * xs_array/mev2_to_barn)
plt.show()


print("testing primakoff production CDF")
theta_list = np.linspace(0, pi, 1000)
cdf_list = np.empty_like(theta_list)
for i, th in enumerate(theta_list):
    cdf_list[i] = primakoff_production_cdf(th, 20, 23, 0.1)

plt.plot(theta_list, cdf_list)
plt.show()


print(r"testing primakoff ALP production angular distribution")
u = np.random.random(10000)
thetas = primakoff_prod_quant(u, 20, 32, 0.1)

plt.hist(thetas, bins=20)
plt.show()


print("plotting primakoff production PDF")
pdf_list_1 = np.empty_like(theta_list)
pdf_list_2 = np.empty_like(theta_list)
pdf_list_3 = np.empty_like(theta_list)
pdf_list_4 = np.empty_like(theta_list)
pdf_list_5 = np.empty_like(theta_list)
for i, th in enumerate(theta_list):
    pdf_list_1[i] = alp.primakoff_production_diffxs(th, 200, 32, 100)
    pdf_list_2[i] = alp.primakoff_production_diffxs(th, 1000, 32, 100)
    pdf_list_3[i] = alp.primakoff_production_diffxs(th, 5000, 32, 100)
    pdf_list_4[i] = alp.primakoff_production_diffxs(th, 10000, 32, 100)
    pdf_list_5[i] = alp.primakoff_production_diffxs(th, 50000, 32, 100)

plt.hist(theta_list, weights=pdf_list_1, bins=500, density=True, histtype='step', label=r"$E_\gamma = 200$ MeV")
plt.hist(theta_list, weights=pdf_list_2, bins=500, density=True, histtype='step', label=r"$E_\gamma = 1$ GeV")
plt.hist(theta_list, weights=pdf_list_3, bins=500, density=True, histtype='step', label=r"$E_\gamma = 5$ GeV")
plt.hist(theta_list, weights=pdf_list_4, bins=500, density=True, histtype='step', label=r"$E_\gamma = 10$ GeV")
plt.hist(theta_list, weights=pdf_list_5, bins=500, density=True, histtype='step', label=r"$E_\gamma = 50$ GeV")
plt.xlabel(r"$\theta_a$ [rad]", fontsize=15)
plt.ylabel(r"$\dfrac{1}{\sigma}\dfrac{d\sigma}{d\theta}$", fontsize=15)
plt.title(r"$m_a = 0.1$ GeV")
plt.yscale('log')
plt.ylim(bottom=1e-3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
plt.clf()

print("plotting primakoff production PDF")
pdf_list_1 = np.empty_like(theta_list)
pdf_list_2 = np.empty_like(theta_list)
pdf_list_3 = np.empty_like(theta_list)
pdf_list_4 = np.empty_like(theta_list)
pdf_list_5 = np.empty_like(theta_list)
for i, th in enumerate(theta_list):
    pdf_list_1[i] = alp.primakoff_production_diffxs(th, 1000, 32, 0.000001)
    pdf_list_2[i] = alp.primakoff_production_diffxs(th, 1000, 32, 0.001)
    pdf_list_3[i] = alp.primakoff_production_diffxs(th, 1000, 32, 10)
    pdf_list_4[i] = alp.primakoff_production_diffxs(th, 1000, 32, 100)
    pdf_list_5[i] = alp.primakoff_production_diffxs(th, 1000, 32, 500)

plt.hist(theta_list, weights=pdf_list_1, bins=500, density=True, label=r"$m_a = 0.00000$ MeV")
plt.hist(theta_list, weights=pdf_list_2, bins=500, density=True, label=r"$m_a = 0.001$ MeV")
plt.hist(theta_list, weights=pdf_list_3, bins=500, density=True, label=r"$m_a = 10$ MeV")
plt.hist(theta_list, weights=pdf_list_4, bins=500, density=True, label=r"$m_a = 100$ MeV")
plt.hist(theta_list, weights=pdf_list_5, bins=500, density=True, label=r"$m_a = 500$ MeV")
plt.xlabel(r"$\theta_a$ [rad]")
plt.ylabel(r"$\dfrac{1}{\sigma}\dfrac{d\sigma}{d\theta}$")
plt.legend()
plt.show()