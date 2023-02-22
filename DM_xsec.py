from util import *
from exp_config import *


def ldotl(echi, mchi, q, dE, particle):
    """
    echi: DM energy
    mchi: DM mass
    q: momentum transfer (usually zero)
    dE: excitation energy
    particle: fermion or scalar for DM
    return: the curren term l dot l
    """
    if echi <= mchi + dE:
        return 0
    pchi = momentum(echi, mchi)   # incoming dark matter momentum
    pchip = momentum(echi - dE, mchi) # outgoing dark matter momentum

    if particle == 'fermion':
        return 3- ( .5*(pchi**2 + pchip**2 - q**2) +3/4* mchi**2 ) / (echi * (echi - dE))
    elif particle == 'scalar':
        return (pchi**2 + pchip**2 - q**2) / (2*echi * (echi - dE))
    else:
        raise Exception('DM particle must be fermion or scalar')


def dsigmadcos(echi, mchi, bgt, dE, eps, mA, Ji, particle):
    e = 0.30282212
    gD = np.sqrt(4*np.pi*alpha_D)
    mmed = 3*mchi

    res = 2 * e**2 * eps**2 * gD**2 * (echi - dE)**2 / ((mmed**2)**2)
    res *= 1 / (2*np.pi) * (4*np.pi) / (2*Ji+1)
    res *= gA**2/(12*np.pi) * ldotl(echi, mchi, 0.0, dE, particle) * 2.0*bgt # 2 is convention for bgt

    return res


# GT total cross section with cos integration
def sigmaGT0(echi, mchi, bgt, dE, eps, mA, Ji, particle):
    if echi <= mchi + dE:
        return 0
    return 2* dsigmadcos(echi, mchi, bgt, dE, eps, mA, Ji, particle)


def sigmaGTsum(echi, mchi, eps, mAs, Jis, nucl_exes, dE, particle):
    """
    echi: DM energy
    mchi: DM mass
    eps: coupling (epsilon)
    mAs: array of nuclear mass
    Jis: array of nuclear spin
    nucl_exes: array of nuclear excitation energies (should have same length as mA)
    dE: excitation energy = gamma energy = signal energy
    particle: fermion or scalar for DM
    return: total cross section [MeV^-2]
    """
    if len(mAs) != len(Jis) or len(mAs) != len(nucl_exes) or len(Jis) != len(nucl_exes):
        raise ValueError("mAs, Jis, and nucl_exes must have same length")

    if echi <= mchi + dE:
        return 0
    s = 0

    for mA, Ji, nucl_ex in zip(mAs, Jis, nucl_exes):
        gt_strength = conv_strength(dE, nucl_ex, width_keV=150)
        s += sigmaGT0(echi, mchi, gt_strength, dE, eps, mA, Ji, particle)

    return s


##########################################################
##########################################################
# Below is legacy code for GT cross section with Er integration

er_tolerance = 5e-3
def ermax(echi, mchi, dE, mA):
    # maximum recoil energy
    if echi <= mchi + dE:
        return 0
    pchi = momentum(echi, mchi)
    pchip = momentum(echi - dE, mchi)
    return (pchi + pchip)**2 / (2 * mA) * (1 - er_tolerance)

def ermin(echi, mchi, dE, mA):
    # minimum recoil energy
    if echi <= mchi + dE:
        return 0
    pchi = momentum(echi, mchi)
    pchip = momentum(echi - dE, mchi)
    return (pchi - pchip)**2 / (2 * mA) * (1 + er_tolerance)


def l33(echi, mchi, q, dE):
    """
    echi: dark matter energy
    mchi: dark matter mass
    q: momentum transfer
    dE: excitation energy
    """
    if echi <= mchi + dE:
        return 0
    pchi = momentum(echi, mchi)   # incoming dark matter momentum
    pchip = momentum(echi - dE, mchi) # outgoing dark matter momentum
    return 1 + ( -3/2* (pchi**2+pchip**2) + 3/2*q**2 - mchi**2 / 4  ) / (echi * (echi - dE))


def lterms(echi, mchi, q, dE, particle):
    if echi <= mchi + dE:
        return 0
    res = ldotl(echi, mchi, q, dE, particle) - l33(echi, mchi, q, dE)
    res = res / 2
    if res < 0:
        return 0
    return res


def dsigmadEr(er, echi, mchi, bgt, dE, eps, mA, Ji, particle):
    """
    GT differential cross section for dark matter scattering
    :param er: recoil energy
    :param echi: dark matter energy
    :param mchi: dark matter mass
    :param bgt: GT strength
    :param dE: excitation energy
    :param eps: coupling (epsilon)
    :param mA: atomic mass
    """
    if echi <= mchi + dE + er:
        return 0

    e = 0.30282212
    gD = np.sqrt(4*np.pi*alpha_D)
    mmed = 3*mchi
    pchi = momentum(echi, mchi)
    pchip = momentum(echi - dE - er, mchi)

    res = 2 * e**2 * eps**2 * gD**2 * (echi - er - dE)**2 / (pchi*pchip * (2*mA*er + mmed**2)**2)
    res *= mA / (2*np.pi) * (4*np.pi) / (2*Ji+1)
    res *= gA**2/(6*np.pi) * lterms(echi, mchi, np.sqrt(2*mA*er), dE, particle) * bgt
    return res


def sigmaGT(echi, mchi, bgt, dE, eps, mA, Ji, particle):
    # GT total cross section with Er integration
    if echi <= mchi + dE:
        return 0
    def rate(er):
        return dsigmadEr(er, echi, mchi, bgt, dE, eps, mA, Ji, particle)
    ermin_ = ermin(echi, mchi, dE, mA)
    ermax_ = ermax(echi, mchi, dE, mA)
    return quad(rate, ermin_, ermax_)[0]

