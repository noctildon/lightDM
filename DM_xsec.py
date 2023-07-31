from util import *
from exp_config import *
from pathlib import Path


def dsigmadEr_el(er, echi, mchi, eps, mA, Z, helm=True):
    """
    Elastic DM-nucleus differential cross section
    er: recoil energy
    echi: dark matter energy
    mchi: dark matter mass
    eps: coupling (epsilon)
    mA: nuclear mass
    Ji: nuclear spin
    Z: atomic number
    helm: True if using Helm form factor
    return: elastic differential cross section [MeV^-2]
    """
    if echi <= mchi + er or er >= ermax(echi, mchi, dE=0, mA=mA):
        return 0

    helm_form = FHelm(mA/mN, er)**2 if helm else 1

    gD = np.sqrt(4*np.pi*alpha_D)
    mmed = 3*mchi
    pchi = momentum(echi, mchi)

    res = e_charge**2 * eps**2 * gD**2 * Z**2 * helm_form / (4*np.pi)
    res /= pchi**2 * (2*mA*er + mmed **2)**2
    res *= 2*echi**2 * mA * (1- er/echi - mA*er/(2*echi**2)) + er**2 * mA
    return res


def ldotl(echi, mchi, mA, q, dE, particle):
    """
    echi: DM energy
    mchi: DM mass
    mA: nuclear mass
    q: momentum transfer
    dE: excitation energy
    particle: fermion or scalar for DM
    return: the curren term l dot l
    """
    er = q**2 / (2*mA)
    if echi <= mchi + dE + er:
        return 0
    pchi = momentum(echi, mchi)   # incoming dark matter momentum
    pchip = momentum(echi - dE - er, mchi) # outgoing dark matter momentum

    if particle == 'fermion':
        return 3- ( .5*(pchi**2 + pchip**2 - q**2) +3/4* mchi**2 ) / (echi * (echi - dE - er))
    elif particle == 'scalar':
        return (pchi**2 + pchip**2 - q**2) / (2*echi * (echi - dE - er))
    else:
        raise Exception('DM particle must be fermion or scalar')


def dsigmadcos(cos_theta, echi, mchi, mmed, bgt, dE, eps, mA, Ji, particle='fermion', q_bound=None):
    """
    Inelastic DM-nucleus differential cross section
    cos_theta: cosine of scattering angle
    echi: dark matter energy
    mchi: dark matter mass
    mmed: mediator mass
    bgt: GT strength
    dE: excitation energy
    eps: coupling (epsilon)
    mA: nuclear mass
    Ji: nuclear spin
    q_bound: momentum transfer upper bound
    return: GT differential cross section for dark matter scattering
    """
    gD = np.sqrt(4*np.pi*alpha_D)

    pchi = momentum(echi, mchi)
    pchip = momentum(echi - dE, mchi)

    # recoil momentum squared
    q_squared = pchi**2 + pchip**2 - 2*pchi*pchip*cos_theta
    if q_bound is not None:
        if q_squared > q_bound**2:
            return 0

    res = 2 * e_charge**2 * eps**2 * gD**2 * (echi - dE) * pchip / (q_squared + mmed**2 - dE**2)**2
    res *= 1 / (2*np.pi) * (4*np.pi) / (2*Ji+1)
    res *= gA**2/(12*np.pi) * ldotl(echi, mchi, mA, np.sqrt(q_squared), dE, particle) * 2.0*bgt # 2 is convention for bgt

    return res


def sigmaGT_theta(echi, mchi, mmed, bgt, dE, eps, mA, Ji, particle='fermion', q_bound=None):
    """
    GT total cross section with cos_theta integration
    """
    if echi <= mchi + dE:
        return 0

    # integrate over cos_theta
    def ff(cos_theta):
        return dsigmadcos(cos_theta, echi, mchi, mmed, bgt, dE, eps, mA, Ji, particle, q_bound)
    return quad(ff, -1, 1)[0]


def dsigmadEr(er, echi, mchi, mmed, bgt, dE, eps, mA, Ji, particle, q_bound=None):
    """
    Inelastic DM-nucleus differential cross section
    er: recoil energy
    echi: dark matter energy
    mchi: dark matter mass
    mmed: mediator mass
    bgt: GT strength
    dE: excitation energy
    eps: coupling (epsilon)
    mA: nuclear mass
    Ji: nuclear spin
    return: GT differential cross section for dark matter scattering
    """
    if echi <= mchi + dE + er:
        return 0

    gD = np.sqrt(4*np.pi*alpha_D)
    pchi = momentum(echi, mchi)
    pchip = momentum(echi - dE - er, mchi)

    er_min = ermin(echi, mchi, dE, mA)
    er_max = ermax(echi, mchi, dE, mA)
    if er < er_min or er > er_max:
        return 0

    if q_bound is not None:
        if np.sqrt(2*mA*er) > q_bound:
            return 0

    res = 2 * e_charge**2 * eps**2 * gD**2 * (echi - er - dE) * pchip / (pchi*pchip * (2*mA*er + mmed**2 - dE**2)**2)
    res *= mA / (2*np.pi) * (4*np.pi) / (2*Ji+1)
    res *= gA**2/(12*np.pi) * ldotl(echi, mchi, mA, np.sqrt(2*mA*er), dE, particle) * 2.0*bgt # 2 is convention for bgt
    return res


# GT total cross section with Er integration
def sigmaGT_er(echi, mchi, mmed, bgt, dE, eps, mA, Ji, particle='fermion', q_bound=None):
    if echi <= mchi + dE:
        return 0
    def ff(er):
        return dsigmadEr(er, echi, mchi, mmed, bgt, dE, eps, mA, Ji, particle, q_bound)
    er_max = ermax(echi, mchi, dE, mA)
    er_min = ermin(echi, mchi, dE, mA)
    return quad(ff, er_min, er_max)[0]


def sigmaGTsum(echi, mchi, mmed, eps, mAs, Jis, nucl_exes, dE, particle='fermion', conv_width=1, q_bound=None):
    """
    echi: DM energy
    mchi: DM mass
    mmed: mediator mass
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
        gt_strength = conv_strength(dE, nucl_ex, conv_width=conv_width)

        # s += sigmaGT_er(echi, mchi, mmed, gt_strength, dE, eps, mA, Ji, particle) # integrate over Er
        s += sigmaGT_theta(echi, mchi, mmed, gt_strength, dE, eps, mA, Ji, particle, q_bound) # integrate over theta

    return s


def dsigmadErsum(er, echi, mchi, mmed, eps, mAs, Jis, nucl_exes, dE, particle='fermion', conv_width=1, q_bound=None):
    if len(mAs) != len(Jis) or len(mAs) != len(nucl_exes) or len(Jis) != len(nucl_exes):
        raise ValueError("mAs, Jis, and nucl_exes must have same length")

    if echi <= mchi + dE:
        return 0

    s = 0
    for mA, Ji, nucl_ex in zip(mAs, Jis, nucl_exes):
        er_min = ermin(echi, mchi, dE, mA)
        er_max = ermax(echi, mchi, dE, mA)
        if er < er_min or er > er_max:
            continue

        gt_strength = conv_strength(dE, nucl_ex, conv_width=conv_width)
        s += dsigmadEr(er, echi, mchi, mmed, gt_strength, dE, eps, mA, Ji, particle, q_bound)

    return s


def sigmaGTsum_er(echi, mchi, mmed, eps, mAs, Jis, nucl_exes, dE, particle='fermion', conv_width=1, q_bound=None):
    """
    Integrate dsigmadEr over recoil energy. Should be the same as sigmaGTsum(), but much slower
    """
    if len(mAs) != len(Jis) or len(mAs) != len(nucl_exes) or len(Jis) != len(nucl_exes):
        raise ValueError("mAs, Jis, and nucl_exes must have same length")

    def ff(er):
        return dsigmadErsum(er, echi, mchi, mmed, eps, mAs, Jis, nucl_exes, dE, particle, conv_width, q_bound)

    global_ermax = ermax(echi, mchi, dE, max(mAs))
    global_ermin = ermin(echi, mchi, dE, min(mAs))
    return quad(ff, global_ermin, global_ermax)[0]


if __name__ == "__main__":
    s1 = sigmaGT_er(echi=20.1, mchi=10, mmed=30, bgt=1, dE=10, eps=1e-2, mA=40e3, Ji=0)
    s2 = sigmaGT_theta(echi=20.1, mchi=10, mmed=30, bgt=1, dE=10, eps=1e-2, mA=40e3, Ji=0)
    print(s1, s2)

    mAs, Jis, Zs, nucl_exes = get_mA_Ji('CCM200')
    s0 = sigmaGTsum(echi=30, mchi=10, mmed=30, eps=0.1, mAs=mAs, Jis=Jis, nucl_exes=nucl_exes, dE=3)
    s1 = sigmaGTsum_er(echi=30, mchi=10, mmed=30, eps=0.1, mAs=mAs, Jis=Jis, nucl_exes=nucl_exes, dE=3)
    print('after sum and intergation:', s0, s1)
