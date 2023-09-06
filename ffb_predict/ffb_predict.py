"""
Prediction with FFB model for JWST

author: Zhaozhou Li (lizz.astro@gmail.com)
"""
import numpy as np
from numpy import pi, log10, log, exp
from scipy.interpolate import CubicSpline, Akima1DInterpolator
from scipy.special import erf, roots_legendre
from scipy.interpolate import RegularGridInterpolator
from astropy import units as u
from h5attr import H5Attr

from colossus.cosmology import cosmology
from colossus.lss import mass_function, peaks


# cosmology
# -----------------------------
# params = {'flat': True, 'H0': 67.32, 'Om0': 0.3158, 'Ob0': 0.156 * 0.3158, 'sigma8': 0.812, 'ns': 0.96605}  # Planck15
params = {'flat': True, 'H0': 70, 'Om0': 0.3, 'Ob0': 0.048, 'sigma8': 0.82, 'ns': 0.95}
cosmo = cosmology.setCosmology('myCosmo', params)
fb = cosmo.Ob0 / cosmo.Om0  # 0.16

# Press-Schechter parameters
ps_ext_args = dict(
    ps_args={'model': 'eisenstein98'},
    sigma_args={'filt': 'tophat'},
    deltac_args={'corrections': True}
)

# global variables
# -----------------------------
UM_SHMR_MINSLOP = 0.3  # manual fix with minimum slop of dlnM*/dlnMh, can be None to disable
FFB_SFR_SIG = 0.3  # dex
FFB_SFR_TH_SIG = 0.15  # dex, transition width, equiv to ~0.6 dex full transition region
FFB_LGMH_PIVOT = 10.8  # Msun
FFB_LGMH_QUENCH = 12  # Msun, can be None to disable
FFB_SFE_MAX = 1  # maximum star formation efficiency for FFB galaxies
HALO_MAH_MODEL = 'Dekel13'  # Model for halo growth history, Dekel13 or Zhao09
DEKEL13_BETA = 0.14  # beta=0.14 in Dekel+13 eq. 7
DEKEL13_PIVOT = 12  # pivot mass
UV_FACTOR = 1  # not used


# math functions
# -----------------------------
def sigmoid(x, m=0, s=1, a=0, b=1):
    """
    varying from a to b, with m the midpoint and s the width

    example:
        x = linspace(-5, 5, 100)
        plot(x, sigmoid(x, s=1))
        abline((0, 0.5), 0.25)
    """
    return exp(-np.logaddexp(0, -(x - m) / s)) * (b - a) + a


def normal(x, m, s):
    return exp(-0.5 * ((x - m) / s)**2) / (s * (2 * pi)**0.5)


def lgnormal_mean(med, sig_dex):
    return med * exp(0.5 * (sig_dex * log(10))**2)


# Universe Machine functions
# -----------------------------
def um_vpeak_Mh_z(Mh, z):
    # peak historical halo mass, using Bryan & Norman virial overdensity.
    # B19, eq E2
    # Mh: Msun, vpeak: km/s
    a = 1.0 / (1.0 + z)
    Mz = 1.64e12 / ((a / 0.378)**-0.142 + (a / 0.378)**-1.79)
    vpeak = 200 * (Mh / Mz)**(1 / 3)
    return vpeak


def um_Mh_vpeak_z(vpeak, z):
    # peak historical halo mass, using Bryan & Norman virial overdensity.
    # B19, eq E2
    # Mh: Msun, vpeak: km/s
    a = 1.0 / (1.0 + z)
    Mz = 1.64e12 / ((a / 0.378)**-0.142 + (a / 0.378)**-1.79)
    Mh = Mz * (vpeak / 200)**3
    return Mh


def um_sfr_func1(z, x0, xa, xla, xz):
    a = 1.0 / (1.0 + z)
    return x0 + xa * (1 - a) + xla * log(1 + z) + xz * z


def um_sfr_func2(z, x0, xa, xz):
    a = 1.0 / (1.0 + z)
    return x0 + xa * (1 - a) + xz * z


def um_sfr_med_sf(lgMh, z):
    # B19, appendix H
    vpeak = um_vpeak_Mh_z(10**lgMh, z)  # km/s
    vz = 10**um_sfr_func1(z, 2.151, -1.658, 1.680, -0.233)  # km/s
    eps = 10**um_sfr_func1(z, 0.109, -3.441, 5.079, -0.781)  # Msun/yr
    alpha = um_sfr_func1(z, -5.598, -20.731, 13.455, -1.321)
    beta = um_sfr_func2(z, -1.911, 0.395, 0.747)
    gamma = 10**um_sfr_func2(z, -1.699, 4.206, -0.809)
    delta = 0.055

    # B19, eq 4, 5
    v = vpeak / vz
    SFR_SF = eps * ((v**alpha + v**beta)**(-1)
                    + gamma * exp(-0.5 * (log10(v) / delta)**2))
    return SFR_SF


def um_sfr_med_q(lgMh, z):
    # XXX
    Mstar = 10**um_lgMs_med(lgMh, z)  # Msun
    SFR_Q = 10**-11.8 * Mstar  # Msun/yr
    return SFR_Q


def um_sfr_sig_sf(lgMh, z):
    # B19, eq 3, appendix H
    # a = 1.0 / (1.0 + z)
    # sig_sf = np.fmin(-4.361 + 26.926 * (1 - a), 0.3)  # dex
    # sig_sf = np.clip(sig_sf, 1e-5, None)  # minimum sig
    sig_sf = 0.3  # dex, the above formula is problematic
    return sig_sf


def um_sfr_sig_q(lgMh, z):
    sig_q = 0.36  # dex
    return sig_q


def um_lgMhQ(z):
    # B19, eq 12-15, appendix H
    a = 1.0 / (1.0 + z)
    vq = 10**(2.248 - 0.018 * (1 - a) + 0.124 * z)
    Mh = um_Mh_vpeak_z(vq, z)
    return log10(Mh)


def um_fQ(lgMh, z):
    a = 1.0 / (1.0 + z)
    vpeak = um_vpeak_Mh_z(10**lgMh, z)  # km/s

    # B19, eq 12-15, appendix H
    qmin = np.fmax(0, -1.944 - 2.419 * (1 - a))
    vq = 10**(2.248 - 0.018 * (1 - a) + 0.124 * z)
    sigvq = 0.227 + 0.037 * (1 - a) - 0.107 * log(1 + z)  # B19, eq 14
    # note the typo in appendix H
    sigvq = np.fmax(0.01, sigvq)  # ZZ: ensure sigvq>=0.01

    x = log10(vpeak / vq) / (2**0.5 * sigvq)
    f_Q = qmin + (1 - qmin) * (0.5 + 0.5 * erf(x))  # eq 12

    return f_Q


def um_sfr(lgMh, z):
    SFR_SF = um_sfr_med_sf(lgMh, z)
    SFR_Q = um_sfr_med_q(lgMh, z)
    f_Q = um_fQ(lgMh, z)
    return f_Q * SFR_Q + (1 - f_Q) * SFR_SF


def um_lgMs_med_basic(lgMh, z, model='obs'):
    # Mh: Msun, Ms: Msun
    # B19, appendix J
    a = 1.0 / (1.0 + z)
    a1 = a - 1.0
    lna = log(a)

    if model == 'obs':
        # B19, Table J1: Obs, All, All, Excl
        (EFF_0, EFF_A, EFF_A2, EFF_Z,
            M1_0, M1_A, M1_A2, M1_Z,
            ALPHA_0, ALPHA_A, ALPHA_A2, ALPHA_Z,
            BETA_0, BETA_A, BETA_Z, DELTA_0,
            GAMMA_0, GAMMA_A, GAMMA_Z) = (
            -1.4346, 1.8313, 1.3683, -0.2169,
            12.0354, 4.5562, 4.4171, -0.7314,
            1.9633, -2.3156, -1.7321, 0.1776,
            0.4818, -0.8406, -0.4707, 0.4109,
            -1.0342, -3.1004, -1.0545)
    elif model == 'true':
        # B19, Table J1: True, All, All, Excl
        (EFF_0, EFF_A, EFF_A2, EFF_Z,
            M1_0, M1_A, M1_A2, M1_Z,
            ALPHA_0, ALPHA_A, ALPHA_A2, ALPHA_Z,
            BETA_0, BETA_A, BETA_Z, DELTA_0,
            GAMMA_0, GAMMA_A, GAMMA_Z) = (
            -1.4305, 1.7958, 1.3596, -0.2156,
            12.0400, 4.6752, 4.5131, -0.7444,
            1.9731, -2.3534, -1.7833, 0.1860,
            0.4732, -0.8843, -0.4861, 0.4068,
            -1.0879, -3.2414, -1.0785)
    lgm1 = M1_0 + a1 * M1_A - lna * M1_A2 + z * M1_Z
    eff = EFF_0 + a1 * EFF_A - lna * EFF_A2 + z * EFF_Z
    alpha = ALPHA_0 + a1 * ALPHA_A - lna * ALPHA_A2 + z * ALPHA_Z
    beta = BETA_0 + a1 * BETA_A + z * BETA_Z
    delta = DELTA_0
    gamma = 10**(GAMMA_0 + a1 * GAMMA_A + z * GAMMA_Z)

    x = lgMh - lgm1
    lgMs = lgm1 + (eff - log10(10**(-alpha * x) + 10**(-beta * x))
                   + gamma * exp(-0.5 * (x / delta)**2))
    return lgMs  # Msun


def um_lgMs_sig(lgMh, z):
    """
    x = linspace(10, 15, 100)
    plot(x, sigmoid(x, 12.5, s=1, a=0.3, b=0.2))
    grid(ls='--')
    xlim(10, 15)
    ylim(0, 0.5)
    """
    # sig = sigmoid(lgMh, 12.5, s=1, a=0.3, b=0.2)  # B19, fig 12
    sig = sigmoid(lgMh, 13.9 - z * 0.3, s=0.2, a=0.4, b=0.1)  # according to Han
    sig = sigmoid(lgMh, 13.9 - z * 0.3, s=0.2, a=0.4, b=0.3)  # added scatter due to scatter in MUV(Ms)
    return sig  # dex


def um_lgMs_med(lgMh, z):
    "ensure dlgMs/dlgMh >= UM_SHMR_MINSLOP"
    lgMs = um_lgMs_med_basic(lgMh, z)

    if UM_SHMR_MINSLOP is not None:
        lgMh_ = np.linspace(0, 16, 161)
        lgMs_ = um_lgMs_med_basic(lgMh_, z=z)

        f = CubicSpline(lgMh_, lgMs_)
        roots = f.derivative().solve(UM_SHMR_MINSLOP, extrapolate=False)
        if len(roots) == 0:
            lgMh_peak = lgMh_[-1]
        else:
            lgMh_peak = roots[0]
        lgMs_peak = f(lgMh_peak)

        if np.isscalar(lgMh):
            if lgMh > lgMh_peak:
                lgMs = lgMs_peak + UM_SHMR_MINSLOP * (lgMh - lgMh_peak)
        else:
            ix = lgMh > lgMh_peak
            lgMs[ix] = lgMs_peak + UM_SHMR_MINSLOP * (lgMh[ix] - lgMh_peak)

    return lgMs


def um_AUV(MUV, z, derivative=False):
    # B19, eq 23, 24
    M_dust = -20.594 - 0.054 * (np.fmax(4, z) - 4)
    alpha_dust = 0.559
    z = 10**(0.4 * alpha_dust * (M_dust - MUV))
    AUV = 2.5 * log10(1 + z)

    if derivative:
        dAUV_dMUV = - alpha_dust * z / (1 + z)
        return AUV, dAUV_dMUV
    else:
        return AUV


# FFB functions
# -----------------------------
def ffb_lgMh_crit(z):
    "Halo mass threshold for FFB"
    # lgMh_crit = 10.8 - 6.2 * log10((1 + z) / 10)  # D23, eq 62
    lgMh_crit = FFB_LGMH_PIVOT - 6.2 * log10((1 + z) / 10)  # D23, eq 62
    return lgMh_crit


def ffb_sfr_med(lgMh, z):
    sfr_avg = func_Mdot_baryon(lgMh, z=z) * FFB_SFE_MAX  # keep sfr_avg fixed
    fac_avg_med = exp(0.5 * (ffb_sfr_sig(lgMh, z=z) * log(10))**2)
    sfr_med = sfr_avg / fac_avg_med  # convert avg to median
    return sfr_med  # Msun/yr


def ffb_sfr_sig(lgMh, z):
    return FFB_SFR_SIG  # dex


def ffb_fFFB_sf(lgMh, z):
    lgMh_crit = ffb_lgMh_crit(z)
    f_FFB = sigmoid(lgMh, m=lgMh_crit, s=FFB_SFR_TH_SIG, a=0, b=1)
    if FFB_LGMH_QUENCH is not None:
        f_FFB *= sigmoid(lgMh, m=FFB_LGMH_QUENCH, s=FFB_SFR_TH_SIG, a=1, b=0)
    return f_FFB


# UV luminosity
# -----------------------------
def func_MUV_sfr_Salpeter(SFR, z):
    kappa_UV = 1.15e-28 * UV_FACTOR
    lum = SFR / kappa_UV * (u.erg / u.s / u.Hz)
    dist_lum = 10 * u.pc
    flux = lum / (4 * pi * dist_lum**2)
    MUV = flux.to(u.ABmag).value
    return MUV


def func_MUV_sfr(SFR, z):
    # Behroozi+2020, eq B3
    kappa_UV = 5.1e-29 * (1 + exp(-20.79 / (1 + z) + 0.98))
    lum = SFR / kappa_UV * (u.erg / u.s / u.Hz)
    dist_lum = 10 * u.pc
    flux = lum / (4 * pi * dist_lum**2)
    MUV = flux.to(u.ABmag).value
    return MUV


def func_MUV_lgMs(lgMs, z):
    MUV = -2.3 * (lgMs - 9) - 20.5 - 2.5 * log10(UV_FACTOR)
    return MUV


# Combining UM and FFB
# -----------------------------
def p_lgSFR_lgMh(lgSFR, lgMh, z):
    med_ffb = ffb_sfr_med(lgMh, z)
    sig_ffb = ffb_sfr_sig(lgMh, z)
    med_sf = um_sfr_med_sf(lgMh, z)
    sig_sf = um_sfr_sig_sf(lgMh, z)
    med_q = um_sfr_med_q(lgMh, z)
    sig_q = um_sfr_sig_q(lgMh, z)

    f_q = um_fQ(lgMh, z)
    f_FFB_ = ffb_fFFB_sf(lgMh, z)
    f_SF_ = 1 - f_q
    f_ffb = f_SF_ * f_FFB_
    f_sf = f_SF_ * (1 - f_FFB_)

    prob = (f_ffb * normal(lgSFR, log10(med_ffb), sig_ffb)
            + f_sf * normal(lgSFR, log10(med_sf), sig_sf)
            + f_q * normal(lgSFR, log10(med_q), sig_q)
            )
    return prob


def func_sfr_avg(lgMh, z):
    med_ffb = ffb_sfr_med(lgMh, z)
    sig_ffb = ffb_sfr_sig(lgMh, z)
    med_sf = um_sfr_med_sf(lgMh, z)
    sig_sf = um_sfr_sig_sf(lgMh, z)
    med_q = um_sfr_med_q(lgMh, z)
    sig_q = um_sfr_sig_q(lgMh, z)

    f_q = um_fQ(lgMh, z)
    f_FFB_ = ffb_fFFB_sf(lgMh, z)
    f_SF_ = 1 - f_q
    f_ffb = f_SF_ * f_FFB_
    f_sf = f_SF_ * (1 - f_FFB_)

    sfr_avg = (f_ffb * lgnormal_mean(med_ffb, sig_ffb)
               + f_sf * lgnormal_mean(med_sf, sig_sf)
               + f_q * lgnormal_mean(med_q, sig_q))
    return sfr_avg


def func_dif_sfr_avg(lgMh, z, Mdot):
    "difference of SFR between UM and FFB"
    med_sf = um_sfr_med_sf(lgMh, z)
    sig_sf = um_sfr_sig_sf(lgMh, z)

    f_q = um_fQ(lgMh, z)
    f_FFB_ = ffb_fFFB_sf(lgMh, z)
    f_SF_ = 1 - f_q
    f_ffb = f_SF_ * f_FFB_

    dif_sfr_avg = f_ffb * (Mdot * FFB_SFE_MAX - lgnormal_mean(med_sf, sig_sf))
    return dif_sfr_avg.clip(0)  # min: 0


def func_lgMs_med(lgMh, z, z_max=30, n_grid=200):
    # compute Mstar from the cumulative difference btw UM and FFB
    z_ob, lgM_ob = z, lgMh
    lgzp1_hist = np.linspace(log10(z_ob + 1), log10(z_max + 1), n_grid)
    z_hist = 10**lgzp1_hist - 1
    lgM_hist = mah_interp(z_ob, lgM_ob, z_hist)
    Mdot_hist = 10**mah_der_interp(z_ob, lgM_ob, z_hist) * fb

    dif_sfr = func_dif_sfr_avg(lgM_hist, z_hist, Mdot_hist)

    dtdlgzp1_hist = cosmo.lookbackTime(
        z_hist, derivative=1) * log(10) * (1 + z_hist) * 1e9  # yr/dex

    func = CubicSpline(lgzp1_hist, dif_sfr * dtdlgzp1_hist, axis=1).antiderivative()
    dif_Ms = func(lgzp1_hist[-1]) - func(lgzp1_hist[0])

    # UM mass and FFB correction
    Ms_um = 10**um_lgMs_med(lgMh, z)
    Ms = Ms_um + dif_Ms

    return log10(Ms)


def um_lgMs_med_integ(lgMh, z, z_max=30, n_grid=200):
    # compute UM Ms from cumulative SFR, for consistency check
    z_ob, lgM_ob = z, lgMh
    lgzp1_hist = np.linspace(log10(z_ob + 1), log10(z_max + 1), n_grid)
    z_hist = 10**lgzp1_hist - 1

    lgM_hist = mah_interp(z_ob, lgM_ob, z_hist)
    sfr = um_sfr(lgM_hist, z_hist)

    dtdlgzp1_hist = cosmo.lookbackTime(
        z_hist, derivative=1) * log(10) * (1 + z_hist) * 1e9  # yr/dex

    func = CubicSpline(lgzp1_hist, sfr * dtdlgzp1_hist, axis=1).antiderivative()
    Ms = func(lgzp1_hist[-1]) - func(lgzp1_hist[0]) + 1
    return log10(Ms)


def func_lgMh_med_integ(lgMh, z, z_max=30, n_grid=200):
    # compute halo mass from cumulative Mdot, for consistency check
    z_ob, lgM_ob = z, lgMh
    lgzp1_hist = np.linspace(log10(z_ob + 1), log10(z_max + 1), n_grid)
    z_hist = 10**lgzp1_hist - 1

    # lgM_hist = mah_interp(z_ob, lgM_ob, z_hist)
    Mdot = 10**mah_der_interp(z_ob, lgM_ob, z_hist)

    dtdlgzp1_hist = cosmo.lookbackTime(
        z_hist, derivative=1) * log(10) * (1 + z_hist) * 1e9  # yr/dex

    func = CubicSpline(lgzp1_hist, Mdot * dtdlgzp1_hist, axis=1).antiderivative()
    Mh = func(lgzp1_hist[-1]) - func(lgzp1_hist[0]) + 1
    return log10(Mh)


def p_lgMs_lgMh(lgMs, lgMh, z):
    lgMs_med = func_lgMs_med(lgMh, z)
    lgMs_sig = um_lgMs_sig(lgMh, z)

    prob = normal(lgMs, lgMs_med, lgMs_sig)
    return prob


def func_Mdot_baryon(lgMh, z):
    if not np.isscalar(z):
        raise ValueError('z should be a scalar')

    Mdot = 10**mah_der_interp(z, lgMh, z) * fb
    return Mdot  # Msun/yr


def func_sfr_SFE(lgMh, z):
    SFR = func_sfr_avg(lgMh, z)
    Mdot_baryon = func_Mdot_baryon(lgMh, z)
    return SFR / Mdot_baryon


# Halo mass func and number densities
# -----------------------------
def func_dN_dlgMsub_cond(lgMsub, lgMh):
    "unevolved subhalo mass funct, Han+ 2017, HBT+, table 1 and eq 4"
    a1, alpha1, a2, alpha2, b, beta = 0.11, 0.95, 0.32, 0.08, 8.9, 1.9
    mu = 10**(lgMsub - lgMh)
    dndlgmu = (a1 * mu**-alpha1 + a2 * mu**-alpha2) * exp(-b * mu**beta) * log(10)
    return dndlgmu


class HaloMassFunc:
    def __init__(self, z, compute_sub=False):
        nu_max = 10  # maximum peak height, lgM_max=16.7 at z=0, 11.6 at z=18
        nu_max = 8  # maximum peak height, lgM_max=16.5 at z=0, 10.8 at z=18
        self.lgMh_max = log10(peaks.massFromPeakHeight(nu=nu_max, z=z, **ps_ext_args) / cosmo.h)
        self.z = z

        # central MF
        lgMcen = np.arange(5, self.lgMh_max + 0.001, 0.025)
        dNdlgMcen = mass_function.massFunction(
            10**lgMcen * cosmo.h, z=z, mdef='vir', model='Watson13'.lower(),
            q_out='dndlnM', **ps_ext_args) * cosmo.h**3 * log(10)  # comoving Mpc^-3
        self.func_lgdNdlgM_cen = Akima1DInterpolator(lgMcen, log10(dNdlgMcen))
        self.lgMcen, self.dNdlgMcen = lgMcen, dNdlgMcen

        # not used
        if compute_sub:
            # prepare grids
            x0, w0 = roots_legendre(80)
            x0, w0 = x0 * 0.5 + 0.5, w0 * 0.5

            # sub MF
            lgMsub = lgMcen
            lgMcen_ = lgMsub + (self.lgMh_max - lgMsub) * x0.reshape(-1, 1)
            dNdlgMsub_cond = func_dN_dlgMsub_cond(lgMsub, lgMcen_)
            dNdlgMcen_ = 10**self.func_lgdNdlgM_cen(lgMcen_)
            dNdlgMsub = (dNdlgMsub_cond * dNdlgMcen_ * w0.reshape(-1, 1)).sum(0)
            self.func_lgdNdlgM_sub = Akima1DInterpolator(lgMsub, log10(dNdlgMsub))

        self.compute_dNdlgSFR()
        self.compute_dNdMUV_SFR()

        self.compute_dNdlgMs()
        self.compute_dNdMUV_Ms()

    def compute_dNdlgSFR(self):
        lgSFR = np.linspace(-6, 6, 121)
        lgMh, dNdlgMh = self.lgMcen, self.dNdlgMcen
        pr_lgSFR_lgMh = p_lgSFR_lgMh(lgSFR.reshape(-1, 1), lgMh, self.z)
        dNdlgSFR = simps1d(pr_lgSFR_lgMh * dNdlgMh, dx=lgMh[1] - lgMh[0])
        self.func_lgdNdlgSFR = Akima1DInterpolator(lgSFR, log10(dNdlgSFR))
        self.lgSFR, self.dNdlgSFR = lgSFR, dNdlgSFR

    def compute_dNdMUV_SFR(self):
        MUV = func_MUV_sfr(10**self.lgSFR, z=self.z)
        dNdMUV = self.dNdlgSFR / 2.5
        self.func_lgdNdMUV_SFR = Akima1DInterpolator(MUV[::-1], log10(dNdMUV)[::-1])
        self.MUV_SFR, self.dNdMUV_SFR = MUV, dNdMUV

        # dust extinction
        AUV, dAUV_dMUV = um_AUV(MUV, z=self.z, derivative=True)
        self.MUV_SFR_obs = MUV + AUV
        self.dNdMUV_SFR_obs = dNdMUV / np.abs(1 + dAUV_dMUV)
        self.func_lgdNdMUV_SFR_obs = Akima1DInterpolator(self.MUV_SFR_obs[::-1], log10(self.dNdMUV_SFR_obs)[::-1])

    def compute_dNdlgMs(self):
        lgMh, dNdlgMh = self.lgMcen, self.dNdlgMcen
        lgMs = np.arange(5, self.lgMh_max - 1 + 0.001, 0.025)
        pr_lgMs_lgMh = p_lgMs_lgMh(lgMs.reshape(-1, 1), lgMh, self.z)
        dNdlgMs = simps1d(pr_lgMs_lgMh * dNdlgMh, dx=lgMh[1] - lgMh[0])
        self.func_lgdNdlgMs = Akima1DInterpolator(lgMs, log10(dNdlgMs))
        self.lgMs, self.dNdlgMs = lgMs, dNdlgMs

    def compute_dNdMUV_Ms(self):
        MUV = func_MUV_lgMs(self.lgMs, z=self.z)
        dNdMUV = self.dNdlgMs / 2.3  # XXX
        self.func_lgdNdMUV_Ms = Akima1DInterpolator(MUV[::-1], log10(dNdMUV)[::-1])
        self.MUV_Ms, self.dNdMUV_Ms = MUV, dNdMUV

        # dust extinction
        AUV, dAUV_dMUV = um_AUV(MUV, z=self.z, derivative=True)
        self.MUV_Ms_obs = MUV + AUV
        self.dNdMUV_Ms_obs = dNdMUV / np.abs(1 + dAUV_dMUV)
        self.func_lgdNdMUV_Ms_obs = Akima1DInterpolator(self.MUV_Ms_obs[::-1], log10(self.dNdMUV_Ms_obs)[::-1])


# Halo growth history
# -----------------------------
def mah_interp(z_ob, lgM_ob, z_hist):
    if HALO_MAH_MODEL == 'Zhao09':
        return mah_interp_Zhao09(z_ob, lgM_ob, z_hist)
    elif HALO_MAH_MODEL == 'Dekel13':
        return mah_interp_Dekel13(z_ob, lgM_ob, z_hist)
    else:
        raise ValueError("HALO_MAH_MODEL' should be Dekel13 or Zhao09")


def mah_der_interp(z_ob, lgM_ob, z_hist):
    if HALO_MAH_MODEL == 'Zhao09':
        return mah_der_interp_Zhao09(z_ob, lgM_ob, z_hist)
    elif HALO_MAH_MODEL == 'Dekel13':
        return mah_der_interp_Dekel13(z_ob, lgM_ob, z_hist)
    else:
        raise ValueError("HALO_MAH_MODEL' should be Dekel13 or Zhao09")


def mah_interp_Zhao09(z_ob, lgM_ob, z_hist):
    """
    M(z_hist|lgM_ob, z_ob)
    lgM should be between 1.5e5Msun and 8 sigma peak.
    z should be between 0 and 30.
    """
    fp = H5Attr('mandc-1.03main/run/mchistory_cDekel.h5', lazy=False)
    z_list = np.sort([fp[f'lg(z+1)/{key}/ziz'][0] for key in fp['lg(z+1)']])
    lgzp1_list = log10(z_list + 1)

    lgzp1_ob = log10(z_ob + 1)
    lgzp1_hist = log10(z_hist + 1)

    ix_z = lgzp1_list.searchsorted(lgzp1_ob, side='right') - 1  # find closest z bin
    gp = fp[f'lg(z+1)/{lgzp1_list[ix_z]:.2f}']

    lgM_z = log10(gp.Miz)
    lgM_i = lgM_z.T[0]
    lgzp1_i = log10(gp.ziz + 1)
    mah_interp = RegularGridInterpolator([lgM_i, lgzp1_i], lgM_z,
                                         method='linear', bounds_error=False)

    x, y = np.broadcast_arrays(lgM_i, lgzp1_ob)
    lgM_zob = mah_interp(np.stack([x, y], axis=-1))  # lgM[z_ob | lgM_i, z_0]
    lgM_i_ = np.interp(lgM_ob, lgM_zob, lgM_i)  # lgM[z_0 | lgM_ob, z_ob]

    x, y = np.meshgrid(lgM_i_, lgzp1_hist, indexing='ij')
    lgM_hist = mah_interp(np.stack([x, y], axis=-1))

    if np.isscalar(z_hist):
        return lgM_hist.reshape(np.shape(lgM_ob))
    else:
        return lgM_hist  # shape: lgM_ob, z_hist


def mah_der_interp_Zhao09(z_ob, lgM_ob, z_hist):
    """
    dM/dt(z_hist|lgM_ob, z_ob)
    lgM should be between 1.5e5Msun and 8 sigma peak.
    z should be between 0 and 30.
    """
    fp = H5Attr('mandc-1.03main/run/mchistory_cDekel.h5', lazy=False)
    z_list = np.sort([fp[f'lg(z+1)/{key}/ziz'][0] for key in fp['lg(z+1)']])
    lgzp1_list = log10(z_list + 1)

    lgzp1_ob = log10(z_ob + 1)
    lgzp1_hist = log10(z_hist + 1)

    ix_z = lgzp1_list.searchsorted(lgzp1_ob, side='right') - 1  # find closest z bin
    gp = fp[f'lg(z+1)/{lgzp1_list[ix_z]:.2f}']

    lgM_z = log10(gp.Miz)
    lgMdot_z = log10(gp.Mdot)
    lgM_i = lgM_z.T[0]
    lgzp1_i = log10(gp.ziz + 1)
    mah_interp = RegularGridInterpolator([lgM_i, lgzp1_i], lgM_z,
                                         method='linear', bounds_error=False)

    x, y = np.broadcast_arrays(lgM_i, lgzp1_ob)
    lgM_zob = mah_interp(np.stack([x, y], axis=-1))  # lgM[z_ob | lgM_i, z_0]
    lgM_i_ = np.interp(lgM_ob, lgM_zob, lgM_i)  # lgM[z_0 | lgM_ob, z_ob]

    mdot_interp = RegularGridInterpolator([lgM_i, lgzp1_i], lgMdot_z,
                                          method='linear', bounds_error=False)

    x, y = np.meshgrid(lgM_i_, lgzp1_hist, indexing='ij')
    lgMdot_hist = mdot_interp(np.stack([x, y], axis=-1))

    if np.isscalar(z_hist):
        return lgMdot_hist.reshape(np.shape(lgM_ob))
    else:
        return lgMdot_hist  # shape: lgM_ob, z_hist


def mah_interp_Dekel13(z_ob, lgM_ob, z_hist):
    s = 0.03  # 0.03 Gyr^-1
    t1 = 17.00  # Gyr, H0=70, omegam=0.3
    alpha = 1.5 * s * t1

    lgM_ob_T = lgM_ob if np.isscalar(lgM_ob) else lgM_ob.reshape(-1, 1)  # transpose of lgM_ob

    if DEKEL13_BETA == 0:
        lgM_hist = lgM_ob_T - alpha * (z_hist - z_ob) / log(10)
    else:
        beta, lgM_c0 = DEKEL13_BETA, abs(DEKEL13_PIVOT)
        if DEKEL13_PIVOT > 0:
            lgM_hist = lgM_c0 - log10(10**(-beta * (lgM_ob_T - lgM_c0)) + alpha * beta * (z_hist - z_ob)) / beta
        else:
            # negative for varying pivot
            lgM_hist = lgM_c0 - log10(10**(-beta * (lgM_ob_T - lgM_c0)) + exp(alpha * beta * (z_hist - z_ob)) - 1) / beta

    if np.isscalar(z_hist):
        return lgM_hist.reshape(np.shape(lgM_ob))
    else:
        return lgM_hist  # shape: lgM_ob, z_hist


def mah_der_interp_Dekel13(z_ob, lgM_ob, z_hist):
    s = 0.03  # 0.03 Gyr^-1
    t1 = 17.00  # Gyr, H0=70, omegam=0.3
    alpha = 1.5 * s * t1

    lgM_hist = mah_interp_Dekel13(z_ob, lgM_ob, z_hist)
    lgMdot_hist = lgM_hist + log10(s * (1 + z_hist)**2.5) - 9

    if DEKEL13_BETA != 0:
        beta, lgM_c0 = DEKEL13_BETA, abs(DEKEL13_PIVOT)
        if DEKEL13_PIVOT > 0:
            lgM_c = lgM_c0
        else:
            lgM_c = lgM_c0 - alpha * (z_hist - z_ob) / log(10)
        lgMdot_hist += beta * (lgM_hist - lgM_c)

    if np.isscalar(z_hist):
        return lgMdot_hist.reshape(np.shape(lgM_ob))
    else:
        return lgMdot_hist  # shape: lgM_ob, z_hist


# FFB galaxy size
# -----------------------------

def ffb_rdisc(lgMh, z):
    """
    Mh: Msun
    z:
    rdisc: kpc
    """
    rdisc = 0.31 * 10**((lgMh - 10.8) / 3) * ((1 + z) / 10)**-1
    return rdisc


def ffb_rshell(lgMh, z):
    """
    Mh: Msun
    z:
    rdisc: kpc
    """
    rsh = 0.79 * 10**((lgMh - 10.8) * -0.06) * ((1 + z) / 10)**-2.5
    return rsh


def ffb_rdisk_Mcrit(z):
    return 0.29 * ((1 + z) / 10)**-3.07


def ffb_rshell_Mcrit(z):
    return 0.79 * ((1 + z) / 10)**-2.13


def ffb_lgMcrit_disc(z):
    "Halo mass threshold for FFB"
    lgMh_crit = 10.8 + log10(0.8) - 6.2 * log10((1 + z) / 10)  # D23, eq 62
    return lgMh_crit


def ffb_lgMcrit_shell(z):
    "Halo mass threshold for FFB"
    lgMh_crit = 10.8 + log10(1) - 6.2 * log10((1 + z) / 10)  # D23, eq 62
    return lgMh_crit


# FFB dust attenuation
# -----------------------------
def AUV_tau(tau):
    "convert tau to AUV"
    return 2.5 * tau / log(10)


def ffb_f_sfe(sfe):
    return (5 * sfe * (1 - 0.8 * sfe))**0.5


def ffb_tau_shell(sfe, lgMh, z):
    "UV optical depth for shell scenario, Li+23, eq 29"
    return (1.65 + 0.27) * ffb_f_sfe(sfe) * 10**(1.2 * (lgMh - 10.8)) * ((1 + z) / 10)**5


def ffb_tau_disc(sfe, lgMh, z):
    "UV optical depth for disc scenario, Li+23, eq 29"
    return (2.10 + 0.35) * ffb_f_sfe(sfe) * 10**(0.81 * (lgMh - 10.8)) * ((1 + z) / 10)**3.5


def ffb_AUV_shell(lgMh, z):
    sfe = func_sfr_SFE(lgMh, z)
    tau = ffb_tau_shell(sfe, lgMh, z)
    AUV = AUV_tau(tau)
    return AUV


def ffb_AUV_disc(lgMh, z):
    sfe = func_sfr_SFE(lgMh, z)
    tau = ffb_tau_disc(sfe, lgMh, z)
    AUV = AUV_tau(tau)
    return AUV


# integration functions
# -----------------------------
# author: Zhaozhou Li
# source: https://github.com/syrte/handy/blob/master/integrate.py

def slice_set(ix, ndim, axis):
    # author: Zhaozhou Li
    # source: https://github.com/syrte/handy/blob/master/integrate.py
    ix_list = [slice(None)] * ndim
    ix_list[axis] = ix
    return tuple(ix_list)


def sum2d(a):
    """sum of last two dimensions
    """
    # author: Zhaozhou Li
    # source: https://github.com/syrte/handy/blob/master/integrate.py
    return a.reshape(*a.shape[:-2], -1).sum(-1)


def simps1d(y, dx=1.0, axis=-1, even='avg'):
    # author: Zhaozhou Li
    # source: https://github.com/syrte/handy/blob/master/integrate.py
    y = np.asarray(y)
    ndim = y.ndim

    # when shape of y is odd
    if y.shape[axis] % 2 == 1:
        ix0 = slice_set(0, ndim, axis)
        ix1 = slice_set(-1, ndim, axis)
        ixo = slice_set(slice(1, -1, 2), ndim, axis)  # odd
        ixe = slice_set(slice(2, -2, 2), ndim, axis)  # even
        out = (y[ix0] + y[ix1] + 4 * y[ixo].sum(axis) + 2 * y[ixe].sum(axis)) * (dx / 3)
        return out
    elif even == 'avg':
        ix0 = slice_set(0, ndim, axis)
        ix1 = slice_set(-1, ndim, axis)
        ix2 = slice_set(1, ndim, axis)
        ix3 = slice_set(-2, ndim, axis)
        ix4 = slice_set(slice(2, -2), ndim, axis)
        out = (2.5 * (y[ix0] + y[ix1]) + 6.5 * (y[ix2] + y[ix3]) +
               6 * y[ix4].sum(axis)) * (dx / 6)
        return out
    elif even == 'first':
        ix0 = slice_set(-1, ndim, axis)
        ix1 = slice_set(-2, ndim, axis)
        ix3 = slice_set(slice(None, -1), ndim, axis)
        return simps1d(y[ix3], dx, axis) + 0.5 * dx * (y[ix0] + y[ix1])
    elif even == 'last':
        ix0 = slice_set(0, ndim, axis)
        ix1 = slice_set(1, ndim, axis)
        ix3 = slice_set(slice(1, None), ndim, axis)
        return simps1d(y[ix3], dx, axis) + 0.5 * dx * (y[ix0] + y[ix1])
    else:
        raise ValueError("'even' must be one of 'avg', 'first' or 'last'")


def simps2d(z, dx=1, dy=1):
    """integrate over last two dimensions

    >>> simps2d(np.ones((5, 5)))
    16.0
    """
    # author: Zhaozhou Li
    # source: https://github.com/syrte/handy/blob/master/integrate.py
    z = np.asarray(z)
    nx, ny = z.shape[-2:]
    if nx % 2 != 1 or ny % 2 != 1:
        raise ValueError('input array should be odd shape')

    ixo = slice(1, -1, 2)  # odd
    ixe = slice(2, -2, 2)  # even

    # corner points, with weight 1
    s1 = (z[..., 0, 0] + z[..., 0, -1] + z[..., -1, 0] + z[..., -1, -1])

    # edges excluding corners, with weight 2 or 4
    s2 = 2 * (z[..., 0, ixe].sum(-1) + z[..., -1, ixe].sum(-1) +
              z[..., ixe, 0].sum(-1) + z[..., ixe, -1].sum(-1))
    s3 = 4 * (z[..., 0, ixo].sum(-1) + z[..., -1, ixo].sum(-1) +
              z[..., ixo, 0].sum(-1) + z[..., ixo, -1].sum(-1))

    # interior points, with weight 4, 8 or 16
    s4 = (4 * sum2d(z[..., ixe, ixe]) + 16 * sum2d(z[..., ixo, ixo]) +
          8 * sum2d(z[..., ixe, ixo]) + 8 * sum2d(z[..., ixo, ixe]))

    out = (s1 + s2 + s3 + s4) * (dx * dy / 9)
    return out
