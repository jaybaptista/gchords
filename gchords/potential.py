from abc import ABC, abstractmethod
import os
import symlib
import agama
import numpy as np
from tqdm import tqdm

agama.setUnits(length=1.0, mass=1.0, velocity=1.0)

def agama_times(cosmology, scale_factors):
    # cosmic age at each scale factor in Agama internal time units (~0.97779222 Gyr)
    return cosmology.age(1/np.asarray(scale_factors) - 1) / 0.97779222

# ---------------------------------------------------------------------------------------------------------------------
# Spherical (Einasto / NFW) profile fits, ported from the old mwgcs package (getConvergenceRadius in mwgcs/interface.py
# and SymphonyHaloProfile in mwgcs/fit.py) via energy_sorting/test_inner_extrapolation.ipynb.
# ---------------------------------------------------------------------------------------------------------------------

PROFILE_UNTRACKED, PROFILE_EINASTO, PROFILE_NFW = -1, 0, 1
# why spherical_profile used the NFW fallback instead of the Einasto fit
FALLBACK_NONE, FALLBACK_FEW_PARTICLES, FALLBACK_FEW_BINS, FALLBACK_FIT_FAILED, FALLBACK_AT_BOUND = 0, 1, 2, 3, 4

def host_potential_at(params, x_host, ok_host, x):
    '''
    Potential (km^2/s^2) at host-centric positions x (physical kpc) of the spherical host potential used by
    SymphonyPotential.circular_energy_radius: symlib.profile_info's potential of the valid host particles, with its
    PE / vmax**2 scaling and its 4 G M / R_out outer boundary term (should be G M / R_out) undone.

    The positions x are passed to profile_info as zero-mass particles alongside the host particles, so U(x) follows
    exactly the same discretisation (and eps softening) as the host particles' own U. The corrected U is independent
    of where the outer boundary R_out falls, so positions beyond every host particle get -G M_tot / r. Rows of x with
    NaNs are returned as NaN.
    '''
    x = np.atleast_2d(x)
    U = np.full(len(x), np.nan)
    good = np.all(np.isfinite(x), axis=1)
    if not np.any(good) or np.sum(ok_host) == 0:
        return U
    x_all = np.concatenate([x_host, x[good]])
    ok_all = np.concatenate([ok_host, np.zeros(np.sum(good), dtype=bool)])
    _, vmax, pe, _ = symlib.profile_info(params, x_all, ok_all)
    r_out = np.sqrt(np.nanmax(np.linalg.norm(x_all, axis=1))**2 + params['eps']**2)
    mp = params['mp'] / params['h100']
    U[good] = pe[len(x_host):] * vmax**2 + 3 * agama.G * np.sum(ok_host) * mp / r_out
    return U

def get_convergence_radius(params, a):
    '''max(0.055 x mean interparticle spacing, 3 eps), both in physical kpc, at scale factor a.'''
    H0 = params["h100"] * 3.2407792894443648e-18  # 1/s
    _G = 4.517103049894965e-39  # kpc^3 / Msun / s^2
    rho_m = params["Om0"] * (3 / (8 * np.pi * _G)) * H0**2
    l_0 = (params["mp"] / params["h100"] / rho_m) ** (1 / 3)
    return np.max((5.5e-2 * a * l_0, 3 * params["eps"] / params["h100"] * a))

def get_density_profile(r, mp, rvir, bins=50):
    '''Density in `bins` equal-count radial bins of the particles inside rvir; bin radius is the mean particle radius.'''
    _r = np.sort(r[r < rvir])
    bin_r = np.array_split(_r, bins)
    bin_edges = np.concatenate([[0], [b[-1] for b in bin_r]])  # rightmost particle of each bin
    bin_volume = (4 * np.pi / 3) * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    # number of particles with edge[i] <= r < edge[i+1] (as in mwgcs, the outermost particle is dropped)
    bin_count = np.diff(np.searchsorted(_r, bin_edges, side='left'))
    bin_radius = np.array([np.mean(b) for b in bin_r])
    return bin_radius, mp * bin_count / bin_volume

def get_log_density_profile(r, mp, r_lo, r_hi, bins=50, min_count=20):
    '''
    Density in `bins` log-spaced radial bins between r_lo and r_hi, at the bins' geometric-centre radii, keeping
    only bins with at least min_count particles. Returns (radius, density, count).
    '''
    edges = np.logspace(np.log10(r_lo), np.log10(r_hi), bins + 1)
    count = np.histogram(r, edges)[0]
    volume = (4 * np.pi / 3) * (edges[1:]**3 - edges[:-1]**3)
    keep = count >= min_count
    return np.sqrt(edges[1:] * edges[:-1])[keep], (mp * count / volume)[keep], count[keep]

def einasto_log_density(r, alpha, Rs, logScaleDensity):
    '''log10 Einasto density (Gao+2008 / Gao+2014 parametrisation); Rs is r_-2 and 10**logScaleDensity is rho(r_-2).'''
    # https://ui.adsabs.harvard.edu/abs/2014MNRAS.444..222G/abstract; A cancels, kept for parity with mwgcs
    A = 1.715 * (alpha ** (-0.00183)) * (alpha + 0.0817) ** (-0.179488)
    Rmax = A * Rs
    return logScaleDensity - (2 / alpha) * (((A * r) / Rmax) ** alpha - 1) / np.log(10)

def nfw_log_density(r, Rs, logScaleDensity):
    '''log10 NFW density, with Rs = r_-2 and 10**logScaleDensity = rho(r_-2) = rho_0 / 4, as for einasto_log_density.'''
    x = r / Rs
    return logScaleDensity + np.log10(4 / (x * (1 + x)**2))

def nfw_from_mass(m, rvir, c):
    '''(Rs, log10 rho(Rs)) of the NFW profile with M(<rvir) = m and concentration c = rvir / Rs.'''
    Rs = rvir / c
    rho_0 = m / (4 * np.pi * Rs**3 * (np.log(1 + c) - c / (1 + c)))
    return Rs, np.log10(rho_0 / 4)

# Untruncated analytic mass, potential and density of the spherical_profile fits, in Msun, kpc, km^2/s^2 and
# Msun/kpc^3. params = (alpha, Rs, log10_rho_s) as saved by spherical_profile (alpha unused for NFW).

def einasto_mass(r, alpha, Rs, logScaleDensity):
    from scipy.special import gamma, gammainc
    t = 2 * (r / Rs)**alpha / alpha
    norm = 4 * np.pi * 10**logScaleDensity * Rs**3 * np.exp(2 / alpha) * (alpha / 2)**(3 / alpha) / alpha
    return norm * gamma(3 / alpha) * gammainc(3 / alpha, t)

def einasto_potential(r, alpha, Rs, logScaleDensity):
    from scipy.special import gamma, gammaincc
    t = 2 * (r / Rs)**alpha / alpha
    outer = 4 * np.pi * agama.G * 10**logScaleDensity * Rs**2 * np.exp(2 / alpha) * (alpha / 2)**(2 / alpha) / alpha
    return -agama.G * einasto_mass(r, alpha, Rs, logScaleDensity) / r - outer * gamma(2 / alpha) * gammaincc(2 / alpha, t)

def nfw_mass(r, Rs, logScaleDensity):
    x = r / Rs
    return 16 * np.pi * 10**logScaleDensity * Rs**3 * (np.log1p(x) - x / (1 + x))

def nfw_potential(r, Rs, logScaleDensity):
    return -16 * np.pi * agama.G * 10**logScaleDensity * Rs**3 * np.log1p(r / Rs) / r

def profile_circular(r, params, profile_type):
    '''
    Circular-orbit energy E_circ(r) = Phi(r) + G M(<r) / (2 r) (km^2/s^2, Phi -> 0 at infinity) and tidal strength
    max(|4 pi G rho - 2 G M / r^3|, G M / r^3) in Gyr^-2 -- the same definitions as
    SymphonyPotential.circular_energy_radius -- of a spherical_profile fit (PROFILE_EINASTO or PROFILE_NFW).
    '''
    r = np.asarray(r, dtype=float)
    alpha, Rs, log_rho = params
    if profile_type == PROFILE_EINASTO:
        M, phi = einasto_mass(r, alpha, Rs, log_rho), einasto_potential(r, alpha, Rs, log_rho)
        rho = 10**einasto_log_density(r, alpha, Rs, log_rho)
    elif profile_type == PROFILE_NFW:
        M, phi, rho = nfw_mass(r, Rs, log_rho), nfw_potential(r, Rs, log_rho), 10**nfw_log_density(r, Rs, log_rho)
    else:
        raise ValueError(f"profile_type must be PROFILE_EINASTO or PROFILE_NFW, got {profile_type}.")
    G = agama.G
    lam_t = G * M / r**3
    lam_r = 4 * np.pi * G * rho - 2 * G * M / r**3
    return phi + G * M / (2 * r), np.maximum(np.abs(lam_r), np.abs(lam_t)) * 1.0459401725324529

def fit_einasto(r, mp, rvir, r_conv, bins=50, p0=(0.18, 10, 5), bounds=((0.175, 0.01, 1.0), (0.185, 50., 10.)),
                maxfev=1000, binning='equal_count', min_bin_count=20):
    '''
    Fits einasto_log_density to the binned density of radii r beyond r_conv. Returns ((alpha, Rs, logScaleDensity),
    number of bins used).

    binning='equal_count' (mwgcs / the notebook): `bins` equal-count bins of all particles inside rvir, keeping those
    beyond r_conv. For a well-resolved halo the innermost bin then reaches several r_conv, so nothing near r_conv
    constrains the fit. binning='log': `bins` log-spaced bins from r_conv to rvir with >= min_bin_count particles.
    '''
    from scipy.optimize import curve_fit
    if binning == 'equal_count':
        radii, rho = get_density_profile(r, mp, rvir, bins)
    elif binning == 'log':
        radii, rho, _ = get_log_density_profile(r, mp, r_conv, rvir, bins, min_bin_count)
    else:
        raise ValueError("binning must be 'equal_count' or 'log'.")
    mask = (radii > r_conv) & (rho > 0)
    popt, _ = curve_fit(einasto_log_density, radii[mask], np.log10(rho[mask]), p0=p0, bounds=bounds, maxfev=maxfev,
                        nan_policy="omit")
    return popt, np.sum(mask)

def fit_spherical_halo(r, mp, rvir, m_rockstar, cvir_rockstar, r_conv, bins=50, min_fit_particles=1000,
                       min_fit_bins=5, p0=(0.18, 10, 5), bounds=((0.05, 0.01, 1.0), (1.0, 50., 12.)), maxfev=5000,
                       rs_max_rvir=1.0, binning='log', min_bin_count=20, bound_fallback=True):
    '''
    Einasto fit (with NFW fallback) of one halo's particles at radii r (physical kpc, about the halo centre), as
    described in SymphonyPotential.spherical_profile, which calls it for every halo and snapshot. m_rockstar and
    cvir_rockstar are the halo's Rockstar mass and concentration, used for the NFW fallback.

    Returns a dict: params (alpha, Rs, log10_rho_s; alpha NaN for NFW), profile_type (PROFILE_EINASTO or
    PROFILE_NFW), n_fit_bins, n_fit_particles, hit_bounds, fallback_reason, einasto_params (every completed Einasto
    fit, including rejected ones, NaN otherwise).
    '''
    lo_h, hi_h = np.array(bounds[0], dtype=float), np.array(bounds[1], dtype=float)
    if rs_max_rvir is not None:
        hi_h[1] = rs_max_rvir * rvir

    n_out = np.sum((r > r_conv) & (r < rvir))
    out = dict(n_fit_bins=0, n_fit_particles=n_out, hit_bounds=False, fallback_reason=FALLBACK_NONE,
               einasto_params=np.full(3, np.nan))

    popt, reason = None, FALLBACK_FEW_PARTICLES
    if n_out >= min_fit_particles:
        # halo-scaled starting point, inside the bounds
        m_in = mp * np.sum(r < rvir)
        guess = np.clip([p0[0], 0.2 * rvir, np.log10(m_in / (4 * np.pi * (0.2 * rvir)**3))],
                        lo_h + 1e-6 * (hi_h - lo_h), hi_h - 1e-6 * (hi_h - lo_h))
        try:
            popt, nb = fit_einasto(r, mp, rvir, r_conv, bins=bins, p0=guess, bounds=(lo_h, hi_h), maxfev=maxfev,
                                   binning=binning, min_bin_count=min_bin_count)
            out['n_fit_bins'] = nb
            if not np.all(np.isfinite(popt)):
                popt, reason = None, FALLBACK_FIT_FAILED
            elif nb < min_fit_bins:
                popt, reason = None, FALLBACK_FEW_BINS
        except (RuntimeError, ValueError, TypeError):
            popt, reason = None, FALLBACK_FIT_FAILED

    if popt is not None:
        out['einasto_params'] = popt
        tol = 1e-3 * (hi_h - lo_h)
        at_lo, at_hi = popt - lo_h < tol, hi_h - popt < tol
        out['hit_bounds'] = np.any(at_lo | at_hi)
        # Rs = rs_max_rvir * rvir only means no density turnover inside rvir; the curve is still
        # constrained by the particles in the fit range, so it is kept as Einasto
        if rs_max_rvir is not None:
            at_hi[1] = False
        if bound_fallback and np.any(at_lo | at_hi):
            popt, reason = None, FALLBACK_AT_BOUND

    if popt is not None:
        out.update(params=np.asarray(popt, dtype=float), profile_type=PROFILE_EINASTO)
    else:
        Rs, log_rho_s = nfw_from_mass(m_rockstar, rvir, cvir_rockstar)
        out.update(params=np.array([np.nan, Rs, log_rho_s]), profile_type=PROFILE_NFW, fallback_reason=reason)
    return out

def profile_log_density(r, params, profile_type):
    '''log10 density of a fit_spherical_halo / spherical_profile fit (PROFILE_EINASTO or PROFILE_NFW).'''
    if profile_type == PROFILE_EINASTO:
        return einasto_log_density(r, *params)
    if profile_type == PROFILE_NFW:
        return nfw_log_density(r, *params[1:])
    raise ValueError(f"profile_type must be PROFILE_EINASTO or PROFILE_NFW, got {profile_type}.")

# ---------------------------------------------------------------------------------------------------------------------
# Subhalo models: how SymphonyPotential.construct_potential_subhalo represents the material bound to each subhalo.
# ---------------------------------------------------------------------------------------------------------------------

class SubhaloModel(ABC):
    '''
    How a subhalo's bound particles are turned into its own potential at one snapshot.

    construct_potential_subhalo calls build() once per subhalo and snapshot, with the subhalo's bound particles
    centred on its Rockstar position; the returned agama potential is exported, and all of a subhalo's snapshots
    are joined into an Evolving potential that follows the subhalo's trajectory. build() returns None if the
    subhalo cannot be represented at that snapshot (its particles then stay in the host).
    '''

    def __init__(self, min_particles=100):
        # subhalos with fewer bound particles are left in the host potential
        self.min_particles = min_particles

    @abstractmethod
    def build(self, x, mp, props):
        '''
        x: (N, 3) bound-particle positions about the subhalo centre (physical kpc); mp: particle mass (Msun);
        props: dict with the subhalo's Rockstar rvir, m, cvir at this snapshot, the convergence radius r_conv, and
        halo / snapshot indices. Returns an agama.Potential or None.
        '''

    @abstractmethod
    def describe(self):
        '''dict of the model's settings, written next to the potential.'''


class SphericalSubhaloModel(SubhaloModel):
    '''
    Spherical fit (fit_spherical_halo: Einasto, falling back to the Rockstar NFW) of each subhalo's bound particles,
    with the same defaults as SymphonyPotential.spherical_profile. The fit is evaluated as an l = 0 agama Multipole
    whose density is smoothly truncated, rho(r) * exp(-(r / r_t)**cutoff_strength), since neither profile has a
    finite mass.

    truncation='bound_mass' (default) picks r_t so that the truncated profile's total mass equals the bound
    particles' mass (searched in [truncation_range[0], truncation_range[1]] * rvir, clipped to that range if no r_t
    there matches), so the material taken out of the host is all put back in the subhalo and the fitted inner profile
    is kept. truncation='rvir' uses r_t = truncation_rvir * rvir instead (the total mass then differs from the bound
    mass). fit_kwargs are passed to fit_spherical_halo.
    '''

    def __init__(self, min_particles=100, truncation='bound_mass', truncation_rvir=1.0, truncation_range=(0.05, 100.),
                 cutoff_strength=2.0, rmin=1e-3, rmax_trunc=5.0, nrad=25, **fit_kwargs):
        super().__init__(min_particles=min_particles)
        if truncation not in ('bound_mass', 'rvir'):
            raise ValueError("truncation must be 'bound_mass' or 'rvir'.")
        self.truncation = truncation
        self.truncation_rvir = truncation_rvir
        self.truncation_range = tuple(truncation_range)
        self.cutoff_strength = cutoff_strength
        self.rmin = rmin
        self.rmax_trunc = rmax_trunc
        self.nrad = nrad
        self.fit_kwargs = fit_kwargs

    def _truncated_mass(self, params, ptype, r_t, rvir):
        # 4 pi int rho(r) exp(-(r/r_t)^cs) r^2 dr on a log grid well past r_t
        r = np.logspace(np.log10(self.rmin * 1e-2), np.log10(max(r_t, rvir) * 30), 2000)
        rho = 10**profile_log_density(r, params, ptype) * np.exp(-(r / r_t)**self.cutoff_strength)
        trapezoid = getattr(np, "trapezoid", None) or np.trapz
        return trapezoid(4 * np.pi * rho * r**3, np.log(r))

    def truncation_radius(self, params, ptype, m_bound, rvir):
        '''r_t for these fit params (see the class docstring).'''
        if self.truncation == 'rvir':
            return self.truncation_rvir * rvir
        from scipy.optimize import brentq
        lo, hi = self.truncation_range[0] * rvir, self.truncation_range[1] * rvir
        f = lambda log_rt: np.log(self._truncated_mass(params, ptype, np.exp(log_rt), rvir) / m_bound)
        f_lo, f_hi = f(np.log(lo)), f(np.log(hi))
        if f_lo >= 0:
            return lo
        if f_hi <= 0:
            return hi
        return np.exp(brentq(f, np.log(lo), np.log(hi), xtol=1e-4))

    def build(self, x, mp, props):
        r = np.linalg.norm(x, axis=1)
        fit = fit_spherical_halo(r, mp, props['rvir'], props['m'], props['cvir'], props['r_conv'], **self.fit_kwargs)
        params, ptype = fit['params'], fit['profile_type']
        if not np.all(np.isfinite(params[1:])) or params[1] <= 0:
            return None
        r_t = self.truncation_radius(params, ptype, mp * len(x), props['rvir'])
        cs = self.cutoff_strength

        def density(xyz):
            rr = np.maximum(np.linalg.norm(np.atleast_2d(xyz), axis=1), 1e-12)
            return 10**profile_log_density(rr, params, ptype) * np.exp(-(rr / r_t)**cs)

        return agama.Potential(type='Multipole', density=density, lmax=0, symmetry='spherical', rmin=self.rmin,
                               rmax=self.rmax_trunc * r_t, gridSizeR=self.nrad)

    def describe(self):
        return dict(type='spherical', min_particles=self.min_particles, truncation=self.truncation,
                    truncation_rvir=self.truncation_rvir, truncation_range=self.truncation_range,
                    cutoff_strength=self.cutoff_strength, rmin=self.rmin, rmax_trunc=self.rmax_trunc, nrad=self.nrad,
                    **self.fit_kwargs)


class MultipoleSubhaloModel(SubhaloModel):
    '''
    Multipole expansion of order lmax of each subhalo's bound particles (as the host in construct_potential), on a
    radial grid from rmin to max(rmax_rvir * rvir, the outermost bound particle). symmetry defaults to 'spherical'
    for lmax = 0 and 'None' otherwise.
    '''

    def __init__(self, lmax=2, min_particles=100, rmin=1e-3, rmax_rvir=1.2, nrad=25, symmetry=None):
        super().__init__(min_particles=min_particles)
        self.lmax = lmax
        self.rmin = rmin
        self.rmax_rvir = rmax_rvir
        self.nrad = nrad
        self.symmetry = symmetry if symmetry is not None else ('spherical' if lmax == 0 else 'None')

    def build(self, x, mp, props):
        rmax = max(self.rmax_rvir * props['rvir'], np.max(np.linalg.norm(x, axis=1)))
        return agama.Potential(type='Multipole', symmetry=self.symmetry, lmax=self.lmax, rmin=self.rmin, rmax=rmax,
                               gridSizeR=self.nrad, particles=(x, mp * np.ones(len(x))))

    def describe(self):
        return dict(type='multipole', lmax=self.lmax, min_particles=self.min_particles, rmin=self.rmin,
                    rmax_rvir=self.rmax_rvir, nrad=self.nrad, symmetry=self.symmetry)


SUBHALO_MODELS = {'spherical': SphericalSubhaloModel, 'multipole': MultipoleSubhaloModel}

def subhalo_model_from_spec(spec):
    '''
    SubhaloModel from a config dict such as {'type': 'multipole', 'lmax': 2}: 'type' picks the class
    (SUBHALO_MODELS), the remaining keys are its constructor arguments. None returns None (no subhalos).
    '''
    if spec is None:
        return None
    spec = dict(spec)
    kind = spec.pop('type', None)
    if kind not in SUBHALO_MODELS:
        raise ValueError(f"subhalo model type must be one of {sorted(SUBHALO_MODELS)}, got {kind!r}.")
    return SUBHALO_MODELS[kind](**spec)

class Potential(ABC):
    def __init__(self, interface, write_dir):
        self.interface = interface
        self.phi = None
        self.write_dir = write_dir
        self.potential_exists = self.check_potential_exists()

        if self.potential_exists:
            print("Potential file already exists. Reading potential from file...")
            self.read_potential(os.path.join(self.write_dir, 'potential.ini'))

    @abstractmethod
    def construct_potential(self):
        pass

    def check_potential_exists(self):
        return os.path.exists(os.path.join(self.write_dir, 'potential.ini'))

    def read_potential(self, directory):
        self.phi = agama.Potential(file=directory)

    def tidal_tensor(self):
        if self.phi is None:
            raise ValueError("Potential not constructed or read from file.")
        pass

    def tidal_strength(self):
        if self.phi is None:
            raise ValueError("Potential not constructed or read from file.")
        pass

    def circular_energy_radius(self, *args, **kwargs):
        # per-snapshot energy of a circular orbit vs radius, E_c(r) = U(r) + K_circ(r); not available by default
        return None

class AgamaPotential(Potential):
    def tidal_tensor(self, x, t=0.0):
        if self.phi is None:
            raise ValueError("Potential not constructed or read from file.")
        
        x = np.atleast_2d(x)
        N = len(x)

        if np.isscalar(t):
            t = np.full(N, t)
        elif len(t) != N:
            raise ValueError("Length of time array must match number of positions.")

        # one vectorised call (agama accepts a per-point t), identical to evaluating point by point
        d = np.atleast_2d(self.phi.eval(x, der=True, t=np.asarray(t, dtype=float)))

        d2phidx2  = d[:, 0]
        d2phidy2  = d[:, 1]
        d2phidz2  = d[:, 2]
        d2phidxdy = d[:, 3]
        d2phidydz = d[:, 4]
        d2phidxdz = d[:, 5]
        tidal_tensors = np.array([
            [d2phidx2,  d2phidxdy, d2phidxdz],
            [d2phidxdy, d2phidy2,  d2phidydz],
            [d2phidxdz, d2phidydz, d2phidz2]
        ])

        tidal_tensors = np.moveaxis(tidal_tensors, -1, 0)
        
        return tidal_tensors[0] if tidal_tensors.shape[0] == 1 else tidal_tensors
    
    def tidal_strength(self, x, t=0.0):
        tidal_tensor = self.tidal_tensor(x, t)
        if tidal_tensor.ndim == 3:
            if (tidal_tensor.shape[0] == 3) and (tidal_tensor.shape[1] == 3):
                tidal_tensor = np.moveaxis(tidal_tensor, -1, 0)
            eigenvalues = np.linalg.eigvals(tidal_tensor)
            
            if eigenvalues.shape[0] == 1:
                return np.max(np.abs(eigenvalues))
            else:
                return np.max(np.abs(eigenvalues), axis=1)
        else:
            eigenvalues = np.linalg.eigvals(tidal_tensor)
            return np.max(np.abs(eigenvalues))

class SymphonyPotential(AgamaPotential):
    def check_potential_exists(self):
        potential_ini = os.path.join(self.write_dir, 'potential.ini')

        if not os.path.exists(potential_ini):
            return False

        missing = [
            f for f in self._snapshot_files_in_potential(potential_ini)
            if not os.path.exists(os.path.join(self.write_dir, f))
        ]

        if missing:
            print(
                f"potential.ini exists but is missing {len(missing)} snapshot file(s) "
                f"it references (e.g. {missing[:5]}) -- treating the potential as incomplete."
            )
            return False

        return True

    def _snapshot_files_in_potential(self, potential_ini):
        # every file listed under a Timestamps block, in all [Potential ...] sections (construct_potential writes
        # one section, construct_potential_subhalo one per halo)
        files = []
        in_timestamps = False

        with open(potential_ini) as f:
            for line in f:
                line = line.strip()
                if line == 'Timestamps':
                    in_timestamps = True
                    continue
                if not line or line.startswith('['):
                    in_timestamps = False
                    continue
                if in_timestamps:
                    files.append(line.split()[1])

        return files

    def _existing_snapshot_numbers(self):
        return sorted(
            int(f.split('_')[1].split('.')[0])
            for f in os.listdir(self.write_dir) if f.startswith('snap_') and f.endswith('.ini')
        )

    def _snapshots_to_build(self, snapshots, overwrite, sub_dir=None):
        '''
        Snapshots (from `snapshots`) that still need a host file. On a resume, the newest existing host file (and its
        subhalo files) is removed and rebuilt in case an older, non-atomic run left it truncated; every snapshot
        without a host file is (re)built, so snapshots finished out of order by parallel workers are kept.
        '''
        if overwrite:
            return list(snapshots)
        existing = self._existing_snapshot_numbers()
        if existing:
            newest = existing[-1]
            print(f"Found an incomplete potential -- rebuilding snapshot {newest} (in case a previous failed run "
                  f"left it corrupted) and every snapshot without a file.")
            os.remove(os.path.join(self.write_dir, f'snap_{newest}.ini'))
            if sub_dir is not None:
                for k, f in self._subhalo_snapshot_files(sub_dir):
                    if k == newest:
                        os.remove(f)
            existing = set(existing[:-1])
        return [s for s in snapshots if s not in existing]

    @staticmethod
    def _export_atomic(pot, path):
        # write next to the target and rename, so an interrupted run never leaves a truncated .ini behind
        tmp = f"{path}.tmp{os.getpid()}"
        pot.export(tmp)
        os.replace(tmp, path)

    def construct_potential(self, lmax, rmin=1e-2, rmax_rvir=1.2, nrad=25, mode='all', min_particles=100, symmetry='None', stride=1, overwrite=False,
                            spherical_profile=False, spherical_profile_kwargs=None, n_workers=1):
        '''
        Builds the host's evolving multipole potential in write_dir. If spherical_profile is True, also fits the
        Einasto / NFW spherical approximation of every halo at every snapshot (SymphonyPotential.spherical_profile,
        called with spherical_profile_kwargs) into write_dir/spherical_profile.npz -- this runs even when the
        multipole potential already exists and is not rebuilt. An existing spherical_profile.npz is only recomputed
        if spherical_profile_kwargs contains overwrite=True.

        n_workers: build the snapshots in parallel on this many forked worker processes (each reads one snapshot's
        particles at a time, so memory grows with n_workers). Snapshots are independent; 1 = serial.
        '''
        os.makedirs(self.write_dir, exist_ok=True)

        if spherical_profile:
            self.spherical_profile(**(spherical_profile_kwargs or {}))

        if self.potential_exists and not overwrite:
            print("Potential already exists. Use `overwrite=True` to reconstruct potential.")
            return

        todo = self._snapshots_to_build(range(len(self.interface.scale_factors)), overwrite)

        def build(snapshot):
            p_snap = self.interface.particles.read(snapshot, mode=mode, halo=0)
            x = p_snap['x'][p_snap['ok']]

            if len(x) < min_particles:
                # skip this snapshot if there are too few particles to construct a reliable potential
                return

            m = self.interface.mp * np.ones(len(x))

            potential = agama.Potential(
                type='Multipole',
                symmetry=symmetry,
                lmax=lmax,
                rmin=rmin,
                rmax=rmax_rvir * self.interface.rs["rvir"][0, snapshot],
                gridSizeR=nrad,
                particles=(x, m)
            )

            self._export_atomic(potential, os.path.join(self.write_dir, f'snap_{snapshot}.ini'))

        from gchords.parallel import run_chunked
        run_chunked(lambda chunk: [build(s) for s in chunk], todo, n_workers=n_workers, chunk_size=1,
                    desc=f"Constructing potentials ({n_workers} workers)...")

        existing_snapshots = set(self._existing_snapshot_numbers())
        start_snapshot = min(existing_snapshots)
        end_snapshot = len(self.interface.scale_factors) - 1

        age_int = self._agama_times()

        with open(os.path.join(self.write_dir, 'potential.ini'), 'w') as f:
            f.write("[Potential]\n")
            f.write("type=Evolving\n")
            f.write("interpLinear=True\n")
            f.write("Timestamps\n")
            for k in range(start_snapshot, end_snapshot + 1, stride):
                if k not in existing_snapshots:
                    # no potential was constructed for this snapshot (e.g. too few particles)
                    continue
                f.write(f"{age_int[k]} snap_{k}.ini\n")
            f.write("\n")

        self.phi = agama.Potential(file=os.path.join(self.write_dir, 'potential.ini'))
        self.potential_exists = True

    def _agama_times(self):
        return agama_times(self.interface.cosmology, self.interface.scale_factors)

    def circular_energy_radius(self, n_points=50, r_min=1e-2, r_max=2e3, spacing='log', snapshots=None, mode='all',
                               min_particles=100, k_density=32, file_name='circular_energy_radius.npz', overwrite=False):
        '''
        Tabulates the energy of a circular orbit vs radius, E_c(r) = U(r) + K_circ(r), and the tidal strength for the
        host at each snapshot, on a subsample of bound host particles (no binning), and writes it to write_dir/file_name.

        At each snapshot, n_points target radii are placed evenly (in log r if spacing='log', else linearly) on
        [r_min, r_max], and the bound valid host particle nearest to each target (in log r or r) is selected; a
        particle picked for several targets is kept once, and the remaining columns are NaN.

        U is the per-particle potential from symlib.profile_info, after undoing its PE/vmax**2 scaling and its
        4 G M / R_out outer boundary term (should be G M / R_out). The tidal strength and K_circ are from the
        unsoftened spherical potential of the enclosed mass M(<r) (profile_info's own softening, which shifts r to
        sqrt(r**2 + eps**2) with eps in comoving kpc/h, is ignored), whose Hessian has eigenvalues
            lambda_t = (1/r) dU/dr = G M / r**3                  (tangential, twice)
            lambda_r = d2U/dr2    = 4 pi G rho - 2 G M / r**3    (radial)
        with M(<r) from all valid host particles and 4 pi r**2 rho = dM/dr estimated from the k_density valid
        particles on either side in radius. The tidal strength is max(|lambda_r|, |lambda_t|) in Gyr^-2, the same
        definition and units as AgamaPotential.tidal_strength * 1.0459401725324529 in compute_cluster_tidal_field.
        K_circ = v_c**2 / 2 = G M / (2 r). Snapshots with fewer than min_particles valid particles are NaN.
        Energies are in km^2/s^2, radii in physical kpc.

        Returns a dict of the saved arrays, each (n_snapshots, n_points) unless noted:
            snapshots, scale_factor, time (n_snapshots; time in Agama units, as in potential.ini),
            r_target (n_points), r, particle_index (index into the snapshot's host particles, -1 if empty),
            U, K, E (the particle's own energy, K + U), K_circ, E_circ, M_enc, lambda_r, lambda_t, tidal_strength
        '''
        path = os.path.join(self.write_dir, file_name)
        if os.path.exists(path) and not overwrite:
            print(f"{path} already exists. Use `overwrite=True` to recompute it.")
            return dict(np.load(path))
        os.makedirs(self.write_dir, exist_ok=True)

        if spacing == 'log':
            r_target = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
        elif spacing == 'linear':
            r_target = np.linspace(r_min, r_max, n_points)
        else:
            raise ValueError("spacing must be 'log' or 'linear'.")
        if snapshots is None:
            snapshots = np.arange(len(self.interface.scale_factors))
        snapshots = np.asarray(snapshots, dtype=int)

        params, mp, G = self.interface.params, self.interface.mp, agama.G
        keys = ('r', 'U', 'K', 'E', 'K_circ', 'E_circ', 'M_enc', 'lambda_r', 'lambda_t', 'tidal_strength')
        out = {k: np.full((len(snapshots), n_points), np.nan) for k in keys}
        out['particle_index'] = np.full((len(snapshots), n_points), -1, dtype=int)

        for i, snapshot in enumerate(tqdm(snapshots, desc="Tabulating circular energies...")):
            p_snap = self.interface.particles.read(snapshot, mode=mode, halo=0)
            ok = p_snap['ok']
            if np.sum(ok) < min_particles:
                continue

            _, vmax, pe, _ = symlib.profile_info(params, p_snap['x'], ok)
            r = np.linalg.norm(p_snap['x'], axis=1)
            # profile_info places its outer boundary at the largest radius of any particle, valid or not
            r_out = np.sqrt(np.nanmax(r)**2 + params['eps']**2)
            U = pe * vmax**2 + 3 * G * np.sum(ok) * mp / r_out
            K = np.sum(p_snap['v']**2, axis=1) / 2
            E = K + U

            # nearest bound particle to each target radius, each particle used at most once
            bound_idx = np.where(ok & (E < 0) & (r > 0))[0]
            if len(bound_idx) == 0:
                continue
            bound_idx = bound_idx[np.argsort(r[bound_idx])]
            r_b = r[bound_idx]
            coord = np.log10 if spacing == 'log' else (lambda a: a)
            j = np.clip(np.searchsorted(coord(r_b), coord(r_target)), 1, len(r_b) - 1)
            j = np.where(np.abs(coord(r_b[j - 1]) - coord(r_target)) <= np.abs(coord(r_b[j]) - coord(r_target)),
                         j - 1, j) if len(r_b) > 1 else np.zeros(n_points, dtype=int)
            sel = bound_idx[np.unique(j)]
            n_sel = len(sel)

            # enclosed mass and local density from the valid particles, as seen by profile_info
            r_valid = np.sort(r[ok])
            r_s = r[sel]
            n_enc = np.searchsorted(r_valid, r_s)
            M = n_enc * mp
            lo = np.clip(n_enc - k_density, 0, len(r_valid) - 1)
            hi = np.clip(n_enc + k_density, 0, len(r_valid) - 1)
            dMdr = (hi - lo) * mp / (r_valid[hi] - r_valid[lo])
            rho = dMdr / (4 * np.pi * r_s**2)

            lam_t = G * M / r_s**3
            lam_r = 4 * np.pi * G * rho - 2 * G * M / r_s**3
            K_circ = G * M / (2 * r_s)

            row = slice(0, n_sel)
            out['particle_index'][i, row] = sel
            out['r'][i, row], out['M_enc'][i, row] = r_s, M
            out['U'][i, row], out['K'][i, row], out['E'][i, row] = U[sel], K[sel], E[sel]
            out['K_circ'][i, row], out['E_circ'][i, row] = K_circ, U[sel] + K_circ
            out['lambda_r'][i, row], out['lambda_t'][i, row] = lam_r, lam_t
            # (km/s/kpc)^2 -> Gyr^-2, as in GChords.compute_cluster_tidal_field
            out['tidal_strength'][i, row] = np.maximum(np.abs(lam_r), np.abs(lam_t)) * 1.0459401725324529

        out.update(snapshots=snapshots, scale_factor=self.interface.scale_factors[snapshots],
                   time=self._agama_times()[snapshots], r_target=r_target)
        np.savez_compressed(path, **out)
        return out

    def spherical_profile(self, halos=None, snapshots=None, mode='all', bins=50, min_fit_particles=1000,
                          min_fit_bins=5, p0=(0.18, 10, 5), bounds=((0.05, 0.01, 1.0), (1.0, 50., 12.)),
                          maxfev=5000, rs_max_rvir=1.0, binning='log', min_bin_count=20, bound_fallback=True,
                          file_name='spherical_profile.npz', overwrite=False):
        '''
        Fits a spherical density profile to every halo (host = 0 and subhalos) at every snapshot and writes it to
        write_dir/file_name.

        For each tracked halo (rs['ok']), the valid particles it owns are centred on its Rockstar position, and an
        Einasto profile is fit to the log density of its particles between the convergence radius r_conv
        (get_convergence_radius; the same for all halos at a snapshot) and its Rockstar rvir, in `bins` log-spaced
        radial bins with at least min_bin_count particles each (binning='log'). The fit is then used inside r_conv,
        where the particles cannot be trusted, so it must be constrained near r_conv.

        This differs from the fit in energy_sorting/test_inner_extrapolation.ipynb (ported from mwgcs), which is still
        available with binning='equal_count', bounds=((0.175, 0.01, 1.0), (0.185, 50., 10.)), rs_max_rvir=None:
          - mwgcs bins all particles inside rvir into equal-count bins and keeps those beyond r_conv. For a host with
            ~10^6 particles the innermost bin then reaches ~5-8 kpc (several r_conv), so the fit is an extrapolation
            near r_conv; its tidal strength just outside r_conv was ~2x too low at late times for MWest Halo004.
          - alpha is free in [0.05, 1] instead of [0.175, 0.185].
          - the upper bound on Rs is rs_max_rvir * rvir of that halo at that snapshot instead of bounds[1][1] (a fixed
            50 kpc pinned the late-time MWest hosts at the bound).

        If the Einasto fit fails -- fewer than min_fit_particles valid particles between r_conv and rvir, fewer than
        min_fit_bins bins beyond r_conv, curve_fit fails, or (if bound_fallback) the fit ends on a bound of alpha,
        log10_rho_s or the lower bound of Rs -- the halo is instead given the NFW profile with its Rockstar mass m inside rvir and
        concentration cvir (from vmax / vvir, so the NFW reproduces Rockstar's vmax). A fit on a bound is not
        constrained by the particles (typically small, stripped subhalos whose density is steeper than r^-2 at every
        resolved radius, so r_-2 and alpha are set by unresolved scales), and the simulation's resolution does not
        represent the true inner structure there either, so the NFW is used rather than the numerics. For MWest
        Halo004 such NFWs matched the particle density in the fit range to ~0.11 dex rms vs ~0.08 dex for the bounded
        Einasto fits. A fit with Rs on its rs_max_rvir * rvir upper bound is kept: it only means there is no density
        turnover inside rvir, and the curve is still constrained in the fit range (e.g. the MWest Halo004 host at
        snapshot 170 matched the particle tidal strength near r_conv far better than its NFW did); it is flagged in
        hit_bounds. Untracked halos (not yet formed, or disrupted) are NaN.

        params[..., :] = (alpha, Rs, log10_rho_s). For both profiles Rs is r_-2 (the radius where dlog rho / dlog r
        = -2) in physical kpc, and rho_s = rho(r_-2) in Msun / kpc^3 (for NFW this is rho_0 / 4); alpha is NaN for
        NFW. Evaluate with einasto_log_density(r, *params) or nfw_log_density(r, *params[1:]) according to
        profile_type.

        Returns a dict of the saved arrays:
            params (n_halos, n_snapshots, 3), profile_type (n_halos, n_snapshots; PROFILE_UNTRACKED = -1,
            PROFILE_EINASTO = 0, PROFILE_NFW = 1), n_fit_bins and n_fit_particles (n_halos, n_snapshots; bins /
            particles beyond r_conv used or available for the Einasto fit), hit_bounds (the Einasto fit ended on a
            bound of alpha, Rs or log10_rho_s; with bound_fallback the NFW was used instead unless only the
            rvir-tied Rs bound was hit), fallback_reason
            (FALLBACK_NONE = 0, FEW_PARTICLES = 1, FEW_BINS = 2, FIT_FAILED = 3, AT_BOUND = 4; 0 for Einasto and
            untracked), einasto_params (n_halos, n_snapshots, 3; every completed Einasto fit, including rejected
            ones, NaN otherwise), halos (n_halos), snapshots, scale_factor, time (n_snapshots; Agama units),
            r_conv (n_snapshots), m, rvir, cvir (n_halos, n_snapshots; Rockstar), param_names.
        '''
        path = os.path.join(self.write_dir, file_name)
        if os.path.exists(path) and not overwrite:
            print(f"{path} already exists. Use `overwrite=True` to recompute it.")
            return dict(np.load(path))
        os.makedirs(self.write_dir, exist_ok=True)

        rs, params, mp = self.interface.rs, self.interface.params, self.interface.mp
        halos = np.arange(rs.shape[0]) if halos is None else np.asarray(halos, dtype=int)
        if snapshots is None:
            snapshots = np.arange(len(self.interface.scale_factors))
        snapshots = np.asarray(snapshots, dtype=int)
        shape = (len(halos), len(snapshots))

        out_params = np.full(shape + (3,), np.nan)
        profile_type = np.full(shape, PROFILE_UNTRACKED, dtype=np.int8)
        n_fit_bins = np.zeros(shape, dtype=int)
        n_fit_particles = np.zeros(shape, dtype=int)
        hit_bounds = np.zeros(shape, dtype=bool)
        fallback_reason = np.zeros(shape, dtype=np.int8)
        einasto_params = np.full(shape + (3,), np.nan)
        r_conv_snap = np.array([get_convergence_radius(params, self.interface.scale_factors[s]) for s in snapshots])

        for j, snapshot in enumerate(tqdm(snapshots, desc="Fitting spherical profiles...")):
            tracked = rs['ok'][halos, snapshot]
            if not np.any(tracked):
                continue
            # 'all' / 'current' modes read every halo at once anyway
            p_all = self.interface.particles.read(snapshot, mode=mode)
            r_conv = r_conv_snap[j]

            for i in np.flatnonzero(tracked):
                h = halos[i]
                rvir = rs['rvir'][h, snapshot]
                p = p_all[h]
                r = np.linalg.norm(p['x'][p['ok']] - rs['x'][h, snapshot], axis=1)
                fit = fit_spherical_halo(
                    r, mp, rvir, rs['m'][h, snapshot], rs['cvir'][h, snapshot], r_conv, bins=bins,
                    min_fit_particles=min_fit_particles, min_fit_bins=min_fit_bins, p0=p0, bounds=bounds,
                    maxfev=maxfev, rs_max_rvir=rs_max_rvir, binning=binning, min_bin_count=min_bin_count,
                    bound_fallback=bound_fallback)
                out_params[i, j] = fit['params']
                profile_type[i, j] = fit['profile_type']
                n_fit_bins[i, j] = fit['n_fit_bins']
                n_fit_particles[i, j] = fit['n_fit_particles']
                hit_bounds[i, j] = fit['hit_bounds']
                fallback_reason[i, j] = fit['fallback_reason']
                einasto_params[i, j] = fit['einasto_params']

        m = np.where(profile_type != PROFILE_UNTRACKED, rs['m'][halos][:, snapshots], np.nan)
        rvir = np.where(profile_type != PROFILE_UNTRACKED, rs['rvir'][halos][:, snapshots], np.nan)
        cvir = np.where(profile_type != PROFILE_UNTRACKED, rs['cvir'][halos][:, snapshots], np.nan)
        out = dict(params=out_params, profile_type=profile_type, n_fit_bins=n_fit_bins,
                   n_fit_particles=n_fit_particles, hit_bounds=hit_bounds, fallback_reason=fallback_reason,
                   einasto_params=einasto_params, bound_fallback=bound_fallback, halos=halos, snapshots=snapshots,
                   scale_factor=self.interface.scale_factors[snapshots], time=self._agama_times()[snapshots],
                   r_conv=r_conv_snap, m=m, rvir=rvir, cvir=cvir,
                   param_names=np.array(['alpha', 'Rs', 'log10_rho_s']),
                   rs_max_rvir=np.nan if rs_max_rvir is None else rs_max_rvir, binning=binning,
                   bounds=np.asarray(bounds, dtype=float))
        np.savez_compressed(path, **out)
        return out

    def construct_potential_subhalo(self, subhalo_model, lmax, rmin=1e-2, rmax_rvir=1.2, nrad=25, mode='all',
                                    min_particles=100, symmetry='None', snapshots=None, overwrite=False, n_workers=1):
        '''
        Like construct_potential, but the material bound to each subhalo gets its own potential instead of being
        part of the host's multipole.

        At each snapshot, every subhalo that has fallen into the host (snapshot >= first_infall_snap) and is still
        tracked by Rockstar has its valid particles cut to those bound to it (is_bound, about its Rockstar position
        and velocity). A particle bound to several (nested) subhalos is given to the one with the fewest bound
        particles. Subhalos with at least subhalo_model.min_particles bound particles are represented by
        subhalo_model.build() (e.g. SphericalSubhaloModel, MultipoleSubhaloModel), centred on the subhalo, and written
        to write_dir/subhalos/halo_<i>/snap_<k>.ini. The host is the same multipole as in construct_potential (lmax,
        rmin, rmax_rvir, nrad, symmetry, min_particles) of the host's valid particles minus those subhalo
        potentials' particles -- i.e. its smoothly accreted particles plus everything stripped from subhalos -- and is
        written to write_dir/snap_<k>.ini.

        potential.ini then joins them in the layout of the BFE full_cosmo.dat potentials: an Evolving [Potential
        halo_0] for the host and an Evolving [Potential halo_<i>] per subhalo, whose center follows
        subhalos/traj_<i>.txt (t, x, y, z, vx, vy, vz: the Rockstar host-centric position and velocity, linearly
        extrapolated past disruption). Each subhalo's timestamps are bracketed by null.ini (zero mass) at the
        snapshots before its first and after its last potential, so it ramps on and off over one snapshot; snapshots
        inside that range with no potential (too few bound particles) are interpolated across. The model settings
        are written to subhalo_model.json.

        Construction resumes after a failure like construct_potential: every file is written atomically, the host
        file of a snapshot is written after its subhalo files, and every snapshot without a host file is rebuilt (its
        stale subhalo files removed first). overwrite=True deletes this directory's snap_*.ini and subhalos/ first.
        snapshots restricts construction to those snapshots (default: all; mainly for testing). n_workers builds
        snapshots in parallel on forked worker processes, as in construct_potential (each worker holds one
        snapshot's particles for every halo, ~1-2 GB for an MWest host).
        '''
        import json
        import shutil

        if mode not in ('all', 'current'):
            raise ValueError("construct_potential_subhalo needs mode='all' or 'current' (every halo is read).")
        os.makedirs(self.write_dir, exist_ok=True)

        if self.potential_exists and not overwrite:
            print("Potential already exists. Use `overwrite=True` to reconstruct potential.")
            return

        sub_dir = os.path.join(self.write_dir, 'subhalos')
        n_snap = len(self.interface.scale_factors)
        snapshots = np.arange(n_snap) if snapshots is None else np.unique(np.asarray(snapshots, dtype=int))

        if overwrite:
            for k in self._existing_snapshot_numbers():
                os.remove(os.path.join(self.write_dir, f'snap_{k}.ini'))
            shutil.rmtree(sub_dir, ignore_errors=True)
        todo = self._snapshots_to_build([int(s) for s in snapshots], overwrite, sub_dir=sub_dir)
        stale = {}
        for k, f in self._subhalo_snapshot_files(sub_dir):
            stale.setdefault(k, []).append(f)

        rs, params, mp = self.interface.rs, self.interface.params, self.interface.mp
        first_infall = np.asarray(self.interface.hist['first_infall_snap'])
        n_halo = rs.shape[0]
        n_min = subhalo_model.min_particles

        def build(snapshot):
            # subhalo files left by an interrupted build of this snapshot
            for f in stale.get(snapshot, []):
                if os.path.exists(f):
                    os.remove(f)
            p_all = self.interface.particles.read(snapshot, mode=mode)
            host = p_all[0]
            if np.sum(host['ok']) < min_particles:
                return
            r_conv = get_convergence_radius(params, self.interface.scale_factors[snapshot])

            # bound particles of every subhalo inside the host
            ids, xs, owner = [], [], []
            for i in range(1, n_halo):
                if not (rs['ok'][i, snapshot] and 0 <= first_infall[i] <= snapshot):
                    continue
                p = p_all[i]
                ok = p['ok']
                if np.sum(ok) < n_min:
                    continue
                b = is_bound(p['x'][ok], p['v'][ok], rs['x'][i, snapshot], rs['v'][i, snapshot], params)
                if np.sum(b) < n_min:
                    continue
                ids.append(p['id'][ok][b])
                xs.append(p['x'][ok][b])
                owner.append(np.full(np.sum(b), i))

            claimed = []
            if ids:
                ids, xs, owner = np.concatenate(ids), np.concatenate(xs), np.concatenate(owner)
                # a particle bound to several subhalos goes to the one with the fewest bound particles
                n_bound = np.bincount(owner, minlength=n_halo)[owner]
                order = np.lexsort((owner, n_bound, ids))
                ids, xs, owner = ids[order], xs[order], owner[order]
                first = np.ones(len(ids), dtype=bool)
                first[1:] = ids[1:] != ids[:-1]
                ids, xs, owner = ids[first], xs[first], owner[first]

                for i in np.unique(owner):
                    sel = owner == i
                    if np.sum(sel) < n_min:
                        continue
                    props = dict(rvir=rs['rvir'][i, snapshot], m=rs['m'][i, snapshot], cvir=rs['cvir'][i, snapshot],
                                 r_conv=r_conv, halo=int(i), snapshot=int(snapshot))
                    pot = subhalo_model.build(xs[sel] - rs['x'][i, snapshot], mp, props)
                    if pot is None:
                        continue
                    d = os.path.join(sub_dir, f'halo_{i}')
                    os.makedirs(d, exist_ok=True)
                    self._export_atomic(pot, os.path.join(d, f'snap_{snapshot}.ini'))
                    claimed.append(ids[sel])

            # host: valid host particles not represented by a subhalo potential
            in_host = host['ok']
            if claimed:
                in_host = in_host & ~np.isin(host['id'], np.concatenate(claimed))
            x = host['x'][in_host]
            if len(x) < min_particles:
                return

            potential = agama.Potential(
                type='Multipole',
                symmetry=symmetry,
                lmax=lmax,
                rmin=rmin,
                rmax=rmax_rvir * self.interface.rs["rvir"][0, snapshot],
                gridSizeR=nrad,
                particles=(x, mp * np.ones(len(x)))
            )
            # written last, so its presence marks the snapshot as complete for resuming
            self._export_atomic(potential, os.path.join(self.write_dir, f'snap_{snapshot}.ini'))

        from gchords.parallel import run_chunked
        run_chunked(lambda chunk: [build(s) for s in chunk], todo, n_workers=n_workers, chunk_size=1,
                    desc=f"Constructing host + subhalo potentials ({n_workers} workers)...")

        self._write_subhalo_potential_ini(sub_dir)
        with open(os.path.join(self.write_dir, 'subhalo_model.json'), 'w') as f:
            json.dump(dict(subhalo_model=subhalo_model.describe(),
                           host=dict(lmax=lmax, rmin=rmin, rmax_rvir=rmax_rvir, nrad=nrad, mode=mode,
                                     min_particles=min_particles, symmetry=symmetry)), f, indent=2, default=str)

        self.phi = agama.Potential(file=os.path.join(self.write_dir, 'potential.ini'))
        self.potential_exists = True

    @staticmethod
    def _subhalo_snapshot_files(sub_dir):
        '''(snapshot, path) of every subhalos/halo_<i>/snap_<k>.ini.'''
        out = []
        if not os.path.isdir(sub_dir):
            return out
        for d in os.listdir(sub_dir):
            if not d.startswith('halo_'):
                continue
            for f in os.listdir(os.path.join(sub_dir, d)):
                if f.startswith('snap_') and f.endswith('.ini'):
                    out.append((int(f[5:-4]), os.path.join(sub_dir, d, f)))
        return out

    def _write_subhalo_potential_ini(self, sub_dir):
        '''Writes null.ini, subhalos/traj_<i>.txt and the composite potential.ini of construct_potential_subhalo.'''
        rs = self.interface.rs
        t = self._agama_times()
        last = len(self.interface.scale_factors) - 1

        with open(os.path.join(self.write_dir, 'null.ini'), 'w') as f:
            f.write("[Potential]\ntype=Plummer\nmass=0.\nscaleRadius=1.\n")

        lines = ["[Potential halo_0]", "type=Evolving", "interpLinear=True", "Timestamps"]
        lines += [f"{t[k]} snap_{k}.ini" for k in self._existing_snapshot_numbers()]
        lines.append("")

        by_halo = {}
        for k, path in self._subhalo_snapshot_files(sub_dir):
            by_halo.setdefault(int(os.path.basename(os.path.dirname(path))[5:]), []).append(k)

        for i in sorted(by_halo):
            S = sorted(by_halo[i])
            rows = ([(S[0] - 1, 'null.ini')] if S[0] > 0 else []) + \
                   [(k, f'subhalos/halo_{i}/snap_{k}.ini') for k in S] + \
                   ([(S[-1] + 1, 'null.ini')] if S[-1] < last else [])

            traj = []
            for k, _ in rows:
                if rs['ok'][i, k]:
                    x, v = rs['x'][i, k], rs['v'][i, k]
                else:
                    # past disruption: continue in a straight line from the nearest snapshot with a potential
                    kk = S[np.argmin(np.abs(np.array(S) - k))]
                    v = rs['v'][i, kk]
                    x = rs['x'][i, kk] + v * (t[k] - t[kk])  # km/s * (kpc / (km/s)) time units
                traj.append(f"{t[k]} " + " ".join(f"{c:.9g}" for c in np.concatenate([x, v])))
            with open(os.path.join(sub_dir, f'traj_{i}.txt'), 'w') as f:
                f.write("\n".join(traj) + "\n")

            lines += [f"[Potential halo_{i}]", "type=Evolving", "interpLinear=True", f"center=subhalos/traj_{i}.txt",
                      "Timestamps"]
            lines += [f"{t[k]} {name}" for k, name in rows]
            lines.append("")

        with open(os.path.join(self.write_dir, 'potential.ini'), 'w') as f:
            f.write("\n".join(lines) + "\n")

def is_bound(q, p, subhalo_pos, subhalo_vel, params):
    dq = q - subhalo_pos
    dp = p - subhalo_vel

    ke = np.sum(dp**2, axis=1) / 2
    ok = np.ones(len(ke), dtype=bool)

    if (dq.size == 0) or (len(ke) == 0):
        return np.array([], dtype=bool)

    for _ in range(3):
        if (np.sum(ok) == 0) or (len(dq) == 0):
            return ok
        _, vmax, pe, _ = symlib.profile_info(params, dq, ok=ok)
        E = ke + pe * vmax**2
        ok = E < 0

    return ok


def get_bounded_particles(q, p, subhalo_pos, subhalo_vel, params):
    ok = is_bound(q, p, subhalo_pos, subhalo_vel, params)
    return q[ok], p[ok]
