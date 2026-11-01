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

def fit_einasto(r, mp, rvir, r_conv, bins=50, p0=(0.18, 10, 5), bounds=((0.175, 0.01, 1.0), (0.185, 50., 10.)),
                maxfev=1000):
    '''Fits einasto_log_density to the binned density of radii r beyond r_conv. Returns (alpha, Rs, logScaleDensity).'''
    from scipy.optimize import curve_fit
    radii, rho = get_density_profile(r, mp, rvir, bins)
    mask = (radii > r_conv) & (rho > 0)
    popt, _ = curve_fit(einasto_log_density, radii[mask], np.log10(rho[mask]), p0=p0, bounds=bounds, maxfev=maxfev,
                        nan_policy="omit")
    return popt, np.sum(mask)

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

        d = np.array([self.phi.eval(x_i, der=True, t=t_i) for x_i, t_i in zip(x, t)])

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
        files = []
        in_timestamps = False

        with open(potential_ini) as f:
            for line in f:
                line = line.strip()
                if line == 'Timestamps':
                    in_timestamps = True
                    continue
                if not in_timestamps:
                    continue
                if not line:
                    break
                files.append(line.split()[1])

        return files

    def _existing_snapshot_numbers(self):
        return sorted(
            int(f.split('_')[1].split('.')[0])
            for f in os.listdir(self.write_dir) if f.startswith('snap_') and f.endswith('.ini')
        )

    def construct_potential(self, lmax, rmin=1e-2, rmax_rvir=1.2, nrad=25, mode='all', min_particles=100, symmetry='None', stride=1, overwrite=False,
                            spherical_profile=False, spherical_profile_kwargs=None):
        '''
        Builds the host's evolving multipole potential in write_dir. If spherical_profile is True, also fits the
        Einasto / NFW spherical approximation of every halo at every snapshot (SymphonyPotential.spherical_profile,
        called with spherical_profile_kwargs) into write_dir/spherical_profile.npz -- this runs even when the
        multipole potential already exists and is not rebuilt. An existing spherical_profile.npz is only recomputed
        if spherical_profile_kwargs contains overwrite=True.
        '''
        os.makedirs(self.write_dir, exist_ok=True)

        if spherical_profile:
            self.spherical_profile(**(spherical_profile_kwargs or {}))

        if self.potential_exists and not overwrite:
            print("Potential already exists. Use `overwrite=True` to reconstruct potential.")
            return

        if overwrite:
            start_snapshot_loop = 0
        else:
            existing = self._existing_snapshot_numbers()
            if existing:
                resume_snapshot = existing[-1]
                print(
                    f"Found an incomplete potential -- resuming construction from snapshot "
                    f"{resume_snapshot}, which is regenerated in case it was left corrupted "
                    f"by a previous failed run."
                )
                os.remove(os.path.join(self.write_dir, f'snap_{resume_snapshot}.ini'))
                start_snapshot_loop = resume_snapshot
            else:
                start_snapshot_loop = 0

        for snapshot in tqdm(range(start_snapshot_loop, len(self.interface.scale_factors)), desc="Constructing potentials..."):

            p_snap = self.interface.particles.read(snapshot, mode=mode, halo=0)
            x = p_snap['x'][p_snap['ok']]

            if len(x) < min_particles:
                # skip this snapshot if there are too few particles to construct a reliable potential
                continue

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

            potential.export(os.path.join(self.write_dir, f'snap_{snapshot}.ini'))

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
                          min_fit_bins=5, p0=(0.18, 10, 5), bounds=((0.175, 0.01, 1.0), (0.185, 50., 10.)),
                          maxfev=1000, rs_max_rvir=1.0, file_name='spherical_profile.npz', overwrite=False):
        '''
        Fits a spherical density profile to every halo (host = 0 and subhalos) at every snapshot and writes it to
        write_dir/file_name.

        For each tracked halo (rs['ok']), the valid particles it owns are centred on its Rockstar position, and an
        Einasto profile is fit to the log density in `bins` equal-count radial bins of the particles inside its
        Rockstar rvir, using only bins beyond the convergence radius (get_convergence_radius; the same for all halos
        at a snapshot). This is the fit in energy_sorting/test_inner_extrapolation.ipynb (ported from mwgcs), except
        that if rs_max_rvir is not None the upper bound on Rs is rs_max_rvir * rvir of that halo at that snapshot
        instead of bounds[1][1] (a fixed 50 kpc pinned the late-time MWest hosts at the bound).

        If the Einasto fit is not possible -- fewer than min_fit_particles valid particles between r_conv and rvir,
        fewer than min_fit_bins bins beyond r_conv, or curve_fit fails -- the halo is instead given the NFW profile
        with its Rockstar mass m inside rvir and concentration cvir. Untracked halos (not yet formed, or disrupted)
        are NaN.

        params[..., :] = (alpha, Rs, log10_rho_s). For both profiles Rs is r_-2 (the radius where dlog rho / dlog r
        = -2) in physical kpc, and rho_s = rho(r_-2) in Msun / kpc^3 (for NFW this is rho_0 / 4); alpha is NaN for
        NFW. Evaluate with einasto_log_density(r, *params) or nfw_log_density(r, *params[1:]) according to
        profile_type.

        Returns a dict of the saved arrays:
            params (n_halos, n_snapshots, 3), profile_type (n_halos, n_snapshots; PROFILE_UNTRACKED = -1,
            PROFILE_EINASTO = 0, PROFILE_NFW = 1), n_fit_bins and n_fit_particles (n_halos, n_snapshots; bins /
            particles beyond r_conv used or available for the Einasto fit), hit_bounds (Einasto fit ended on a bound
            of Rs or log10_rho_s), halos (n_halos), snapshots, scale_factor, time (n_snapshots; Agama units),
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
        r_conv_snap = np.array([get_convergence_radius(params, self.interface.scale_factors[s]) for s in snapshots])
        lo_b, hi_b = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)

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
                n_out = np.sum((r > r_conv) & (r < rvir))
                n_fit_particles[i, j] = n_out
                lo_h, hi_h = lo_b.copy(), hi_b.copy()
                if rs_max_rvir is not None:
                    hi_h[1] = rs_max_rvir * rvir

                popt = None
                if n_out >= min_fit_particles:
                    # halo-scaled starting point, inside the bounds
                    m_in = mp * np.sum(r < rvir)
                    guess = np.clip([p0[0], 0.2 * rvir, np.log10(m_in / (4 * np.pi * (0.2 * rvir)**3))],
                                    lo_h + 1e-6 * (hi_h - lo_h), hi_h - 1e-6 * (hi_h - lo_h))
                    try:
                        popt, nb = fit_einasto(r, mp, rvir, r_conv, bins=bins, p0=guess,
                                               bounds=(lo_h, hi_h), maxfev=maxfev)
                        n_fit_bins[i, j] = nb
                        if nb < min_fit_bins or not np.all(np.isfinite(popt)):
                            popt = None
                    except (RuntimeError, ValueError, TypeError):
                        popt = None

                if popt is not None:
                    out_params[i, j] = popt
                    profile_type[i, j] = PROFILE_EINASTO
                    tol = 1e-3 * (hi_h[1:] - lo_h[1:])
                    hit_bounds[i, j] = np.any((popt[1:] - lo_h[1:] < tol) | (hi_h[1:] - popt[1:] < tol))
                else:
                    Rs, log_rho_s = nfw_from_mass(rs['m'][h, snapshot], rvir, rs['cvir'][h, snapshot])
                    out_params[i, j] = (np.nan, Rs, log_rho_s)
                    profile_type[i, j] = PROFILE_NFW

        m = np.where(profile_type != PROFILE_UNTRACKED, rs['m'][halos][:, snapshots], np.nan)
        rvir = np.where(profile_type != PROFILE_UNTRACKED, rs['rvir'][halos][:, snapshots], np.nan)
        cvir = np.where(profile_type != PROFILE_UNTRACKED, rs['cvir'][halos][:, snapshots], np.nan)
        out = dict(params=out_params, profile_type=profile_type, n_fit_bins=n_fit_bins,
                   n_fit_particles=n_fit_particles, hit_bounds=hit_bounds, halos=halos, snapshots=snapshots,
                   scale_factor=self.interface.scale_factors[snapshots], time=self._agama_times()[snapshots],
                   r_conv=r_conv_snap, m=m, rvir=rvir, cvir=cvir,
                   param_names=np.array(['alpha', 'Rs', 'log10_rho_s']),
                   rs_max_rvir=np.nan if rs_max_rvir is None else rs_max_rvir)
        np.savez_compressed(path, **out)
        return out

    def construct_potential_subhalo(self, halo_index, lmax, rmin=1e-2, rmax_rvir=1.2, nrad=25, mode='all', min_particles=100, symmetry='None', stride=1, overwrite=False):
        os.makedirs(self.write_dir, exist_ok=True)

        # check if it does exist 
        subhalo_potential_exists = False

        if subhalo_potential_exists and not overwrite:
            print("Potential already exists. Use `overwrite=True` to reconstruct potential.")
            return 
        
        
            

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
