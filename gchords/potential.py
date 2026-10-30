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

    def construct_potential(self, lmax, rmin=1e-2, rmax_rvir=1.2, nrad=25, mode='all', min_particles=100, symmetry='None', stride=1, overwrite=False):
        os.makedirs(self.write_dir, exist_ok=True)

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
