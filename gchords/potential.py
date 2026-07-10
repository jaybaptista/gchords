from abc import ABC, abstractmethod
import os
import symlib
import agama
import numpy as np
from tqdm import tqdm

agama.setUnits(length=1.0, mass=1.0, velocity=1.0)

class Potential(ABC):
    def __init__(self, interface, write_dir):
        self.interface = interface
        self.phi = None
        self.write_dir = write_dir
        self.potential_exists = False

        if os.path.exists(os.path.join(self.write_dir, 'potential.ini')):
            print("Potential file already exists. Reading potential from file...")
            self.read_potential(os.path.join(self.write_dir, 'potential.ini'))
            self.potential_exists = True

    @abstractmethod
    def construct_potential(self):
        pass

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

        derivatives = np.array([self.phi.eval(x_i, der=True, t=t_i) for x_i, t_i in zip(x, t)])

        d2phidx2  = derivatives[:, 0]
        d2phidy2  = derivatives[:, 1]
        d2phidz2  = derivatives[:, 2]
        d2phidxdy = derivatives[:, 3]
        d2phidydz = derivatives[:, 4]
        d2phidxdz = derivatives[:, 5]
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
    def construct_potential(self, lmax, rmin=1e-2, rmax_rvir=1.2, nrad=25, mode='all', min_particles=100, symmetry='None', stride=1, overwrite=False):
        os.makedirs(self.write_dir, exist_ok=True)

        if self.potential_exists and not overwrite:
            print("Potential already exists. Use `overwrite=True` to reconstruct potential.")
            return

        for snapshot in tqdm(range(len(self.interface.scale_factors)), desc="Constructing potentials..."):
            
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

        start_snapshot = min(
            int(f.split('_')[1].split('.')[0]) for f in os.listdir(self.write_dir) if f.startswith('snap_')
        )

        end_snapshot = len(self.interface.scale_factors) - 1

        # times in Agama internal time units (~0.97779222 Gyr)
        age_int = self.interface.cosmology.age(1/self.interface.scale_factors - 1) / 0.97779222

        with open(os.path.join(self.write_dir, 'potential.ini'), 'w') as f:
            f.write("[Potential]\n")
            f.write("type=Evolving\n")
            f.write("interpLinear=True\n")
            f.write("Timestamps\n")
            for k in np.arange(start_snapshot, end_snapshot + 1, stride):
                f.write(f"{age_int[k]} snap_{k}.ini\n")
            f.write("\n")

        self.phi = agama.Potential(file=os.path.join(self.write_dir, 'potential.ini'))
        self.potential_exists = True

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
