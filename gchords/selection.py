from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord

'''
Makes mock selections and generates a CSV of observable properties for the selected particles.
'''

DEFAULT_GALCEN_FRAME = coord.Galactocentric(
    galcen_distance=8 * u.kpc,
    z_sun=0 * u.pc,
    roll=0 * u.deg,
)


class SelectionFunction(ABC):
    # fields generate_survey() needs from the particle dataframe
    required_fields = [
        "halo_index",
        "gc_mass",
        "gc_lum",
        "evolved_mass",
        "x", "y", "z",
        "vx", "vy", "vz",
        "is_bound",
    ]

    def __init__(self, frame=None, dist_cut=300.0, r_gc_cut=12.0, seed=None):
        if frame is None:
            frame = DEFAULT_GALCEN_FRAME
        self.frame = frame  # astropy coordinate frame giving the observer's position/orientation
        self.dist_cut = dist_cut  # heliocentric distance cut, kpc
        self.r_gc_cut = r_gc_cut  # inner galactocentric distance cut, kpc
        self.rng = np.random.default_rng(seed)

    def check_required_fields(self, particles):
        missing = [f for f in self.required_fields if f not in particles.columns]
        if missing:
            raise ValueError(f"particles dataframe missing required fields: {missing}")

    def random_rotation_matrix(self):
        theta_x, theta_y, theta_z = self.rng.uniform(0.0, 2 * np.pi, size=3)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)],
        ])
        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)],
        ])
        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1],
        ])

        return Rz @ Ry @ Rx

    def apply_random_rotation(self, particles):
        '''
        Returns a copy of particles with x,y,z and vx,vy,vz rotated by the
        same random rotation matrix.
        '''
        R = self.random_rotation_matrix()

        particles = particles.copy()
        particles[["x", "y", "z"]] = particles[["x", "y", "z"]].values @ R.T
        particles[["vx", "vy", "vz"]] = particles[["vx", "vy", "vz"]].values @ R.T

        return particles

    def get_galactocentric_radius(self, particles):
        return np.linalg.norm(particles[["x", "y", "z"]].values, axis=1)

    def get_heliocentric_distance(self, particles):
        pos = coord.SkyCoord(
            x=particles["x"].values * u.kpc,
            y=particles["y"].values * u.kpc,
            z=particles["z"].values * u.kpc,
            frame=self.frame,
        )
        icrs = pos.transform_to(coord.ICRS())
        return icrs.distance.to(u.kpc).value, icrs.ra.deg, icrs.dec.deg

    def get_final_luminosity(self, particles):
        # preserve each cluster's initial mass-to-light ratio as its mass evolves
        ml_ratio = particles["gc_mass"] / particles["gc_lum"]
        return particles["evolved_mass"] / ml_ratio

    @abstractmethod
    def select_particles(self, particles, rotate=False):
        pass

    @abstractmethod
    def generate_survey(self, particles, rotate=False):
        pass


class SimpleGaiaSelectionFunction(SelectionFunction):
    def __init__(self, frame=None, dist_cut=300.0, r_gc_cut=12.0, gaia_g_limit=21.0, lat_cut=10.0, seed=None):
        super().__init__(frame, dist_cut=dist_cut, r_gc_cut=r_gc_cut, seed=seed)
        self.gaia_g_limit = gaia_g_limit
        self.lat_cut = lat_cut  # minimum |galactic latitude|, deg

    def get_gaia_g(self, luminosity):
        '''
        returns absolute Gaia G magnitude from luminosity
        Solar G band is roughly 4.7 (https://arxiv.org/pdf/1904.04841)
        '''
        M_sun_G = 4.7
        return M_sun_G - 2.5 * np.log10(luminosity)

    def get_apparent_g(self, M_G, distance):
        # distance in kpc
        return M_G + 5 * np.log10(distance * 100)

    def select_particles(self, particles, rotate=False):
        self.check_required_fields(particles)

        if rotate:
            particles = self.apply_random_rotation(particles)

        r_gc = self.get_galactocentric_radius(particles)
        distance, ra, dec = self.get_heliocentric_distance(particles)
        luminosity = self.get_final_luminosity(particles)
        app_g = self.get_apparent_g(self.get_gaia_g(luminosity), distance)

        galactic_b = coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic.b.deg

        return (
            (~particles["is_bound"].values)
            & (r_gc >= self.r_gc_cut)
            & (distance <= self.dist_cut)
            & (app_g < self.gaia_g_limit)
            & (np.abs(galactic_b) >= self.lat_cut)
        )

    def generate_survey(self, particles, rotate=False):
        self.check_required_fields(particles)

        if rotate:
            particles = self.apply_random_rotation(particles)

        mask = self.select_particles(particles)
        selected = particles.loc[mask]

        distance, _, _ = self.get_heliocentric_distance(selected)
        luminosity = self.get_final_luminosity(selected)

        return pd.DataFrame(
            {
                "distance": distance,
                "progenitor_halo": selected["halo_index"].values,
                "luminosity": luminosity.values,
                "mass": selected["evolved_mass"].values,
            }
        )
