import numpy as np
import pandas as pd
import symlib
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from gchords.sampler import ValcinAgeModel

class GChords(object):
    def __init__(self, interface, gc_halo_model, age_model=None, **kwargs):
        self.interface = interface
        self.gc_halo_model = gc_halo_model
        self.age_model = age_model if age_model is not None else ValcinAgeModel()
        self.particle_tags = None
        # particle tagging with Nimbus
        self.weights, _, _ = symlib.tag_stars(
            self.interface.sim_dir,
            self.gc_halo_model.nimbus_model,
        )

        self.is_manually_tagged = False

    def set_particle_tags(self, particle_tags):
        self.particle_tags = particle_tags
        self.is_manually_tagged = True
        _ = self.find_unique_particles()

    def set_particle_tracks(self, particle_tracks):
        self.particle_tracks = particle_tracks['xv']
        self.particle_indices = particle_tracks['particle_index']
        self.index_to_pos = particle_tracks['mapping'].item()

    def generate_clusters(self, write_dir='particles.csv', seed=None, **kwargs):
        infall_snapshots = self.interface.infall_properties["infall_snapshot"]
        n_halos = len(infall_snapshots)

        if seed is not None:
            np.random.seed(seed)

        df = pd.DataFrame(
                columns=[
                    "halo_index",
                    "infall_snap",
                    "disrupt_snap",
                    "gc_mass",
                    "gc_lum",
                    "preinfall_host_idx",
                    "nimbus_index",
                    "feh",
                    "a_form",
                    "infall_host_mstar",
                ]
            )
        
        rows = []

        for k in np.arange(1, n_halos):
            _, _, _mgcs, _lgcs = self.gc_halo_model.generate(
                halo_mass=self.interface.infall_properties["halo_mass"][k],
                stellar_mass=self.interface.infall_properties["stellar_mass"][k],
            )

            # skip halos with no GCs or infall onto non-central host
            if (_mgcs is None) or (self.interface.infall_properties["preinfall_host_idx"][k] != -1):
                continue

            mp = self.weights[k]['mp']

            # TODO: Put this back in, I took this out because it runs too slow?
            # p = self.interface.particles.read(infall_snapshots[k], mode='stars')
            # ok = p[k]['ok']
            # set not 'ok' weights to 0
            # mp[~ok] = 0.0
            
            # e.g., if there aren't enough particles
            if np.sum(mp) <= 0.0:
                # if I can't draw a particle tag, then I can't assign a GC, so skip this halo.
                # NOTE: this may set a resolution floor for GC formation
                continue

            n_available = int(np.sum(mp > 0))

            # resample the GC profile until the drawn count fits within the
            # available non-zero particle slots, up to max_tries attempts
            max_tries = 10
            for _ in range(max_tries):
                if len(_mgcs) <= n_available:
                    break
                _, _, _mgcs, _lgcs = self.gc_halo_model.generate(
                    halo_mass=self.interface.infall_properties["halo_mass"][k],
                    stellar_mass=self.interface.infall_properties["stellar_mass"][k],
                )
                if _mgcs is None:
                    break
            else:
                # after max_tries, truncate to however many slots are available
                _mgcs = _mgcs[:n_available]
                _lgcs = _lgcs[:n_available]

            if _mgcs is None or len(_mgcs) == 0:
                continue

            p_draw = mp / np.sum(mp)
            draws = np.random.choice(len(mp), size=len(_mgcs), replace=False, p=p_draw)
            feh = self.weights[k]['Fe_H'][draws]
            a_form = self.weights[k]['a_form'][draws]

            rows.append(
                pd.DataFrame(
                    {
                        "halo_index": np.repeat(k, len(_mgcs)),
                        "infall_snap": np.repeat(infall_snapshots[k], len(_mgcs)),
                        "disrupt_snap": np.repeat(
                            self.interface.infall_properties["disrupt_snapshot"][k],
                            len(_mgcs),
                        ),
                        "gc_mass": _mgcs,
                        "gc_lum": _lgcs,
                        "preinfall_host_idx": np.repeat(
                            self.interface.infall_properties["preinfall_host_idx"][k],
                            len(_mgcs),
                        ),
                        "infall_host_mstar": np.repeat(
                            self.interface.infall_properties["stellar_mass"][k],
                            len(_mgcs),
                        ),
                        "nimbus_index": draws,
                        "feh": feh,
                        "a_form": a_form,
                    }
                )
            )

        if not rows:
            df.to_csv(write_dir, index=False)

        self.particle_tags = pd.concat(rows, ignore_index=True)
        self.particle_tags.to_csv(write_dir, index=False)

    

    def track_clusters(self, comoving=False, write_dir='particles.npz'):
        # TODO: add capability to track from a_form snapshot.
        if self.particle_tags is None:
            raise ValueError("No particle tags found. Run generate_clusters() first.")
        
        n_tracked_particles = len(self.particle_tags)
        indices = self.find_unique_particles()

        if self.is_manually_tagged:
            print('Manually tag particles--only track unique particles')    
            n_tracked_particles = len(indices)

        # defines mapping from particle index to row in tracking array
        self.index_to_pos = {idx: i for i, idx in enumerate(indices)}
        
        data = np.zeros((len(self.interface.scale_factors), n_tracked_particles, 6)) * np.nan
        unique_particle_tags = self.particle_tags.drop_duplicates(subset="particle_index")
        first_infall_snapshot = int(self.particle_tags["infall_snap"].min())

        for snapshot in tqdm(range(first_infall_snapshot, len(self.interface.scale_factors)), desc="Tracking particles across snapshots..."):
            # Load all the subhalos at a given snapshot and their corresponding particles
            particles = self.interface.particles.read(snapshot, mode="stars", comoving=comoving)
            p_flat = np.hstack(particles)
            
            # select unique particles if manually tagged, otherwise select all tagged particles
            ok = self.particle_tags["infall_snap"] <= snapshot
            if self.is_manually_tagged:
                ok = unique_particle_tags["infall_snap"] <= snapshot

            if ok.any():
                if self.is_manually_tagged:
                    i_t = unique_particle_tags["particle_index"][ok]
                    data[snapshot, ok.values, :3] = p_flat[i_t]["x"]
                    data[snapshot, ok.values, 3:] = p_flat[i_t]["v"]
                else:
                    i_t = self.particle_tags["particle_index"][ok]
                    data[snapshot, ok, :3] = p_flat[i_t]["x"]
                    data[snapshot, ok, 3:] = p_flat[i_t]["v"]
        
        self.particle_tracks = data
        self.particle_indices = indices
        np.savez_compressed(write_dir, xv=data, particle_index=indices, mapping=self.index_to_pos)

    def find_unique_particles(self):
        if self.particle_tags is None:
            raise ValueError("No particle tags found. Run generate_clusters() first.")
        
        _p = self.interface.particles.read(len(self.interface.scale_factors)-1, mode="stars", comoving=False)
        sizes = np.array([len(p) for p in _p])
        edges = np.zeros(len(sizes) + 1, int)
        edges[1:] = np.cumsum(sizes)
        starts = edges[:-1]
        
        # flattened index of particles tagged with GCs
        self.particle_tags["particle_index"] = self.particle_tags["nimbus_index"] + starts[self.particle_tags["halo_index"]]
        return self.particle_tags["particle_index"].unique()
    
    def compute_cluster_tidal_field(self, potential, write_dir='tidal_field.npz'):
        if self.particle_tracks is None:
            raise ValueError("No particle tracks found. Run track_clusters() first.")
        
        _t = []
        _st = []
        _int_st = []

        age_int = self.interface.cosmology.age(1/self.interface.scale_factors - 1) / 0.97779222

        for k in tqdm(range(self.particle_tracks.shape[1])):
            xv = self.particle_tracks[:, k, :]
            start_snapshot = np.where(~np.isnan(xv[:, 0]))[0][0]
            xv = xv[start_snapshot:]

            if start_snapshot == len(self.interface.scale_factors) - 1:
                _t.append([None])
                _st.append([None])
                _int_st.append([0.0])
        
            spl = [
                UnivariateSpline(
                    age_int[start_snapshot:], xv[:, i], s=0, k=min(3, len(age_int[start_snapshot:]) - 1)
                )
                for i in range(3)
            ]

            N_samples = 100 * len(age_int[start_snapshot:]) # sample 100 points per snapshot

            t_sample = np.linspace(age_int[start_snapshot], age_int[-1], N_samples)
            x_sample = np.array([spl_i(t_sample) for spl_i in spl]).T
            st_sample = potential.tidal_strength(x_sample, t=t_sample / 0.97779222) * 1.0459401725324529
            _t.append(t_sample)
            _st.append(st_sample)
            spl_st = PchipInterpolator(t_sample, np.sqrt(st_sample)) # root of tidal frequency is `tidal strength`
            _int_st.append(spl_st.integrate(t_sample[0], t_sample[-1]))

        np.savez_compressed(write_dir, tidal_field=np.array(_st, dtype=object), time=np.array(_t, dtype=object), integrated_tidal_field=_int_st)

    def compute_cluster_masses(self, tidal_data, mass_loss_model, write_dir='particles_evolved.csv', integrated=False):
        '''
        Compute the evolved masses of star clusters over time

        integrated: if True, `st` is time-integrated tidal strength and passes the MassLossModel `integrated` flag.
        '''
        self.particle_tags["evolved_mass"] = np.zeros(len(self.particle_tags))

        if integrated:
            st = tidal_data["integrated_tidal_field"]
            self.particle_tags["evolved_mass"] = mass_loss_model.evolve_mass(initial_mass=self.particle_tags["gc_mass"].values, time=0.0, tidal_strength=st, integrated=True)
        else:
            t = tidal_data["time"]
            st = tidal_data["tidal_field"]
            for i in tqdm(range(len(self.particle_tags)), desc="Numerically evolving cluster masses..."):
                # get the time and tidal strength for this particle's track
                t_i = t[self.index_to_pos[self.particle_tags["particle_index"].values[i]]]
                st_i = st[self.index_to_pos[self.particle_tags["particle_index"].values[i]]]
                self.particle_tags["evolved_mass"].values[i] = mass_loss_model.evolve_mass(initial_mass=self.particle_tags["gc_mass"].values[i], time=t_i, tidal_strength=st_i, integrated=False)        

        self.particle_tags.to_csv(write_dir, index=False)
        return self.particle_tags['evolved_mass'].values
    
    def apply_selection_function(self, selection_function, write_dir='particles_selected.csv'):
        if self.particle_tags is None:
            raise ValueError("No particle tags found. Run generate_clusters() first.")
        
        if self.particle_tracks is None:
            raise ValueError("No particle tracks found. Run track_clusters() first.")
        
        x_obs = selection_function.x_obs
        v_obs = selection_function.v_obs
        selected = selection_function.select_particles((x_obs, v_obs))