import numpy as np
import pandas as pd
import symlib
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from gchords.sampler import ValcinAgeModel
from gchords import potential

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
        self.pool_files = None
        self.pool_boundaries = None

    def set_particle_tags(self, particle_tags):
        self.particle_tags = particle_tags
        self.is_manually_tagged = True
        _ = self.find_unique_particles()

    def set_particle_tracks(self, particle_tracks):
        self.particle_tracks = particle_tracks['xv']
        self.particle_indices = particle_tracks['particle_index']
        self.index_to_pos = particle_tracks['mapping'].item()

    def set_particle_pool(self, files):
        '''
        Pool particle tags across a set of realization csv files (e.g. from
        generate_clusters()), concatenating their rows into particle_tags.

        files: array of csv paths. Row order and per-file boundaries are
            recorded so that write_to_pool() can later write results back
            to each original file.
        '''
        self.pool_files = list(files)
        dfs = [pd.read_csv(f) for f in self.pool_files]
        self.pool_boundaries = np.cumsum([0] + [len(df) for df in dfs])
        self.set_particle_tags(pd.concat(dfs, ignore_index=True))

    def write_to_pool(self, xv=True, bound=False):
        '''
        Write particle_tags (e.g. after compute_cluster_masses()) back to
        the original files pooled by set_particle_pool(), splitting rows
        by the same per-file boundaries used to concatenate them.

        xv: if True, also attach each particle's z=0 phase-space
            coordinates (x, y, z, vx, vy, vz) from particle_tracks.
        bound: if True, also attach an `is_bound` flag for each particle,
            indicating whether it is still bound to its progenitor subhalo
            at the last snapshot. False if the subhalo disrupts, is
            untracked (rs['ok'] is False) at the last snapshot, or the
            particle is dynamically unbound from its subhalo's remaining
            particles.
        '''
        if self.pool_files is None:
            raise ValueError("No particle pool found. Run set_particle_pool() first.")

        df = self.particle_tags

        if xv:
            if self.particle_tracks is None:
                raise ValueError("No particle tracks found. Run track_clusters() first.")

            # last snapshot of each particle's track is its z=0 phase-space position
            pos = np.array([
                self.particle_tracks[-1, self.index_to_pos[idx], :]
                for idx in df["particle_index"]
            ])
            df = df.copy()
            df[["x", "y", "z", "vx", "vy", "vz"]] = pos

        if bound:
            last_snapshot = len(self.interface.scale_factors) - 1
            is_bound = pd.Series(True, index=df.index)

            for halo_index, group in tqdm(
                df.groupby("halo_index"), desc="Checking cluster boundedness..."
            ):
                subhalo_lost = (
                    (group["disrupt_snap"].values[0] != -1)
                    or (not self.interface.rs[halo_index, last_snapshot]["ok"])
                )

                if subhalo_lost:
                    is_bound.loc[group.index] = False
                    continue

                p_sub = self.interface.particles.read(last_snapshot, mode="stars", halo=halo_index)
                ok = p_sub["ok"]
                pos_in_filtered = np.cumsum(ok) - 1

                subhalo_pos = self.interface.rs[halo_index, last_snapshot]["x"]
                subhalo_vel = self.interface.rs[halo_index, last_snapshot]["v"]

                bound_ok = potential.is_bound(
                    p_sub["x"][ok], p_sub["v"][ok], subhalo_pos, subhalo_vel, self.interface.params
                )

                for i in group.index:
                    idx = df.at[i, "nimbus_index"]
                    is_bound.loc[i] = bool(ok[idx]) and bool(bound_ok[pos_in_filtered[idx]])

            df = df.copy() if df is self.particle_tags else df
            df["is_bound"] = is_bound

        for k, f in enumerate(self.pool_files):
            df.iloc[self.pool_boundaries[k]:self.pool_boundaries[k + 1]].to_csv(f, index=False)

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
            infall_snap = infall_snapshots[k]
            infall_a = self.interface.scale_factors[infall_snap]
            infall_z = 1.0 / infall_a - 1.0

            _, _, _mgcs, _lgcs = self.gc_halo_model.generate(
                halo_mass=self.interface.infall_properties["halo_mass"][k],
                stellar_mass=self.interface.infall_properties["stellar_mass"][k],
                z=infall_z,
            )

            # skip halos with no GCs or infall onto non-central host
            if (len(_mgcs) == 0) or (self.interface.infall_properties["preinfall_host_idx"][k] != -1):
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
                    z=infall_z,
                )
                if len(_mgcs) == 0:
                    break
            else:
                # after max_tries, truncate to however many slots are available
                _mgcs = _mgcs[:n_available]
                _lgcs = _lgcs[:n_available]

            if len(_mgcs) == 0:
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
                continue

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

        np.savez_compressed(
            write_dir,
            tidal_field=np.array(_st, dtype=object),
            time=np.array(_t, dtype=object),
            integrated_tidal_field=_int_st,
            particle_index=self.particle_indices,
        )

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

        if self.pool_files is not None:
            self.write_to_pool(xv=False)

        return self.particle_tags['evolved_mass'].values

    def check_consistency(
        self,
        particles_npz='particles.npz',
        tidal_field_npz='tidal_field.npz',
        particle_tags=None,
        evolved_csv=None,
        raise_on_failure=True,
    ):
        '''
        Validate that particle_tags, particles.npz (orbits), tidal_field.npz
        (tidal history), and (optionally) an evolved-mass csv all refer to
        the same set of particles in the same order.

        particle_tags: optional DataFrame (e.g. a fresh concat of the
            per-realization CSVs) to check row-for-row against
            self.particle_tags. Skipped if None.
        evolved_csv: optional path to a compute_cluster_masses() output csv
            to check for traceability back to particles.npz. Skipped if None.
        raise_on_failure: if True (default), raises ValueError listing every
            failed check. If False, returns the list of failure strings
            instead (empty list if everything passed).
        '''
        if self.particle_tags is None:
            raise ValueError("No particle tags found. Run generate_clusters() first.")

        failures = []
        n_checks = [0]

        def _check(name, ok, detail=None):
            n_checks[0] += 1
            print(f"[check_consistency] {name}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(detail if detail is not None else name)

        # 1. row count: concatenated CSVs vs particle_tags
        if particle_tags is not None:
            ok = len(particle_tags) == len(self.particle_tags)
            _check(
                "row count (supplied particle_tags vs self.particle_tags)",
                ok,
                f"Row count mismatch: supplied particle_tags has {len(particle_tags)} rows, "
                f"self.particle_tags has {len(self.particle_tags)} rows.",
            )

        particles_data = np.load(particles_npz, allow_pickle=True)
        tidal_data = np.load(tidal_field_npz, allow_pickle=True)

        npz_particle_index = particles_data["particle_index"]
        mapping = particles_data["mapping"].item()

        # 2. orphaned tags: every particle_index in particle_tags must be tracked
        tag_idx = set(self.particle_tags["particle_index"].unique().tolist())
        npz_idx = set(npz_particle_index.tolist())
        orphans = tag_idx - npz_idx
        _check(
            "orphaned particle_index (particle_tags vs particles.npz)",
            not orphans,
            f"{len(orphans)} particle_index value(s) in particle_tags are missing from "
            f"particles.npz's particle_index (e.g. {sorted(orphans)[:10]}).",
        )

        # 3. particles.npz internal self-consistency: mapping vs particle_index array
        mapping_mismatches = [
            (idx, pos) for idx, pos in mapping.items()
            if pos >= len(npz_particle_index) or npz_particle_index[pos] != idx
        ]
        ok = (not mapping_mismatches) and (len(mapping) == len(npz_particle_index))
        _check(
            "particles.npz mapping vs particle_index self-consistency",
            ok,
            f"particles.npz mapping is inconsistent with its particle_index array: "
            f"{len(mapping_mismatches)} mismatched entries (e.g. {mapping_mismatches[:10]}), "
            f"len(mapping)={len(mapping)} vs len(particle_index)={len(npz_particle_index)}.",
        )

        # 4. row-count match: tidal_field.npz arrays vs particles.npz particle_index count
        n_expected = len(npz_particle_index)
        counts = {
            "tidal_field": len(tidal_data["tidal_field"]),
            "time": len(tidal_data["time"]),
            "integrated_tidal_field": len(tidal_data["integrated_tidal_field"]),
        }
        ok = all(c == n_expected for c in counts.values())
        _check(
            "tidal_field.npz row counts vs particles.npz particle_index count",
            ok,
            f"tidal_field.npz array lengths {counts} do not all match "
            f"particles.npz's particle_index count ({n_expected}).",
        )

        # 5. row identity/order between tidal_field.npz and particles.npz
        if "particle_index" in tidal_data.files:
            ok = np.array_equal(tidal_data["particle_index"], npz_particle_index)
            _check(
                "tidal_field.npz particle_index matches particles.npz particle_index (order+identity)",
                ok,
                "tidal_field.npz's particle_index array does not match particles.npz's "
                "particle_index array element-wise -- the two files are misaligned.",
            )
        else:
            print(
                "[check_consistency] WARNING: tidal_field.npz has no 'particle_index' key "
                "(older file format) -- only row counts were verified (check 4); row order/identity "
                "cannot be confirmed. Re-run compute_cluster_tidal_field() to get a self-describing file."
            )

        # 6. per-particle shape sanity inside tidal_field.npz
        shape_mismatches = [
            i for i in range(len(tidal_data["time"]))
            if len(tidal_data["time"][i]) != len(tidal_data["tidal_field"][i])
        ]
        _check(
            "tidal_field.npz per-particle time/tidal_field shape match",
            not shape_mismatches,
            f"{len(shape_mismatches)} particle(s) have mismatched time/tidal_field lengths "
            f"(e.g. indices {shape_mismatches[:10]}).",
        )

        # 7. evolved-output traceability
        if evolved_csv is not None:
            evolved_df = pd.read_csv(evolved_csv)
            ok = len(evolved_df) == len(self.particle_tags)
            _check(
                "evolved csv row count vs self.particle_tags",
                ok,
                f"{evolved_csv} has {len(evolved_df)} rows, self.particle_tags has "
                f"{len(self.particle_tags)} rows.",
            )

            evolved_orphans = set(evolved_df["particle_index"].unique().tolist()) - set(mapping.keys())
            _check(
                "evolved csv particle_index all present in particles.npz mapping",
                not evolved_orphans,
                f"{len(evolved_orphans)} particle_index value(s) in {evolved_csv} are missing "
                f"from particles.npz's mapping (e.g. {sorted(evolved_orphans)[:10]}).",
            )

        print(f"[check_consistency] {n_checks[0] - len(failures)}/{n_checks[0]} checks passed.")

        if failures and raise_on_failure:
            raise ValueError(
                "check_consistency() found the following issue(s):\n- " + "\n- ".join(failures)
            )
        return failures

    def apply_selection_function(self, selection_function, write_dir='particles_selected.csv'):
        if self.particle_tags is None:
            raise ValueError("No particle tags found. Run generate_clusters() first.")
        
        if self.particle_tracks is None:
            raise ValueError("No particle tracks found. Run track_clusters() first.")
        
        x_obs = selection_function.x_obs
        v_obs = selection_function.v_obs
        selected = selection_function.select_particles((x_obs, v_obs))