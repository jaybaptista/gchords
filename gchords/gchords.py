import os
import numpy as np
import pandas as pd
import symlib
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from gchords.sampler import ValcinAgeModel
from gchords import potential

class GChords(object):
    def __init__(
        self, interface, gc_halo_model, age_model=None, feh_model=None, **kwargs
    ):
        self.interface = interface
        self.gc_halo_model = gc_halo_model
        self.age_model = age_model if age_model is not None else ValcinAgeModel()
        # None => use [Fe/H] of the tagged star particle
        self.feh_model = feh_model
        self.particle_tags = None
        # particle tagging with Nimbus
        self.weights, _, _ = symlib.tag_stars(
            self.interface.sim_dir,
            self.gc_halo_model.nimbus_model,
        )

        self.is_manually_tagged = False
        self.pool_files = None
        self.pool_boundaries = None
        self.circEr = None
        self.particle_tracks = None
        self.particle_energies = None
        self.circular_tidal_field = None
        self.spherical_profile = None

    def set_particle_tags(self, particle_tags):
        self.particle_tags = particle_tags
        self.is_manually_tagged = True
        _ = self.find_unique_particles()

    def set_particle_tracks(self, particle_tracks):
        self.particle_tracks = particle_tracks['xv']
        self.particle_indices = particle_tracks['particle_index']
        self.index_to_pos = particle_tracks['mapping'].item()
        # specific energy in km^2/s^2 (None for particles.npz files written before energies were tracked)
        self.particle_energies = particle_tracks['energy'] if 'energy' in particle_tracks else None

    def set_circular_energy_radius(self, circ_er):
        '''
        Store the host's circular energy-radius table as self.circEr (a dict of arrays).

        circ_er: path to the npz written by SymphonyPotential.circular_energy_radius(), or the dict it returns.
        '''
        if isinstance(circ_er, (str, os.PathLike)):
            circ_er = np.load(circ_er)
        self.circEr = dict(circ_er)

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
            infall_snap_k = infall_snapshots[k]
            if 0 <= infall_snap_k < len(self.interface.scale_factors):
                z_infall = 1.0 / self.interface.scale_factors[infall_snap_k] - 1.0
            else:
                z_infall = None

            _, _, _mgcs, _lgcs = self.gc_halo_model.generate(
                halo_mass=self.interface.infall_properties["halo_mass"][k],
                stellar_mass=self.interface.infall_properties["stellar_mass"][k],
                z=z_infall,
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
                    z=z_infall,
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
            if self.feh_model is None:
                feh = self.weights[k]["Fe_H"][draws]
            else:
                feh = self.feh_model.sample(len(_mgcs))
            # tracer age is not a GC age prediction; AgeModel only sets the GC-halo normalization
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
            self.particle_tags = df
            self.particle_tags.to_csv(write_dir, index=False)
            return

        self.particle_tags = pd.concat(rows, ignore_index=True)
        self.particle_tags.to_csv(write_dir, index=False)

    

    def track_clusters(self, comoving=False, write_dir='particles.npz', energy=True):
        '''
        Tracks the tagged particles' host-centric phase-space coordinates (xv, shape (n_snapshots, n_tracked, 6))
        from their infall snapshot on, and writes them to write_dir.

        energy: if True, also saves each tracked particle's specific energy E = |v|^2 / 2 + U in km^2/s^2 (shape
            (n_snapshots, n_tracked), NaN before infall), where U is the host's spherical particle potential from
            potential.host_potential_at -- the same U as in SymphonyPotential.circular_energy_radius, so E can be
            compared directly with its E_circ. Needs comoving=False, and reads all host particles at every snapshot.
        '''
        # TODO: add capability to track from a_form snapshot.
        if self.particle_tags is None:
            raise ValueError("No particle tags found. Run generate_clusters() first.")
        if energy and comoving:
            raise ValueError("Particle energies need physical units: use comoving=False or energy=False.")

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
        E = np.full(data.shape[:2], np.nan) if energy else None

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

                if energy:
                    rows = np.flatnonzero(ok.values)
                    p_host = self.interface.particles.read(snapshot, mode="all", halo=0)
                    U = potential.host_potential_at(
                        self.interface.params, p_host["x"], p_host["ok"], data[snapshot, rows, :3]
                    )
                    E[snapshot, rows] = np.sum(data[snapshot, rows, 3:]**2, axis=1) / 2 + U

        self.particle_tracks = data
        self.particle_indices = indices
        out = dict(xv=data, particle_index=indices, mapping=self.index_to_pos)
        if energy:
            self.particle_energies = E
            out["energy"] = E
        np.savez_compressed(write_dir, **out)

    def compute_particle_energies(self, write_dir=None):
        '''
        Specific energies (km^2/s^2) of already-tracked particles, for tracks loaded with set_particle_tracks from a
        particles.npz written before track_clusters saved energies. Same definition as track_clusters(energy=True):
        E = |v|^2 / 2 + potential.host_potential_at, from the tracks' host-centric physical xv (track_clusters'
        default comoving=False). Reads all host particles at every snapshot with a tracked particle.

        Stores self.particle_energies (n_snapshots, n_tracked; NaN where untracked) and, if write_dir is given, saves
        it with particle_index. Returns the array.
        '''
        if self.particle_tracks is None:
            raise ValueError("No particle tracks found. Run track_clusters() or set_particle_tracks() first.")
        xv = self.particle_tracks
        E = np.full(xv.shape[:2], np.nan)
        snaps = np.flatnonzero(np.any(np.isfinite(xv[:, :, 0]), axis=1))
        for snapshot in tqdm(snaps, desc="Computing particle energies..."):
            rows = np.flatnonzero(np.isfinite(xv[snapshot, :, 0]))
            p_host = self.interface.particles.read(snapshot, mode="all", halo=0)
            U = potential.host_potential_at(self.interface.params, p_host["x"], p_host["ok"], xv[snapshot, rows, :3])
            E[snapshot, rows] = np.sum(xv[snapshot, rows, 3:]**2, axis=1) / 2 + U
        self.particle_energies = E
        if write_dir is not None:
            np.savez_compressed(write_dir, energy=E, particle_index=self.particle_indices)
        return E

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

    def compute_cluster_tidal_field(self, potential, write_dir='tidal_field.npz', batch_size=1):
        '''
        Tidal strength along every tracked particle's orbit (100 samples per snapshot from its first snapshot).

        batch_size: number of orbits whose samples are evaluated in one potential.tidal_strength call. Each point is
        evaluated independently, so the result does not depend on it; larger batches give agama's OpenMP threads
        more points per call (useful for many-component subhalo potentials). 1 = one call per orbit.
        '''
        if self.particle_tracks is None:
            raise ValueError("No particle tracks found. Run track_clusters() first.")

        n_tracks = self.particle_tracks.shape[1]
        _t = [None] * n_tracks
        _st = [None] * n_tracks
        _int_st = [None] * n_tracks

        age_int = self.interface.cosmology.age(1/self.interface.scale_factors - 1) / 0.97779222

        with tqdm(total=n_tracks) as bar:
            for b0 in range(0, n_tracks, max(1, int(batch_size))):
                batch = []
                for k in range(b0, min(n_tracks, b0 + max(1, int(batch_size)))):
                    xv = self.particle_tracks[:, k, :]
                    start_snapshot = np.where(~np.isnan(xv[:, 0]))[0][0]
                    xv = xv[start_snapshot:]

                    if start_snapshot == len(self.interface.scale_factors) - 1:
                        _t[k] = [None]
                        _st[k] = [None]
                        _int_st[k] = 0.0
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
                    batch.append((k, t_sample, x_sample))

                if batch:
                    x_all = np.concatenate([b[2] for b in batch])
                    t_all = np.concatenate([b[1] for b in batch])
                    st_all = np.atleast_1d(potential.tidal_strength(x_all, t=t_all / 0.97779222)) * 1.0459401725324529
                    off = 0
                    for k, t_sample, _ in batch:
                        st_sample = st_all[off:off + len(t_sample)]
                        off += len(t_sample)
                        _t[k] = t_sample
                        _st[k] = st_sample
                        spl_st = PchipInterpolator(t_sample, np.sqrt(st_sample)) # root of tidal frequency is `tidal strength`
                        _int_st[k] = spl_st.integrate(t_sample[0], t_sample[-1])
                bar.update(min(n_tracks, b0 + max(1, int(batch_size))) - b0)

        np.savez_compressed(
            write_dir,
            tidal_field=np.array(_st, dtype=object),
            time=np.array(_t, dtype=object),
            integrated_tidal_field=_int_st,
            particle_index=self.particle_indices,
        )

    def set_spherical_profile(self, sph):
        '''
        Store the halos' spherical (Einasto / NFW) fits as self.spherical_profile (a dict of arrays); the host
        (halo 0) fit is used inside the convergence radius by compute_cluster_circular_tidal_field.

        sph: path to the npz written by SymphonyPotential.spherical_profile(), or the dict it returns.
        '''
        if isinstance(sph, (str, os.PathLike)):
            sph = np.load(sph)
        self.spherical_profile = dict(sph)

    def circular_table(self, snapshot, use_profile=True, n_inner=200, r_inner_min=1e-4):
        '''
        The host's circular-orbit table at one snapshot, as used by compute_cluster_circular_tidal_field:
        (r, E_circ, tidal_strength, r_conv, profile_type), with r increasing and E_circ non-decreasing, or None if
        the snapshot has no usable table.

        Outside the convergence radius r_conv it is the particle table from circular_energy_radius, with the
        table's particles sorted separately in E_circ and in r and matched by rank (so E_circ(r) is monotonic) and
        the tidal strength ordered by r. If use_profile is True and the host has a spherical_profile fit at this
        snapshot, everything at r <= r_conv is instead the fit's (profile_circular): its tidal strength, and its
        E_circ shifted by a constant so that it equals the particle table's (interpolated in log r) at r_conv. The
        fit is sampled on n_inner log-spaced radii from min(r_inner_min, r_conv / 1e3) to r_conv. If the host has
        no fit here, or use_profile is False, the particle table is used at every radius (r_conv and
        profile_type are then NaN and PROFILE_UNTRACKED).
        '''
        j = np.flatnonzero(np.asarray(self.circEr["snapshots"]) == snapshot)
        if len(j) == 0:
            return None
        j = j[0]
        r_tab = self.circEr["r"][j]
        Ec_tab = self.circEr["E_circ"][j]
        ts_tab = self.circEr["tidal_strength"][j]
        good = np.isfinite(r_tab) & np.isfinite(Ec_tab) & np.isfinite(ts_tab) & (r_tab > 0) & (ts_tab > 0)
        if np.sum(good) < 2:
            return None
        order = np.argsort(r_tab[good])
        r = r_tab[good][order]
        ts = ts_tab[good][order]
        Ec = np.sort(Ec_tab[good])

        r_conv, ptype = np.nan, potential.PROFILE_UNTRACKED
        if use_profile:
            sph = self.spherical_profile
            k = np.flatnonzero(np.asarray(sph["snapshots"]) == snapshot)
            h = np.flatnonzero(np.asarray(sph["halos"]) == 0)
            if len(k) and len(h):
                k, h = k[0], h[0]
                ptype = int(sph["profile_type"][h, k])
                params = sph["params"][h, k]
                r_conv = float(sph["r_conv"][k])
                ok = ptype != potential.PROFILE_UNTRACKED and np.all(np.isfinite(params[1:])) and r_conv > 0
                if not ok:
                    r_conv, ptype = np.nan, potential.PROFILE_UNTRACKED

        if ptype != potential.PROFILE_UNTRACKED:
            r_in = np.logspace(np.log10(min(r_inner_min, r_conv / 1e3)), np.log10(r_conv), n_inner)
            Ec_in, ts_in = potential.profile_circular(r_in, params, ptype)
            # match the fit's E_circ to the particle table's at r_conv
            Ec_in += np.interp(np.log(r_conv), np.log(r), Ec) - Ec_in[-1]
            out = r > r_conv
            r = np.concatenate([r_in, r[out]])
            Ec = np.concatenate([Ec_in, Ec[out]])
            ts = np.concatenate([ts_in, ts[out]])

        return r, Ec, ts, r_conv, ptype

    def compute_cluster_circular_tidal_field(self, write_dir='circular_tidal_field.npz', use_profile=True,
                                             n_inner=200, r_inner_min=1e-4):
        '''
        For each tracked particle at each snapshot, the radius R_c of the circular orbit with the particle's energy,
        and the tidal strength (Gyr^-2) of the host's spherically-averaged potential there.

        Needs particle energies (track_clusters(energy=True) or set_particle_tracks), the host's circular
        energy-radius table (set_circular_energy_radius; SymphonyPotential.circular_energy_radius), whose E_circ
        uses the same potential as the particle energies, and, if use_profile is True, the halos' spherical fits
        (set_spherical_profile; SymphonyPotential.spherical_profile).

        The particle table cannot be trusted inside the convergence radius, so there both E_circ(r) and the tidal
        strength come from the host's Einasto / NFW fit, with the fit's E_circ matched to the particle table at
        r_conv (see circular_table). R_c is interpolated in log r at the particle's E, and the tidal strength in
        log-log at R_c. Energies outside the table are clamped to its innermost / outermost radius. Snapshots the
        table does not cover are NaN; snapshots without a host fit use the particle table at all radii.

        Saves and returns a dict: r_circ, tidal_strength_circ, inside_r_conv (R_c <= r_conv, so both came from the
        fit) (n_snapshots, n_tracked); r_conv, host_profile_type (n_snapshots; NaN / PROFILE_UNTRACKED where the
        fit was not used); time (n_snapshots; Agama units, as in compute_cluster_tidal_field); particle_index.
        '''
        if self.particle_energies is None:
            raise ValueError("No particle energies found. Run track_clusters(energy=True) first.")
        if self.circEr is None:
            raise ValueError("No circular energy-radius table found. Run set_circular_energy_radius() first.")
        if use_profile and self.spherical_profile is None:
            raise ValueError("No spherical profiles found. Run set_spherical_profile() first, "
                             "or pass use_profile=False to use the particle table at every radius.")

        E = self.particle_energies
        n_snap = E.shape[0]
        r_circ = np.full(E.shape, np.nan)
        ts_circ = np.full(E.shape, np.nan)
        inside = np.zeros(E.shape, dtype=bool)
        r_conv_snap = np.full(n_snap, np.nan)
        ptype_snap = np.full(n_snap, potential.PROFILE_UNTRACKED, dtype=np.int8)

        for snapshot in tqdm(self.circEr["snapshots"], desc="Finding circular orbit radii..."):
            rows = np.flatnonzero(np.isfinite(E[snapshot]))
            if len(rows) == 0:
                continue
            table = self.circular_table(snapshot, use_profile=use_profile, n_inner=n_inner, r_inner_min=r_inner_min)
            if table is None:
                continue
            r, Ec, ts, r_conv, ptype = table
            log_rc = np.interp(E[snapshot, rows], Ec, np.log(r))
            r_circ[snapshot, rows] = np.exp(log_rc)
            ts_circ[snapshot, rows] = np.exp(np.interp(log_rc, np.log(r), np.log(ts)))
            r_conv_snap[snapshot], ptype_snap[snapshot] = r_conv, ptype
            if np.isfinite(r_conv):
                inside[snapshot, rows] = r_circ[snapshot, rows] <= r_conv

        time = self.interface.cosmology.age(1 / self.interface.scale_factors - 1) / 0.97779222
        out = dict(r_circ=r_circ, tidal_strength_circ=ts_circ, inside_r_conv=inside, r_conv=r_conv_snap,
                   host_profile_type=ptype_snap, time=time, particle_index=self.particle_indices)
        np.savez_compressed(write_dir, **out)
        self.circular_tidal_field = out
        return out

    def compute_cluster_masses(self, tidal_data, mass_loss_model, write_dir='particles_evolved.csv', integrated=False,
                               circular_tidal_data=None, n_workers=1):
        '''
        Compute the evolved masses of star clusters over time

        integrated: if True, `st` is time-integrated tidal strength and passes the MassLossModel `integrated` flag.
        circular_tidal_data: output of compute_cluster_circular_tidal_field (dict or npz), needed by mass loss
            models with requires_circular_tidal_field (e.g. CircularOrbitMassLossModel); defaults to the one
            computed in this session.
        n_workers: with integrated=False, evolve the clusters in chunks on this many forked worker processes
            (gchords.parallel.run_chunked). Each cluster is integrated independently, so the masses do not depend
            on it. 1 = the serial loop.
        '''
        needs_circ = getattr(mass_loss_model, "requires_circular_tidal_field", False)
        if needs_circ:
            if integrated:
                raise ValueError("integrated=True is not available for models that need the circular tidal field.")
            circ = circular_tidal_data if circular_tidal_data is not None else getattr(self, "circular_tidal_field", None)
            if circ is None:
                raise ValueError("This mass loss model needs circular_tidal_data "
                                 "(run compute_cluster_circular_tidal_field() first).")
            circ_t = np.asarray(circ["time"])
            circ_ts = np.asarray(circ["tidal_strength_circ"])

        self.particle_tags["evolved_mass"] = np.zeros(len(self.particle_tags))

        if integrated:
            st = tidal_data["integrated_tidal_field"]
            self.particle_tags["evolved_mass"] = mass_loss_model.evolve_mass(initial_mass=self.particle_tags["gc_mass"].values, time=0.0, tidal_strength=st, integrated=True)
        elif n_workers > 1:
            t = tidal_data["time"]
            st = tidal_data["tidal_field"]
            pos_of = [self.index_to_pos[p] for p in self.particle_tags["particle_index"].values]
            m0 = self.particle_tags["gc_mass"].values

            def evolve_chunk(rows):
                out = []
                for i in rows:
                    t_i, st_i = t[pos_of[i]], st[pos_of[i]]
                    if len(t_i) < 2:
                        # untracked cluster: no mass loss, as in the serial loop below
                        out.append(m0[i])
                        continue
                    extra = {}
                    if needs_circ:
                        extra = dict(circular_time=circ_t, circular_tidal_strength=circ_ts[:, pos_of[i]])
                    out.append(mass_loss_model.evolve_mass(initial_mass=m0[i], time=t_i, tidal_strength=st_i,
                                                           integrated=False, **extra))
                return out

            from gchords.parallel import run_chunked
            chunks = run_chunked(evolve_chunk, range(len(self.particle_tags)), n_workers=n_workers,
                                 desc=f"Numerically evolving cluster masses ({n_workers} workers, chunks)...")
            self.particle_tags["evolved_mass"] = np.array([m for c in chunks for m in c], dtype=float)
        else:
            t = tidal_data["time"]
            st = tidal_data["tidal_field"]
            for i in tqdm(range(len(self.particle_tags)), desc="Numerically evolving cluster masses..."):
                # get the time and tidal strength for this particle's track
                t_i = t[self.index_to_pos[self.particle_tags["particle_index"].values[i]]]
                st_i = st[self.index_to_pos[self.particle_tags["particle_index"].values[i]]]
                if len(t_i) < 2:
                    # untracked cluster (particle track starts only at the final snapshot,
                    # see compute_cluster_tidal_field's skip-branch) -- no tidal history to
                    # interpolate against, so treat as no mass loss, matching the
                    # integrated=True path's `_int_st.append(0.0)` sentinel.
                    self.particle_tags["evolved_mass"].values[i] = self.particle_tags["gc_mass"].values[i]
                    continue
                extra = {}
                if needs_circ:
                    pos = self.index_to_pos[self.particle_tags["particle_index"].values[i]]
                    extra = dict(circular_time=circ_t, circular_tidal_strength=circ_ts[:, pos])
                self.particle_tags["evolved_mass"].values[i] = mass_loss_model.evolve_mass(initial_mass=self.particle_tags["gc_mass"].values[i], time=t_i, tidal_strength=st_i, integrated=False, **extra)

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