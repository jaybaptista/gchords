'''
Rewrite the times of an existing BFE potential directory (full_cosmo.dat with sph_bfe_<k>/ coefficients and
traj/traj_<i>.txt centres) to gchords' convention, agama_times() in potential.py, which potential.ini and
circular_energy_radius.npz also use.

Every timestamp in full_cosmo.dat (sph_bfe_<k>/... and null.dat lines) and the first column of every traj file
is mapped to its snapshot through the host's (halo_0) sph_bfe_<k> lines, then replaced with that snapshot's
new time. Nothing else in the files changes. Any time that matches no snapshot stops the rewrite before writing.
Originals are kept as full_cosmo.dat.orig_<tag> and traj.orig_<tag>/.

usage: python -m gchords.retime HALO_DIR SIM_DIR [--write] [--tag TAG]
'''
import argparse
import datetime
import glob
import os
import re
import shutil

import numpy as np
import symlib
from colossus.cosmology import cosmology

from .potential import agama_times

TIMESTAMP = re.compile(r'^(\S+) (null\.dat|sph_bfe_(\d+)/\S+)$')


def symphony_agama_times(sim_dir):
    params = symlib.simulation_parameters(sim_dir)
    cosmo = cosmology.setCosmology("cosmo", params=symlib.colossus_parameters(params))
    return agama_times(cosmo, symlib.scale_factors(sim_dir))


def retime_bfe_potential(halo_dir, sim_dir, write=False, tag=None):
    new_t = symphony_agama_times(sim_dir)
    fc_path = os.path.join(halo_dir, 'full_cosmo.dat')
    lines = open(fc_path).read().split('\n')

    old_t = {}
    for line in lines:
        m = TIMESTAMP.match(line)
        if m and m.group(3) is not None and '/coef_subhalo_0.' in m.group(2):
            old_t[int(m.group(3))] = float(m.group(1))
    if sorted(old_t) != list(range(len(new_t))):
        raise ValueError(f"{fc_path}: host entries are not snapshots 0..{len(new_t) - 1}")
    old_t = np.array([old_t[k] for k in range(len(new_t))])
    if not np.all(np.diff(old_t) > 0):
        raise ValueError(f"{fc_path}: host times are not increasing")
    if np.allclose(old_t, new_t, rtol=1e-12, atol=0):
        print(f"{halo_dir}: times already in the gchords convention, nothing to do")
        return False

    def new_time(t):
        k = int(np.argmin(np.abs(old_t - t)))
        if not np.isclose(old_t[k], t, rtol=1e-12, atol=0):
            raise ValueError(f"{halo_dir}: time {t!r} matches no snapshot")
        return repr(float(new_t[k])), k

    out, n_ts = [], 0
    for line in lines:
        m = TIMESTAMP.match(line)
        if m:
            t, k = new_time(float(m.group(1)))
            if m.group(3) is not None and int(m.group(3)) != k:
                raise ValueError(f"{fc_path}: {m.group(2)} has the time of snapshot {k}")
            out.append(f"{t} {m.group(2)}")
            n_ts += 1
        else:
            out.append(line)

    traj = {}
    for f in sorted(glob.glob(os.path.join(halo_dir, 'traj', 'traj_*.txt'))):
        rows = [r for r in open(f).read().split('\n') if r.strip()]
        traj[f] = '\n'.join(f"{new_time(float(r.split(' ', 1)[0]))[0]} {r.split(' ', 1)[1]}" for r in rows) + '\n'

    n_rows = sum(t.count('\n') for t in traj.values())
    print(f"{halo_dir}: {n_ts} timestamps, {len(traj)} traj files ({n_rows} rows); "
          f"snapshot {len(new_t) - 1}: {old_t[-1]:.6g} -> {new_t[-1]:.6g}")
    if not write:
        return True

    tag = tag or datetime.date.today().strftime('%Y%m%d')
    fc_backup, traj_backup = f"{fc_path}.orig_{tag}", os.path.join(halo_dir, f"traj.orig_{tag}")
    if os.path.exists(fc_backup) or os.path.exists(traj_backup):
        raise FileExistsError(f"{halo_dir}: backup with tag {tag} already exists")
    shutil.copy2(fc_path, fc_backup)
    shutil.copytree(os.path.join(halo_dir, 'traj'), traj_backup)

    # write each file next to its target, then replace it, so no file is ever left half written
    for f, text in list(traj.items()) + [(fc_path, '\n'.join(out))]:
        with open(f + '.tmp', 'w') as fh:
            fh.write(text)
        os.replace(f + '.tmp', f)
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('halo_dir')
    ap.add_argument('sim_dir')
    ap.add_argument('--write', action='store_true', help='back up and rewrite (default: check only)')
    ap.add_argument('--tag', help='backup suffix (default: today, YYYYMMDD)')
    args = ap.parse_args()
    retime_bfe_potential(args.halo_dir, args.sim_dir, write=args.write, tag=args.tag)
