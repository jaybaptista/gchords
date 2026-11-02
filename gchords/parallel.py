'''
Chunked process parallelism for gchords' per-cluster and per-snapshot loops.

run_chunked(task, items, n_workers) splits items into chunks and returns [task(chunk) for chunk in chunks], in
order. With n_workers > 1 the chunks run in a 'fork' multiprocessing pool, so task may be a closure over large
objects (the interface, tidal field, potential): forked workers inherit them without pickling. Only each chunk's
items and result are sent between processes.

Workers set agama to one OpenMP thread, so n_workers processes use n_workers cores. Forking after the parent has
used agama's OpenMP threads was tested to be safe (results bit-identical to the parent's).
'''
import multiprocessing as mp
import os

from tqdm import tqdm

_TASK = None


def available_cores(default=1):
    '''Cores this process may use: SLURM_CPUS_PER_TASK, else the CPU affinity mask, else default.'''
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        return int(os.environ['SLURM_CPUS_PER_TASK'])
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return default


def _run(chunk):
    import agama
    import numpy as np
    agama.setNumThreads(1)
    # Forked workers inherit the parent's global numpy RNG state, so without this every worker would draw the same
    # stream. symlib's Particles.read draws from it to dither its 16-bit-quantized positions, so the potential
    # builds would get identical dither patterns in every worker. Reseed from OS entropy, as a fresh process would.
    np.random.seed()
    return _TASK(chunk)


def run_chunked(task, items, n_workers=1, chunk_size=None, desc=None):
    '''
    [task(chunk) for chunk in chunks of items], in order. chunk_size defaults to about 4 chunks per worker (for
    load balancing). n_workers <= 1 runs serially in this process.
    '''
    global _TASK
    items = list(items)
    if not items:
        return []
    n_workers = max(1, int(n_workers))
    if chunk_size is None:
        chunk_size = max(1, -(-len(items) // (4 * n_workers)))
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    if n_workers == 1:
        return [task(c) for c in tqdm(chunks, desc=desc, disable=desc is None)]

    _TASK = task
    try:
        with mp.get_context('fork').Pool(min(n_workers, len(chunks))) as pool:
            return list(tqdm(pool.imap(_run, chunks), total=len(chunks), desc=desc, disable=desc is None))
    finally:
        _TASK = None
