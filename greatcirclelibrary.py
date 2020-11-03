"""A module to help analyze great circles in maps of the CMB sky."""

import numpy as np
import healpy as hp
from multiprocessing import Pool
from scipy.optimize import differential_evolution
from numba import guvectorize, int64, float64, prange, jit
from functools import partial

NSIDE = 256
DTHETA_INC = 0.14
DIPOLE_RANGE = (10, 100)
AXIS_OF_EVIL_ANGLE = [110 * np.pi / 180, 60 * np.pi / 180]


#
# PREPROCESSING FUNCTIONS
#

@jit(parallel=True)
def preprocess_maps(paths, new_l_max, new_nside=NSIDE, parity=''):
    """
    Preprocess maps of the full CMB sky.

    Take in paths to full CMB maps in .FITS format, then extract select
    multipole moments from downgraded versions of the maps with their monopoles
    and kinetic dipoles removed.
    """
    new_alms = np.zeros([len(paths), hp.sphtfunc.Alm.getsize(new_l_max)],
                        dtype='complex')

    standardized_parity = parity.replace(' ', '').lower()

    if standardized_parity == '':
        start = 2
        step = 1
    elif standardized_parity == 'even':
        start = 2
        step = 2
    elif standardized_parity == 'odd':
        start = 3
        step = 2

    for i in prange(len(paths)):

        dgraded_map = hp.pixelfunc.ud_grade(hp.read_map(paths[i]), new_nside)

        old_alms = hp.sphtfunc.map2alm(dgraded_map)

        old_l_max = hp.sphtfunc.Alm.getlmax(old_alms.shape[0])

        for L in range(start, new_l_max + 1, step):
            for m in range(L + 1):

                old_index = hp.Alm.getidx(old_l_max, L, m)

                new_index = hp.Alm.getidx(new_l_max, L, m)

                new_alms[i][new_index] = old_alms[old_index]

    return new_alms


def shift_to_axis_of_evil(alms):
    """
    Rotate a set of alms from Galactic coordinates to Axis of Evil coordinates.

    In the Axis of Evil coordinate system, the Axis of Evil points in the
    positive z direction.
    """
    for i in range(alms.shape[0]):

        hp.rotate_alm(alms[i], AXIS_OF_EVIL_ANGLE[0], AXIS_OF_EVIL_ANGLE[1], 0)


#
# SIMULATION FUNCTIONS
#

@jit(parallel=True)
def generate_random_reorientations(alms, n):
    """Generate n random reorientation simulations from a given set of alms."""
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    isolated_ls = np.zeros([l_max + 1, alms.shape[0]], dtype='complex')

    for L in range(l_max + 1):

        isolated_ls[L] = get_l(alms, L)

    simulations = np.zeros([n, alms.shape[0]], dtype='complex')

    for i in prange(n):

        new_isolated_ls = np.copy(isolated_ls)

        for L in range(l_max + 1):

            randomly_rotate(new_isolated_ls[L])

        simulations[i] = np.sum(new_isolated_ls, axis=0)

    return simulations


@jit(parallel=True)
def generate_same_cls_simulations(alms, n):
    """Generate n same cl spectrum simulations from a given set of alms."""
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    my_cls = hp.alm2cl(alms)

    simulations = np.zeros([n, alms.shape[0]], dtype='complex')

    for i in prange(n):

        simulations[i] = hp.sphtfunc.synalm(my_cls, lmax=l_max, verbose=False)

        for L in np.flatnonzero(my_cls):

            my_sum = 0

            for m in range(L + 1):

                my_sum += abs(simulations[i][hp.Alm.getidx(l_max, L, m)]) ** 2

            scaling_factor = np.sqrt(my_cls[L] * (1 + 2 * L) / (2 * my_sum))

            for m in range(L + 1):

                simulations[i][hp.Alm.getidx(l_max, L, m)] *= scaling_factor

            simulations[i][hp.Alm.getidx(l_max, L, 0)] *= np.sqrt(2)

    return simulations


@jit(parallel=True)
def generate_cosmic_variance_simulations(alms, n):
    """Generate n simulations with cosmic variance from a given set of alms."""
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    my_cls = hp.alm2cl(alms)

    simulations = np.zeros([n, alms.shape[0]], dtype='complex')

    for i in prange(n):

        simulations[i] = hp.sphtfunc.synalm(my_cls, lmax=l_max, verbose=False)

    return simulations


@jit(parallel=True)
def generate_random_dipole_simulations(alms, n, dipole_range=DIPOLE_RANGE):
    """
    Generate n simulations with random dipoles.

    Given a set of alms, generate n simulations where the dipole of the given
    alms has been replaced by a random dipole.
    """
    simulations = np.zeros([n, alms.shape[0]], dtype='complex')

    new_alms = np.copy(alms)

    remove_l(new_alms, 1)

    dipole_index = hp.Alm.getidx(hp.sphtfunc.Alm.getlmax(alms.shape[0]), 1, 0)

    for i in prange(n):

        new_dipole = np.zeros_like(new_alms)

        new_dipole[dipole_index] = np.random.uniform(*dipole_range)

        randomly_rotate(new_dipole, random_psi=False)

        simulations[i] = new_alms + new_dipole

    return simulations


@jit(parallel=True)
def generate_random_dipole_reorientation_simulations(alms, n):
    """
    Generate n simulations with randomly reoriented dipoles.

    Given a set of alms, generate n simulations where the dipole of the given
    alms has been randomly reoriented.
    """
    simulations = np.zeros([n, alms.shape[0]], dtype='complex')

    new_alms = np.copy(alms)

    dipole = get_l(new_alms, 1)

    remove_l(new_alms, 1)

    for i in prange(n):

        new_dipole = np.copy(dipole)

        randomly_rotate(new_dipole)

        simulations[i] = new_alms + new_dipole

    return simulations


#
# GREAT CIRCLE GENERATION FUNCTIONS
#

def great_circle_from_seed_pixel(pixel, nside):
    """Generate the equatorial great circle from a given pixel."""
    pixel_size = hp.pixelfunc.nside2resol(nside)

    return np.intersect1d(hp.query_disc(nside,
                                        hp.pixelfunc.pix2vec(nside, pixel),
                                        np.pi / 2. + pixel_size),
                          hp.query_disc(nside,
                          tuple(-np.array(hp.pixelfunc.pix2vec(nside, pixel))),
                          (np.pi / 2. + pixel_size)),
                          assume_unique=True)


def generate_great_circles(nside=NSIDE, dtheta_inc=DTHETA_INC):
    """
    Generate a set of great circles.

    First, uniformly generates pixels on the sky. Then, for each pixel,
    generate the unique great circle that would be the equator if that pixel
    was a pole. Then, convert the jagged array of great circles so they can be
    treated as a tensor and vectorized by numba; also embeds the true length of
    each great circle so it can be used to avoid unnecessary calculation later.
    """
    npix = int(4. * np.pi / (dtheta_inc / 2) ** 2)

    seed_pixels = []

    for i in range(npix):
        phi = np.random.uniform(0., 2 * np.pi)
        theta = np.arcsin(np.random.uniform(-1., 1.)) + np.pi / 2
        seed_pixels.append(hp.pixelfunc.ang2pix(nside, theta, phi))

    partial_function = partial(great_circle_from_seed_pixel, nside=nside)

    with Pool(processes=None) as pool:
        gc_pix_list = list(pool.map(partial_function, seed_pixels))

    max_length = max(list(map(len, gc_pix_list)))

    gc_pix_tensor = []

    for gc in gc_pix_list:
        gc_pix_tensor.append(np.pad(gc, (1, max_length - len(gc)),
                                    'constant',
                                    constant_values=(len(gc) + 1, 999999)))

    return np.array(gc_pix_tensor)


#
# STATISTICS FUNCTIONS
#

@guvectorize([(int64[:], float64[:], float64[:])], '(m),(n)->()',
             target='parallel', nopython=True)
def gc_vars(gc_pix, my_map, res):
    """Get the variances of great circles from a single map."""
    res[0] = np.var(my_map[gc_pix[1:gc_pix[0]]])


@jit(parallel=True)
def multi_gc_vars(gc_pix, alms, nside=NSIDE):
    """Get the variances of great circles from multiple maps in alm form."""
    vars_sims = np.zeros([alms.shape[0], gc_pix.shape[0]])

    for i in prange(alms.shape[0]):
        vars_sims[i] = gc_vars(gc_pix, hp.sphtfunc.alm2map(alms[i], nside,
                                                           verbose=False))

    return vars_sims


@guvectorize([(int64[:], float64[:], float64[:])], '(m),(n)->()',
             target='parallel', nopython=True)
def gc_means(gc_pix, my_map, res):
    """Get the means of great circles from a single map."""
    res[0] = np.mean(my_map[gc_pix[1:gc_pix[0]]])


@jit(parallel=True)
def multi_gc_means(gc_pix, alms, nside=NSIDE):
    """Get the means of great circles from multiple maps in alm form."""
    vars_sims = np.zeros([alms.shape[0], gc_pix.shape[0]])

    for i in prange(alms.shape[0]):
        vars_sims[i] = gc_means(gc_pix, hp.sphtfunc.alm2map(alms[i], nside,
                                                            verbose=False))

    return vars_sims


#
# MINIMIZATION FUNCTIONS
#

@jit(parallel=True)
def generate_orientation_minimizations(alms, gc_pix, nside=NSIDE):
    """Orientation-minimize each map in a given set of alms."""
    new_alms = np.copy(alms)

    for i in prange(new_alms.shape[0]):

        dipole = get_l(new_alms[i], 1)

        remove_l(new_alms[i], 1)

        res = differential_evolution(orientation_minimization_evaluator,
                                     [(0, 2*np.pi), (0, np.pi), (0, 2*np.pi)],
                                     args=(new_alms[i], dipole, gc_pix, nside),
                                     strategy='best1bin',
                                     workers=-1,
                                     popsize=3)

        hp.rotate_alm(dipole, res.x[0], res.x[1], res.x[2])

        new_alms[i] += dipole

    return new_alms


@jit(parallel=True)
def generate_complete_minimizations(alms, gc_pix, nside=NSIDE):
    """Completely minimize each map in a given set of alms."""
    new_alms = np.copy(alms)
    dipole_index = hp.Alm.getidx(hp.sphtfunc.Alm.getlmax(alms.shape[1]), 1, 0)

    for i in prange(new_alms.shape[0]):

        remove_l(new_alms[i], 1)

        dipole = np.zeros_like(new_alms[i])

        res = differential_evolution(complete_minimization_evaluator,
                                     [(0, 100), (0, np.pi), (0, 2*np.pi)],
                                     args=(new_alms[i], dipole_index, gc_pix,
                                           nside),
                                     strategy='best1bin',
                                     workers=-1,
                                     popsize=3)

        dipole[dipole_index] = res.x[0]

        hp.rotate_alm(dipole, 0, res.x[1], res.x[2])

        new_alms[i] += dipole

    return new_alms


def orientation_minimization_evaluator(angles, alms, dipole, gc_pix, nside):
    """Find variance of great circle variances, given a dipole orientation."""
    new_dipole = np.copy(dipole)

    hp.rotate_alm(new_dipole, angles[0], angles[1], angles[2])

    return np.var(gc_vars(gc_pix, hp.sphtfunc.alm2map(alms + new_dipole, nside,
                          verbose=False)))


def complete_minimization_evaluator(dipole, alms, index, gc_pix, nside):
    """Find variance of great circle variances, given a dipole."""
    new_dipole = np.zeros_like(alms)

    new_dipole[index] = dipole[0]

    hp.rotate_alm(new_dipole, 0, dipole[1], dipole[2])

    return np.var(gc_vars(gc_pix, hp.sphtfunc.alm2map(alms + new_dipole, nside,
                          verbose=False)))


def get_l(alms, L):
    """Return the specified multipole moment from the given set of alms."""
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    isolated_l = np.zeros_like(alms)

    for m in range(L + 1):

        index = hp.Alm.getidx(l_max, L, m)

        isolated_l[index] = alms[index]

    return isolated_l


def remove_l(alms, L):
    """Remove the specified multipole moment from the given set of alms."""
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    for m in range(L + 1):
        alms[hp.Alm.getidx(l_max, L, m)] = 0


def randomly_rotate(alms, random_psi=True):
    """Perform a random rotation on the given set of alms."""
    hp.rotate_alm(alms, random_psi * np.random.uniform(0., 2 * np.pi),
                  np.arcsin(np.random.uniform(- 1., 1.)) + np.pi / 2,
                  np.random.uniform(0., 2 * np.pi))
