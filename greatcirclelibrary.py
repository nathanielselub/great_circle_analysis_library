"""A module to help analyze great circles in the CMB sky."""

import numpy as np
import healpy as hp
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation as R
from numba import guvectorize, int64, float64, prange, jit, njit

NSIDE = 256
NPIX = hp.nside2npix(NSIDE)
GC_N = 4096
MASK_ANGLE = np.pi/8
DIPOLE_RANGE = (10, 100)
CORR_FUNC_RES = 1000
AXIS_OF_EVIL_ANGLE = [110 * np.pi / 180, 60 * np.pi / 180]


@jit(parallel=True)
def preprocess_maps(paths, new_l_max, nside=NSIDE, parity=''):
    """Preprocess maps of the full CMB sky.

    Parameters
    ----------
    paths : np.ndarray
        Paths to full CMB maps in .FITS format.
    new_l_max : int
        Maximum L of multipole moments to extract from the full CMB maps.
    nside : int
        NSIDE of the maps when the extractions occur.
    parity : str
        Specifies whether all, even, or odd multipole moments up to new_l_max
        should be extracted from the full CMB maps.

    Returns
    -------
    np.ndarray
        Returns the specified multipole moments of the full CMB maps, excluding
        their monopoles and kinetic dipoles.

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

        dgraded_map = hp.pixelfunc.ud_grade(hp.read_map(paths[i]), nside)

        old_alms = hp.sphtfunc.map2alm(dgraded_map)

        old_l_max = hp.sphtfunc.Alm.getlmax(old_alms.shape[0])

        for L in range(start, new_l_max + 1, step):
            for m in range(L + 1):

                old_index = hp.Alm.getidx(old_l_max, L, m)

                new_index = hp.Alm.getidx(new_l_max, L, m)

                new_alms[i][new_index] = old_alms[old_index]

    return new_alms


def shift_to_axis_of_evil(alms):
    """Rotate the alms from Galactic coordinates to Axis of Evil coordinates.

    Parameters
    ----------
    alms : np.ndarray
        The alms to be rotated to the Axis of Evil coordinate system, in which
        the Axis of Evil points in the positive z direction.

    """
    for i in range(alms.shape[0]):

        hp.rotate_alm(alms[i], AXIS_OF_EVIL_ANGLE[0], AXIS_OF_EVIL_ANGLE[1], 0)

#
# SIMULATION GENERATION FUNCTIONS
#


@jit(parallel=True)
def generate_random_reorientations(alms, n):
    """Generate random reorientation simulations from the alms.

    Parameters
    ----------
    alms : np.ndarray
        The alms from which the random reorientation simulations are generated.
    n : int
        The number of simulations generated.

    Returns
    -------
    np.ndarray
        The n random reorientation simulations.

    """
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
    """Generate same cl spectrum simulations from the alms.

    Parameters
    ----------
    alms : np.ndarray
        The alms from which the same cl spectrum simulations are generated.
    n : int
        The number of simulations generated.

    Returns
    -------
    np.ndarray
        The n same cl spectrum simulations.

    """
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
    """Generate simulations with cosmic variance from the alms.

    Parameters
    ----------
    alms : np.ndarray
        The alms from which the simulations with cosmic variance are generated.
    n : int
        The number of simulations generated.

    Returns
    -------
    np.ndarray
        The n simulations with cosmic variance.

    """
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    my_cls = hp.alm2cl(alms)

    simulations = np.zeros([n, alms.shape[0]], dtype='complex')

    for i in prange(n):

        simulations[i] = hp.sphtfunc.synalm(my_cls, lmax=l_max, verbose=False)

    return simulations


@jit(parallel=True)
def generate_random_dipole_simulations(alms, n, dipole_range=DIPOLE_RANGE):
    """Generate simulations where random dipoles replace the alms' dipole.

    Parameters
    ----------
    alms : np.ndarray
        The alms from which the simulations with random dipoles are generated.
    n : int
        The number of simulations generated.
    dipole_range : tuple
        The range of possible amplitude for each random dipole.

    Returns
    -------
    np.ndarray
        The n simulations with random dipoles.

    """
    simulations = np.zeros([n, alms.shape[0]], dtype='complex')

    new_alms = np.copy(alms)

    remove_l(new_alms, 1)

    dipole_index = hp.Alm.getidx(hp.sphtfunc.Alm.getlmax(alms.shape[0]), 1, 0)

    for i in prange(n):

        new_dipole = np.zeros_like(new_alms)

        new_dipole[dipole_index] = np.random.uniform(*dipole_range)

        randomly_rotate(new_dipole)

        simulations[i] = new_alms + new_dipole

    return simulations


@jit(parallel=True)
def generate_random_dipole_reorientation_simulations(alms, n):
    """Generate simulations with randomly reoriented dipoles.

    Parameters
    ----------
    alms : np.ndarray
        The alms from which the simulations with randomly reoriented dipoles
        are generated.
    n : int
        The number of simulations generated.

    Returns
    -------
    np.ndarray
        The n simulations with randomly reoriented dipoles.

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

def generate_great_circles(gc_n=GC_N, nside=NSIDE, mask_angle=MASK_ANGLE):
    """Generate a set of great circles.

    Parameters
    ----------
    gc_n : int
        The number of great circles to be generated.
    nside : int
        The NSIDE of the map on which the great circles will be valid.
    mask_angle : float
        The minimum angular distance between every great circle's axis and the
        z-axis pole.

    Returns
    -------
    np.ndarray
        A tensor containing the great circles.
    """
    npoint = 8 * nside

    phi = np.random.uniform(0., 2 * np.pi, (gc_n, 1))
    theta = np.arcsin(np.random.uniform(np.sin(- np.pi / 2 + mask_angle),
                                        np.sin(np.pi/2 - mask_angle),
                                        (gc_n, 1))) + np.pi / 2
    rotation_1 = R.from_rotvec(np.pad(theta, [(0, 0), (1, 1)]))
    rotation_2 = R.from_rotvec(np.pad(phi, [(0, 0), (2, 0)]))
    random_rotation = rotation_2 * rotation_1

    circ_angs = np.linspace(0, 2 * np.pi, npoint)
    circ_coords = np.pad(np.vstack((np.cos(circ_angs), np.sin(circ_angs))),
                         [(0, 1), (0, 0)]).T

    gc_list = np.empty([gc_n, npoint], dtype=int)

    for i in prange(gc_n):
        gc = random_rotation[i].apply(circ_coords)
        gc_list[i] = hp.vec2pix(nside, gc[:, 0], gc[:, 1], gc[:, 2])

    return gc_list


#
# STATISTICAL CALCULATION FUNCTIONS
#

@jit(fastmath=True)
def gc_power_spectra(gc_pix, alms, nside=NSIDE):
    """Get the power spectra of great circles from multiple maps.

    Parameters
    ----------
    gc_pix : np.ndarray
        The great circles whose power spectra are to be calculated.
    alms : np.ndarray
        The set of maps in alm form whose great circle power spectra are to be
        calculated.
    nside : int
        The NSIDE of the maps when the power spectra of the great circles are
        calculated.

    Returns
    -------
    np.ndarray
        The power spectra of great circles from multiple maps.

    """
    spec = np.empty([alms.shape[0], gc_pix.shape[0], 1 + gc_pix.shape[1] // 2])
    for i in prange(alms.shape[0]):
        current_map = hp.sphtfunc.alm2map(alms[i], nside, verbose=False)
        for j in prange(gc_pix.shape[0]):
            spec[i][j] = np.fft.rfft(current_map[gc_pix[j]])

    return np.abs(spec) ** 2


@guvectorize([(int64[:], float64[:], float64[:])], '(m),(n)->()',
             target='parallel', nopython=True)
def gc_vars(gc_pix, my_map, res):
    """Get the variances of great circles from a single map.

    Parameters
    ----------
    gc_pix : np.ndarray
        The great circles whose variances are calculated.
    my_map : np.ndarray
        The map whose great circle variances are calculated.
    res : np.ndarray
        The variances of the great circles.

    """
    res[0] = np.var(my_map[gc_pix])


@jit(parallel=True)
def multi_gc_vars(gc_pix, alms, nside=NSIDE):
    """Get the variances of great circles from multiple maps.

    Parameters
    ----------
    gc_pix : np.ndarray
        The great circles whose variances are to be calculated.
    alms : np.ndarray
        The set of maps in alm form whose great circle variances are to be
        calculated.
    nside : int
        The NSIDE of the maps when the variances of the great circles are
        calculated.

    Returns
    -------
    np.ndarray
        The variances of great circles from multiple maps.

    """
    vars_sims = np.zeros([alms.shape[0], gc_pix.shape[0]])

    for i in prange(alms.shape[0]):
        vars_sims[i] = gc_vars(gc_pix, hp.sphtfunc.alm2map(alms[i], nside,
                                                           verbose=False))

    return vars_sims


@guvectorize([(int64[:], float64[:], float64[:])], '(m),(n)->()',
             target='parallel', nopython=True)
def gc_means(gc_pix, my_map, res):
    """Get the means of great circles from a single map.

    Parameters
    ----------
    gc_pix : np.ndarray
        The great circles whose means are calculated.
    my_map : np.ndarray
        The map whose great circle means are calculated.
    res : np.ndarray
        The means of the great circles.

    """
    res[0] = np.mean(my_map[gc_pix])


@jit(parallel=True)
def multi_gc_means(gc_pix, alms, nside=NSIDE):
    """Get the means of great circles from multiple maps.

    Parameters
    ----------
    gc_pix : np.ndarray
        The great circles whose means are to be calculated.
    alms : np.ndarray
        The set of maps in alm form whose great circle means are to be
        calculated.
    nside : int
        The NSIDE of the maps when the means of the great circles are
        calculated.

    Returns
    -------
    np.ndarray
        The means of great circles from multiple maps.

    """
    vars_sims = np.zeros([alms.shape[0], gc_pix.shape[0]])

    for i in prange(alms.shape[0]):
        vars_sims[i] = gc_means(gc_pix, hp.sphtfunc.alm2map(alms[i], nside,
                                                            verbose=False))

    return vars_sims


def correlation_function(cls, res=CORR_FUNC_RES):
    """Get the means of great circles from multiple maps.

    Parameters
    ----------
    cls : np.ndarray
        The cls with which the correlation function is calculated.
    res : int
        The number of points at which the correlation function is calculated.

    Returns
    -------
    np.ndarray
        The values of the correlation function.

    """
    for L in range(cls.shape[0]):
        cls[L] = cls[L] * ((2. * L + 1.)/(4. * np.pi))

    return np.polynomial.legendre.legval(np.cos(np.linspace(0, np.pi, res)),
                                         cls)


#
# MINIMIZATION FUNCTIONS
#

@jit(parallel=True)
def generate_orientation_minimizations(alms, gc_pix, nside=NSIDE):
    """Orientation-minimize multiple maps.

    Parameters
    ----------
    alms : np.ndarray
        The set of maps in alm form to be orientation-minimized.
    gc_pix : np.ndarray
        The great circles whose variances are used to perform the
        minimizations.
    nside : int
        The NSIDE of the maps when the variances of the great circles are
        calculated.

    Returns
    -------
    np.ndarray
        The set of orientation-minimized maps in alm form.

    """
    new_alms = np.copy(alms)
    dipole_index = hp.Alm.getidx(hp.sphtfunc.Alm.getlmax(alms.shape[1]), 1, 0)

    for i in prange(new_alms.shape[0]):

        dipole = np.zeros_like(new_alms[i])

        dipole[dipole_index] = np.sqrt(3 * hp.alm2cl(new_alms[i])[1])

        remove_l(new_alms[i], 1)

        res = differential_evolution(orientation_minimization_evaluator,
                                     [(0, np.pi), (0, 2 * np.pi)],
                                     args=(new_alms[i], dipole, gc_pix, nside),
                                     strategy='best1bin',
                                     workers=-1,
                                     popsize=2)

        hp.rotate_alm(dipole, 0, res.x[0], res.x[1])

        new_alms[i] += dipole

    return new_alms


@jit(parallel=True)
def generate_complete_minimizations(alms, gc_pix, nside=NSIDE):
    """Completely minimize multiple maps.

    Parameters
    ----------
    alms : np.ndarray
        The set of maps in alm form to be completely minimized.
    gc_pix : np.ndarray
        The great circles whose variances are used to perform the
        minimizations.
    nside : int
        The NSIDE of the maps when the variances of the great circles are
        calculated.

    Returns
    -------
    np.ndarray
        The set of completely minimized maps in alm form.

    """
    new_alms = np.copy(alms)
    dipole_index = hp.Alm.getidx(hp.sphtfunc.Alm.getlmax(alms.shape[1]), 1, 0)

    for i in prange(new_alms.shape[0]):

        remove_l(new_alms[i], 1)

        dipole = np.zeros_like(new_alms[i])

        res = differential_evolution(complete_minimization_evaluator,
                                     [(0, 100), (0, np.pi), (0, 2 * np.pi)],
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
    """Find variance of great circle variances, given a dipole orientation.

    Parameters
    ----------
    angles : np.ndarray
        An array containing the angles describing the dipole orientation.
    alms : np.ndarray
        The alms used to generate the map that the dipole will be added to.
    dipole : np.ndarray
        The dipole to be oriented.
    gc_pix : np.ndarray
        The great circles whose variances are calculated.
    nside : int
        The NSIDE of the map when the variances of the great circles are
        calculated.

    Returns
    -------
    float
        The variance of great circle variances.

    """
    new_dipole = np.copy(dipole)

    hp.rotate_alm(new_dipole, 0, angles[0], angles[1])

    return np.var(gc_vars(gc_pix, hp.sphtfunc.alm2map(alms + new_dipole, nside,
                                                      verbose=False)))


def complete_minimization_evaluator(dipole, alms, index, gc_pix, nside):
    """Find the variance of great circle variances, given a dipole.

    Parameters
    ----------
    dipole : np.ndarray
        An array containing the dipole amplitude and orientation.
    alms : np.ndarray
        The alms used to generate the map that the dipole will be added to.
    index : int
        The index of m = 0 for the dipole.
    gc_pix : np.ndarray
        The great circles whose variances are calculated.
    nside : int
        The NSIDE of the map when the variances of the great circles are
        calculated.

    Returns
    -------
    float
        The variance of great circle variances.

    """
    new_dipole = np.zeros_like(alms)

    new_dipole[index] = dipole[0]

    hp.rotate_alm(new_dipole, 0, dipole[1], dipole[2])

    return np.var(gc_vars(gc_pix, hp.sphtfunc.alm2map(alms + new_dipole, nside,
                                                      verbose=False)))

#
# HELPER FUNCTIONS
#


def get_l(alms, L):
    """Return the specified multipole moment from the alms.

    Parameters
    ----------
    alms : np.ndarray
        The alms from which the specified multipole moment is to be returned.
    L : int
        The multipole moment to return.

    Returns
    -------
    np.ndarray
        The specified multipole moment.

    """
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    isolated_l = np.zeros_like(alms)

    for m in range(L + 1):

        index = hp.Alm.getidx(l_max, L, m)

        isolated_l[index] = alms[index]

    return isolated_l


def remove_l(alms, L):
    """Remove the specified multipole moment from the alms.

    Parameters
    ----------
    alms : np.ndarray
        The alms from which the specified multipole moment is to be removed.
    L : int
        The multipole moment to remove.

    """
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    for m in range(L + 1):
        alms[hp.Alm.getidx(l_max, L, m)] = 0


def randomly_rotate(alms):
    """Perform a random rotation on the alms.

    Parameters
    ----------
    alms : np.ndarray
        The alms upon which the random rotation is to be performed.

    """
    hp.rotate_alm(alms, np.random.uniform(0., 2 * np.pi),
                  np.arcsin(np.random.uniform(- 1., 1.)) + np.pi / 2,
                  np.random.uniform(0., 2 * np.pi))
