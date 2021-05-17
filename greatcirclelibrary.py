"""A module to help analyze great circles in the CMB sky."""

import numpy as np
import healpy as hp
import camb
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation as R
from numba import guvectorize, int64, float64, prange, jit

NSIDE = 256
N_GC = 20000
N_P = 2000
PARS = camb.CAMBparams(min_l=1)
PARS.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, mnu=0.06, omk=0.001,
                   tau=0.054)
MAX_INF_DIPOLE_AMP = 100


@jit(parallel=True)
def preprocess_maps(paths, new_l_max, nside=NSIDE):
    """Preprocess maps of the full CMB sky.

    Parameters
    ----------
    paths : np.ndarray
        Paths to full CMB maps in .FITS format.
    new_l_max : int
        Maximum ell of multipole moments to extract from the full CMB maps.
    nside : int
        NSIDE of the maps when the extractions occur.

    Returns
    -------
    np.ndarray
        Returns the specified multipole moments of the full CMB maps, excluding
        their respective monopole and dipole moments.

    """
    new_alms = np.zeros([len(paths), hp.sphtfunc.Alm.getsize(new_l_max)],
                        dtype='complex')

    for i in prange(len(paths)):

        dgraded_map = hp.pixelfunc.ud_grade(hp.read_map(paths[i]), nside)

        old_alms = hp.sphtfunc.map2alm(dgraded_map)

        old_l_max = hp.sphtfunc.Alm.getlmax(old_alms.shape[0])

        for ell in range(2, new_l_max + 1):

            for m in range(ell + 1):

                old_index = hp.Alm.getidx(old_l_max, ell, m)

                new_index = hp.Alm.getidx(new_l_max, ell, m)

                new_alms[i][new_index] = old_alms[old_index]

    return new_alms


@jit(parallel=True)
def generate_same_cl_sims(alms, n):
    """Generate same cl spectrum simulations.

    Parameters
    ----------
    alms : np.ndarray
        Alms from which the same cl spectrum simulations are generated.
    n : int
        Number of simulations to generate.

    Returns
    -------
    np.ndarray
        Returns n same cl spectrum simulations.

    """
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    cl_vals = hp.alm2cl(alms)

    simulations = np.zeros([n, alms.shape[0]], dtype='complex')

    for i in prange(n):

        simulations[i] = hp.sphtfunc.synalm(cl_vals, lmax=l_max, verbose=False)

        for ell in np.flatnonzero(cl_vals):

            cl_sum = 0

            for m in range(ell + 1):

                cl_sum += abs(simulations[i][hp.Alm.getidx(l_max, ell, m)])**2

            scaling_factor = np.sqrt(cl_vals[ell] * (1 + 2 * ell)/(2 * cl_sum))

            for m in range(ell + 1):

                simulations[i][hp.Alm.getidx(l_max, ell, m)] *= scaling_factor

            simulations[i][hp.Alm.getidx(l_max, ell, 0)] *= np.sqrt(2)

    return simulations


@jit(parallel=True)
def generate_standard_dipole_sims(alms, n, pars=PARS):
    """Generate simulations with dipoles from the standard model of cosmology.

    Parameters
    ----------
    alms : np.ndarray
        Alms which the standard model dipoles are added to.
    n : int
        Number of simulations to generate.
    pars : camb.model.CAMBparams
        Parameters for the CAMB angular power spectrum generation.

    Returns
    -------
    np.ndarray
        Returns n simulations that have had a standard model dipole added to
        them.

    """
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    dipole_cl = np.zeros(l_max + 1)
    dipole_cl[1] = np.pi * powers['total'][1, 0]

    new_alms = remove_l(alms, 1)

    simulations = np.zeros([n, alms.shape[0]], dtype='complex')

    for i in prange(n):

        simulations[i] = new_alms + hp.sphtfunc.synalm(dipole_cl, lmax=l_max,
                                                       verbose=False)

    return simulations


@jit(parallel=True)
def generate_gcs(n_gc=N_GC, n_p=N_P, nside=NSIDE):
    """Generate a set of great circles.

    Parameters
    ----------
    n_gc : int
        Number of great circles to be generated.
    n_p : int
        Number of points to be sampled from each great circle.
    nside : int
        NSIDE of the map on which the great circles will be valid.

    Returns
    -------
    np.ndarray
        Returns an array containing great circles.
    """
    phi = np.random.uniform(0., 2 * np.pi, (n_gc, 1))
    theta = np.arcsin(np.random.uniform(np.sin(- np.pi / 2), np.sin(np.pi/2),
                                        (n_gc, 1))) + np.pi / 2
    rotation_1 = R.from_rotvec(np.pad(theta, [(0, 0), (1, 1)]))
    rotation_2 = R.from_rotvec(np.pad(phi, [(0, 0), (2, 0)]))
    random_rotation = rotation_2 * rotation_1

    circ_angs = np.linspace(0, 2 * np.pi, n_p)
    circ_coords = np.stack((np.cos(circ_angs), np.sin(circ_angs),
                            np.zeros_like(circ_angs)), axis=-1)

    gcs = np.empty([n_gc, n_p], dtype=int)

    for i in prange(n_gc):
        gc = random_rotation[i].apply(circ_coords)
        gcs[i] = hp.vec2pix(nside, gc[:, 0], gc[:, 1], gc[:, 2])

    return gcs


@guvectorize([(int64[:], float64[:], float64[:])], '(m),(n)->()',
             target='parallel', nopython=True)
def gc_vars(gcs, my_map, res):
    """Get the biased sample variances of great circles from a single map.

    Parameters
    ----------
    gcs : np.ndarray
        Great circles whose biased sample variances are calculated.
    my_map : np.ndarray
        Map from which the biased sample variances of the great circles
        are calculated.
    res : np.ndarray
        Returns the biased sample variances of the great circles.

    """
    res[0] = np.var(my_map[gcs])


@jit(parallel=True)
def multi_gc_vars(gcs, alms, nside=NSIDE):
    """Get the variance of great circle variances from multiple maps.

    Parameters
    ----------
    gcs : np.ndarray
        Great circles whose unbiased sample variances are calculated.
    alms : np.ndarray
        Set of maps in alm form from which the unbiased sample variances of the
        great circles are calculated.
    nside : int
        NSIDE of the maps when the unbiased sample variances of the great
        circles are calculated.

    Returns
    -------
    np.ndarray
        Returns the variance of great circle variances from multiple maps.

    """
    vars_sims = np.zeros(alms.shape[0])

    for i in prange(alms.shape[0]):
        vars_sims[i] = np.var(gc_vars(gcs, hp.sphtfunc.alm2map(alms[i], nside,
                                                               verbose=False)),
                              ddof=1)

    n = gcs.shape[1]

    return ((n / (n - 1)) ** 2) * vars_sims


def correlation_function(cl_vals):
    """Calculate the angular correlation function from a power spectrum.

    Parameters
    ----------
    cl_vals : np.ndarray
        Cls with which the angular correlation function is calculated.

    Returns
    -------
    function
        Returns the angular correlation function calculated from the given cls.

    """
    cl_vals_scaled = np.copy(cl_vals)

    for ell in range(cl_vals_scaled.shape[0]):
        cl_vals_scaled[ell] *= ((2. * ell + 1.) / (4. * np.pi))

    def C(theta):
        return np.polynomial.legendre.legval(np.cos(theta), cl_vals_scaled)

    return C


@jit(parallel=True)
def get_inf_versions(gcs, alms, nside=NSIDE):
    """Get the inferred versions of multiple maps.

    Parameters
    ----------
    gcs : np.ndarray
        Great circles whose biased sample variances are used to perform the
        calculations.
    alms : np.ndarray
        Set of maps in alm form whose inferred versions are calculated.
    nside : int
        NSIDE of the maps when the biased sample variances of the great circles
        are calculated.

    Returns
    -------
    np.ndarray
        Returns the respective inferred versions of the maps in alm form.

    """
    new_alms = np.copy(alms)
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[1])
    dipole_index = hp.Alm.getidx(l_max, 1, 0)

    for i in prange(new_alms.shape[0]):

        new_alms[i] = remove_l(new_alms[i], 1)

        dipole = np.zeros_like(new_alms[i])

        res = differential_evolution(inf_dipole_evaluator,
                                     [(0, MAX_INF_DIPOLE_AMP), (0, np.pi),
                                      (0, 2 * np.pi)],
                                     args=(gcs, new_alms[i], dipole_index,
                                           nside),
                                     workers=-1,
                                     popsize=3)

        dipole[dipole_index] = res.x[0]

        hp.rotate_alm(dipole, 0, res.x[1], res.x[2])

        new_alms[i] += dipole

    return new_alms


def inf_dipole_evaluator(dipole, gcs, alms, index, nside):
    """Find the variance of great circle variances, given a dipole.

    Parameters
    ----------
    dipole : np.ndarray
        Array containing the dipole amplitude and orientation.
    gcs : np.ndarray
        Great circles whose biased sample variances are calculated.
    alms : np.ndarray
        Alms used to generate the map that the dipole is added to.
    index : int
        Index of m = 0 for the dipole.
    nside : int
        NSIDE of the map when the biased sample variances of the great circles
        are calculated.

    Returns
    -------
    float
        Returns the biased sample variance of the biased sample variances of
        the great circles.

    """
    new_dipole = np.zeros_like(alms)

    new_dipole[index] = dipole[0]

    hp.rotate_alm(new_dipole, 0, dipole[1], dipole[2])

    return np.var(gc_vars(gcs, hp.sphtfunc.alm2map(alms + new_dipole, nside,
                                                   verbose=False)))


@jit(parallel=True)
def get_axes_of_max_sect(alms, ell):
    """Get the axes of maximum sectorality for a single ell from multiple maps.

    Parameters
    ----------
    alms : np.ndarray
        Set of maps in alm form from which the multipole moments are taken.
    ell : int
        Ell of the multipole moment whose axis of maximum sectorality is
        calculated.

    Returns
    -------
    np.ndarray
        Returns the locations of the respective axes of maximum sectorality for
        the specified multipole moment from each map.

    """
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[1])
    a_l_l_index = hp.Alm.getidx(l_max, ell, ell)

    locs = np.zeros([alms.shape[0], 3])

    for i in range(alms.shape[0]):

        isolated_moment = get_l(alms[i], ell)

        res = differential_evolution(axis_of_max_sect_evaluator,
                                     [(0, 2 * np.pi), (0, np.pi / 2)],
                                     args=(isolated_moment, a_l_l_index),
                                     workers=-1,
                                     popsize=15)

        ams_rot = R.from_euler('zyz', [0, -res.x[1], -res.x[0]])

        locs[i] = ams_rot.apply([0, 0, -1])

    return locs


def axis_of_max_sect_evaluator(loc, isolated_moment, a_l_l_index):
    """Find the sectorality of a multipole, given a coordinate system rotation.

    Parameters
    ----------
    loc : np.ndarray
        Array containing the rotation to the new coordinate system in which the
        sectorality is calculated.
    isolated_moment : np.ndarray
        The given multipole moment in alm form.
    a_l_l_index : int
        Index of a_l_l for the given multipole moment.

    Returns
    -------
    float
        Returns the reciprocal of the magnitude of the a_l_l coefficient in
        the new coordinate system.

    """
    new_moment = np.copy(isolated_moment)

    hp.rotate_alm(new_moment, loc[0], loc[1], 0)

    return 1 / abs(new_moment[a_l_l_index])


def get_nat_rot(alms):
    """Find a rotation to the natural coordinate system of the alms.

    Parameters
    ----------
    alms : np.ndarray
        Alms whose natural coordinate system rotation is found.

    Returns
    -------
    np.ndarray
        Returns a vector containing Euler angles that describe an extrinsic
        z-y-z rotation to the natural coordinate system.

    """
    rotation_copy = np.copy(alms)

    temp_map = hp.sphtfunc.alm2map(alms, NSIDE)
    dipole_ang = hp.pixelfunc.vec2ang(hp.pixelfunc.fit_dipole(temp_map)[1])
    nat_rotation = [-dipole_ang[1][0], -dipole_ang[0][0], 0]

    hp.rotate_alm(rotation_copy, *nat_rotation)

    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])
    a_3_3 = rotation_copy[hp.Alm.getidx(l_max, 3, 3)]
    nat_rotation[2] = np.arctan2(np.imag(a_3_3), np.real(a_3_3)) / 3

    return nat_rotation


def rot_to_nat_coords(alms):
    """Rotate the alms to their natural coordinate system.

    Parameters
    ----------
    alms : np.ndarray
        Alms to rotate to natural coordinates.

    Returns
    -------
    np.ndarray
        Returns the alms in their natural coordinate system.

    """
    new_alms = np.copy(alms)

    hp.rotate_alm(new_alms, *get_nat_rot(alms))

    return new_alms


def get_l(alms, ell):
    """Get the specified multipole moment from the alms.

    Parameters
    ----------
    alms : np.ndarray
        Alms from which the specified multipole moment is returned.
    ell : int
        Multipole moment to return.

    Returns
    -------
    np.ndarray
        Returns the specified multipole moment.

    """
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    isolated_l = np.zeros_like(alms)

    for m in range(ell + 1):

        index = hp.Alm.getidx(l_max, ell, m)

        isolated_l[index] = alms[index]

    return isolated_l


def remove_l(alms, ell):
    """Remove the specified multipole moment from the alms.

    Parameters
    ----------
    alms : np.ndarray
        Alms from which the specified multipole moment is removed.
    ell : int
        Multipole moment to remove.

    Returns
    -------
    np.ndarray
        Return the alms with the specified multipole moment removed.

    """
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    new_alms = np.copy(alms)

    for m in range(ell + 1):
        new_alms[hp.Alm.getidx(l_max, ell, m)] = 0

    return new_alms
