"""A module to help analyze great circles in the CMB sky."""

import numpy as np
import healpy as hp
import camb
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation as R
from numba import guvectorize, int64, float64, prange, jit

NSIDE = 256
N_GC = 5000
N_P = 2000
N_SIMS = 10000
PARS = camb.CAMBparams(min_l=1)
PARS.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, mnu=0.06, omk=0.001,
                   tau=0.054)
MAX_PREF_DIPOLE_AMP = 100


@jit(parallel=True)
def preprocess_maps(paths, new_l_max, nside=NSIDE):
    """Preprocess maps of the full CMB sky.

    Parameters
    ----------
    paths : np.ndarray
        Paths to full CMB maps in .FITS format.
    new_l_max : int
        Maximum L of multipole moments to extract from the full CMB maps.
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

        for L in range(2, new_l_max + 1):

            for m in range(L + 1):

                old_index = hp.Alm.getidx(old_l_max, L, m)

                new_index = hp.Alm.getidx(new_l_max, L, m)

                new_alms[i][new_index] = old_alms[old_index]

    return new_alms


@jit(parallel=True)
def generate_same_cl_sims(alms, n=N_SIMS):
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

        for L in np.flatnonzero(cl_vals):

            cl_sum = 0

            for m in range(L + 1):

                cl_sum += abs(simulations[i][hp.Alm.getidx(l_max, L, m)]) ** 2

            scaling_factor = np.sqrt(cl_vals[L] * (1 + 2 * L) / (2 * cl_sum))

            for m in range(L + 1):

                simulations[i][hp.Alm.getidx(l_max, L, m)] *= scaling_factor

            simulations[i][hp.Alm.getidx(l_max, L, 0)] *= np.sqrt(2)

    return simulations


@jit(parallel=True)
def generate_standard_dipole_sims(alms, n=N_SIMS, pars=PARS):
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
    dipole_cl[1] = powers['total'][1, 0]

    new_alms = remove_l(alms, 1)

    simulations = np.zeros([n, alms.shape[0]], dtype='complex')

    for i in prange(n):

        simulations[i] = new_alms + hp.sphtfunc.synalm(dipole_cl, lmax=l_max,
                                                       verbose=False)

    return simulations


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

    circ_angs = np.random.uniform(0, 2 * np.pi, (n_gc, n_p))
    circ_coords = np.stack((np.cos(circ_angs), np.sin(circ_angs),
                            np.zeros_like(circ_angs)), axis=-1)

    gcs = np.empty([n_gc, n_p], dtype=int)

    for i in prange(n_gc):
        gc = random_rotation[i].apply(circ_coords[i])
        gcs[i] = hp.vec2pix(nside, gc[:, 0], gc[:, 1], gc[:, 2])

    return gcs


@guvectorize([(int64[:], float64[:], float64[:])], '(m),(n)->()',
             target='parallel', nopython=True)
def gc_vars(gc_pix, my_map, res):
    """Get the biased sample variances of great circles from a single map.

    Parameters
    ----------
    gc_pix : np.ndarray
        Great circles whose biased sample variances are calculated.
    my_map : np.ndarray
        Map from which the biased sample variances of the great circles
        are calculated.
    res : np.ndarray
        Returns the biased sample variances of the great circles.

    """
    res[0] = np.var(my_map[gc_pix])


@jit(parallel=True)
def multi_gc_vars(gc_pix, alms, nside=NSIDE):
    """Get the unbiased sample variances of great circles from multiple maps.

    Parameters
    ----------
    gc_pix : np.ndarray
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
        Returns the unbiased sample variances of great circles from multiple
        maps.

    """
    vars_sims = np.zeros([alms.shape[0], gc_pix.shape[0]])

    for i in prange(alms.shape[0]):
        vars_sims[i] = gc_vars(gc_pix, hp.sphtfunc.alm2map(alms[i], nside,
                                                           verbose=False))

    n = gc_pix.shape[1]

    return (n / (n - 1)) * vars_sims


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
    for L in range(cl_vals.shape[0]):
        cl_vals[L] = cl_vals[L] * ((2. * L + 1.)/(4. * np.pi))

    def C(theta):
        return np.polynomial.legendre.legval(np.cos(theta), cl_vals)

    return C


@jit(parallel=True)
def get_pref_versions(alms, gc_pix, nside=NSIDE):
    """Get the preferred versions of multiple maps.

    Parameters
    ----------
    alms : np.ndarray
        Set of maps in alm form whose preferred versions are calculated.
    gc_pix : np.ndarray
        Great circles whose biased sample variances are used to perform the
        calculations.
    nside : int
        NSIDE of the maps when the biased sample variances of the great circles
        are calculated.

    Returns
    -------
    np.ndarray
        Returns the respective preferred versions of the maps in alm form.

    """
    new_alms = np.copy(alms)
    dipole_index = hp.Alm.getidx(hp.sphtfunc.Alm.getlmax(alms.shape[1]), 1, 0)

    for i in prange(new_alms.shape[0]):

        new_alms[i] = remove_l(new_alms[i], 1)

        dipole = np.zeros_like(new_alms[i])

        res = differential_evolution(pref_dipole_evaluator,
                                     [(0, MAX_PREF_DIPOLE_AMP), (0, np.pi),
                                      (0, 2 * np.pi)],
                                     args=(new_alms[i], dipole_index, gc_pix,
                                           nside),
                                     strategy='best1bin',
                                     workers=-1,
                                     popsize=3)

        dipole[dipole_index] = res.x[0]

        hp.rotate_alm(dipole, 0, res.x[1], res.x[2])

        new_alms[i] += dipole

    return new_alms


def pref_dipole_evaluator(dipole, alms, index, gc_pix, nside):
    """Find the variance of great circle variances, given a dipole.

    Parameters
    ----------
    dipole : np.ndarray
        Array containing the dipole amplitude and orientation.
    alms : np.ndarray
        Alms used to generate the map that the dipole is added to.
    index : int
        Index of m = 0 for the dipole.
    gc_pix : np.ndarray
        Great circles whose biased sample variances are calculated.
    nside : int
        NSIDE of the map when the biased sample variances of the great circles
        are calculated.

    Returns
    -------
    float
        The biased sample variance of the biased sample variances of the great
        circles.

    """
    new_dipole = np.zeros_like(alms)

    new_dipole[index] = dipole[0]

    hp.rotate_alm(new_dipole, 0, dipole[1], dipole[2])

    return np.var(gc_vars(gc_pix, hp.sphtfunc.alm2map(alms + new_dipole, nside,
                                                      verbose=False)))


def get_pref_rot(alms):
    """Find a rotation to the preferred coordinate system of the alms.

    Parameters
    ----------
    alms : np.ndarray
        Alms whose preferred coordinate system rotation is found.

    Returns
    -------
    np.ndarray
        Returns a vector containing Euler angles that describe an extrinsic
        z-y-z rotation to the preferred coordinate system.

    """
    rotation_copy = np.copy(alms)

    temp_map = hp.sphtfunc.alm2map(alms, NSIDE)
    dipole_ang = hp.pixelfunc.vec2ang(hp.pixelfunc.fit_dipole(temp_map)[1])
    pref_rotation = [-dipole_ang[1][0], -dipole_ang[0][0], 0]

    hp.rotate_alm(rotation_copy, *pref_rotation)

    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])
    m_3_3 = rotation_copy[hp.Alm.getidx(l_max, 3, 3)]
    pref_rotation[2] = np.arctan2(np.imag(m_3_3), np.real(m_3_3)) / 3

    return pref_rotation


def rot_to_pref_coords(alms):
    """Rotate the alms to their preferred coordinate system.

    Parameters
    ----------
    alms : np.ndarray
        Alms to rotate to preferred coordinates.

    Returns
    -------
    np.ndarray
        The alms in their preferred coordinate system.

    """
    new_alms = np.copy(alms)

    hp.rotate_alm(new_alms, *get_pref_rot(alms))

    return new_alms


def get_l(alms, L):
    """Get the specified multipole moment from the alms.

    Parameters
    ----------
    alms : np.ndarray
        Alms from which the specified multipole moment is returned.
    L : int
        Multipole moment to return.

    Returns
    -------
    np.ndarray
        Returns the specified multipole moment.

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
        Alms from which the specified multipole moment is removed.
    L : int
        Multipole moment to remove.

    Returns
    -------
    np.ndarray
        Return the alms with the specified multipole moment removed.

    """
    l_max = hp.sphtfunc.Alm.getlmax(alms.shape[0])

    new_alms = np.copy(alms)

    for m in range(L + 1):
        new_alms[hp.Alm.getidx(l_max, L, m)] = 0

    return new_alms
