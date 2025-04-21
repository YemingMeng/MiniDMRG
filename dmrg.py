from scipy.sparse.linalg import eigsh

from tensor import *

""" 
Created on 19.09.02

@author: Meng Ye-Ming
"""


def inverse_label(leg):
    """ transfer label from positive to negative of input leg
    """
    dim, lab = leg
    return Leg(dim, -lab)


class Dmrg:
    """ variational MPS algorithm (finite DMRG)
    """

    def __init__(self, mps, mpo, eps=1e-8, visual=False):
        """ initialize object, set MPS, MPO
            param mps    : TensorChain, initial MPS
            param mpo    : TensorChain, MPO
            param eps    : float, energy convergence precision
            param visual : bool, if True, show text progress
        """
        self._mps = mps
        self._mpo = mpo
        self._visual = visual
        self._ltensor_cache = {}
        self._rtensor_cache = {}
        self._eps = eps
        self._step = 0

    def start(self, minSweep=0):
        """ start main DMRG program
            minSweep     : int, minimal sweep time
        """
        self.right_canonical()
        self._energy   = 1e100
        self._converge = False
        while (not self._converge) or (self._step < minSweep):
            self.right_sweep()
            self.left_sweep()
            self._step += 1

    def right_sweep(self):
        """ starting from site 1 to site N-1, sweep the lattice to right
        """
        e0 = self._energy

        # calculate the effective hamiltonian of first site
        ct = self._mpo[0]
        rt = self.rtensor(0)
        eh = ct*rt

        # calculate the ground state and assign to self._mps[0]
        data = np.transpose(eh.data, [0, 2, 1, 3])
        n = round(float(np.sqrt(np.size(data))))
        _, psi = eigsh(data.reshape(n, n), k=1, which='SA', maxiter=5000)
        self._set_mps_data(psi, 0)
        self.left_canonical(0)

        for i in range(self._L-2):
            # calculate the effective hamiltonian of other sites
            lt = self.ltensor(i+1)
            ct = self._mpo[i+1]
            rt = self.rtensor(i+1)
            eh = lt*ct*rt

            # calculate the ground state and assign to self._mps[i]
            data = np.transpose(eh.data, [0, 2, 4, 1, 3, 5])
            n = round(float(np.sqrt(np.size(data))))
            _, psi = eigsh(data.reshape(n, n), k=1, which='SA', maxiter=5000)
            self._set_mps_data(psi, i+1)
            self.left_canonical(i+1)

        # judging whether convergence or not by energy difference
        self._energy = _[0]/self._L
        if np.abs(self._energy-e0) < self._eps:
            self._converge = True

    def left_sweep(self):
        """ starting from site N to site 2, sweep the lattice to left
            simiar to right_sweep
        """
        e0 = self._energy

        lt = self.ltensor(-1)
        ct = self._mpo[-1]
        eh = lt*ct

        data = np.transpose(eh.data, [0, 2, 1, 3])
        n = round(float(np.sqrt(np.size(data))))
        _, psi = eigsh(data.reshape(n, n), k=1, which='SA', maxiter=5000)
        self._set_mps_data(psi, -1)
        self.right_canonical(-1)

        for i in range(self._L-2):
            lt = self.ltensor(-2-i)
            ct = self._mpo[-2-i]
            rt = self.rtensor(-2-i)
            eh = lt*ct*rt
            data = np.transpose(eh.data, [0, 2, 4, 1, 3, 5])
            n = round(float(np.sqrt(np.size(data))))
            _, psi = eigsh(data.reshape(n, n), k=1, which='SA', maxiter=5000)
            self._set_mps_data(psi, -2-i)
            self.right_canonical(-2-i)
        self._energy = _[0]/self._L
        if np.abs(self._energy-e0) < self._eps:
            self._converge = True

    def right_canonical(self, i=None):
        """ call function right_canonical,
            if i is not None, convert all sites
            to right canonical except the first site.
            otherwise, only convert site i.
            param i : int or None, the position of
                      site to be converted. the cform of
                      site i will be set to 'R', and the
                      cfom of site i-1 will be set to 'O'
        """
        self._mps.right_canonical(i=i)
        if i is not None:
            self._ltensor_cache.pop(i, None)
        if self._visual:
            print('sweep {}:'.format(self._step), ''.join(
                [i.cform for i in self._mps]), end='\n')

    def left_canonical(self, i=None):
        """ call function left_canonical,
            if i is not None, convert all sites
            to left canonical except the last site.
            otherwise, only convert site i.
            param i : int or None, the position of
                      site to be converted. the cform of
                      site i will be set to 'L', and the
                      cfom of site i+1 will be set to 'O'
        """
        self._mps.left_canonical(i=i)
        if i is not None:
            self._rtensor_cache.pop(i, None)
        if self._visual:
            print('sweep {}:'.format(self._step), ''.join(
                [i.cform for i in self._mps]), end='\n')

    @property
    def energy(self):
        """ return energy per site
        """
        return self._energy

    @property
    def _mpst(self):
        """ return transposed mps with negative label
        """
        return self._mps.transpose(inverse_label)

    @property
    def _L(self):
        """ total lattice site number
        """
        return len(self._mps)

    def ltensor(self, i):
        """ recursively calculate product of all tensors at the
            left of position i, and record cache in self._ltensor_cache
            param i : int
        """
        if i < 0:
            i = i+self._L
        assert 0 < i < self._L
        if i in self._ltensor_cache.keys():
            rst = self._ltensor_cache[i]
        else:
            if i == 1:
                rst = self._mpst[i-1]*self._mpo[i-1]*self._mps[i-1]
            else:
                l = self.ltensor(i-1)
                rst = l*self._mpst[i-1]*self._mpo[i-1]*self._mps[i-1]
            self._ltensor_cache[i] = rst
        return rst

    def rtensor(self, i):
        """ recursively calculate product of all tensors at the
            right of position i, and record cache in self._rtensor_cache
            param i : int
        """
        if i < 0:
            i = i+self._L
        assert 0 <= i < self._L-1
        if i in self._rtensor_cache.keys():
            rst = self._rtensor_cache[i]
        else:
            if i == self._L-2:
                rst = self._mpst[i+1]*self._mpo[i+1]*self._mps[i+1]
            else:
                r = self.rtensor(i+1)
                rst = self._mpst[i+1]*(self._mpo[i+1]*(self._mps[i+1]*r))
            self._rtensor_cache[i] = rst
        return rst

    def _set_mps_data(self, data, i):
        """ set data of tensor in position i
            and remove outdated ltensor_cache, rtensor_cache
            param i : int
        """
        if i < 0:
            i = i+self._L
        assert 0 <= i < self._L
        self._mps[i].data = data
        self._ltensor_cache.pop(i+1, None)
        self._rtensor_cache.pop(i-1, None)
