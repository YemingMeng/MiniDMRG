import numpy as np
from tensor import Leg, TensorNode, TensorChain


class OperatorNode(TensorNode):
    """ single tensor node of basic MPO representation
    """

    def __init__(self, oper_mat):
        """ convert data to OperatorNode
            label of self.legs will select in 'up', 'down', 'left', 'right'
            param oper_mat : array_like, with shape (n, n, lhd, lhd)
                             or (n, 1, lhd, lhd) or (1, n, lhd, lhd)
            n, lhd         : virtual MPO dimension and local hilbert
                             dimension
        """
        n = max(len(oper_mat), len(oper_mat[0]))
        lhd, _ = np.shape(oper_mat[0][0])
        self.lhd = lhd
        self.data = np.asarray(oper_mat)
        if len(oper_mat) > 1 and len(oper_mat[0]) > 1:
            self.legs = [
                Leg(dimension=n, label='left'),
                Leg(dimension=n, label='right'),
                Leg(dimension=lhd, label='up'),
                Leg(dimension=lhd, label='down')
            ]
        elif len(oper_mat) > 1 and len(oper_mat[0]) == 1:
            self.legs = [
                Leg(dimension=n, label='left'),
                Leg(dimension=lhd, label='up'),
                Leg(dimension=lhd, label='down')
            ]
        elif len(oper_mat) == 1 and len(oper_mat[0]) > 1:
            self.legs = [
                Leg(dimension=n, label='right'),
                Leg(dimension=lhd, label='up'),
                Leg(dimension=lhd, label='down')
            ]
        else:
            raise RuntimeError('the shape of oper_mat is wrong')


class LocalHamiltonianChain(TensorChain):
    """ MPO tensor chain with nearest neighbor interaction
    """

    def __init__(self):
        """ initialize the object
        """
        self._operator = {}
        self._neighbor_interaction_number = 0
        self.nodes = []
        self._neighbor_interaction = []
        self._onsite_interaction = []
        self._oper_mat = []

    def define_operator(self, name, data):
        """ define operator
            param name : str, name of operator
            param data : array_like, matrix data of operator
        """
        if len(self._operator) == 0:
            self._lhd, _ = np.shape(data)
        self._operator[name] = data

    def print_operator(self):
        """ print all defined operators
        """
        if len(self._operator) > 0:
            for name, data in self._operator.items():
                print('Data of {}:'.format(name))
                print(data, end='\n\n')
        else:
            print('There are no defined operators.')

    def add_neighbor_interaction(self, oname1, oname2, strength):
        """ add neighbor interaction between oparator oname1 and operator
            oname2 with given strength.
            oname1   : str, name of operator 1
            oname2   : str, name of operator 2
            strength : float
        """
        self._neighbor_interaction_number += 1
        assert oname1 in self._operator.keys()
        assert oname2 in self._operator.keys()
        self._neighbor_interaction.append([oname1, oname2, float(strength)])

    def add_onsite_interaction(self, oname, strength):
        """ add neighbor interaction of oparator oname with given strength.
            oname    : str, operator name
            strength : float
        """
        assert oname in self._operator.keys()
        self._onsite_interaction.append([oname, float(strength)])

    def gen_data(self, L=10):
        """ generate MPO data
            param L  : MPO chain length
        """
        dim = self._neighbor_interaction_number+2
        self._oper_mat = [['' for i in range(dim)]for j in range(dim)]
        for i in range(dim):
            for j in range(dim):
                self._oper_mat[i][j] = np.zeros((self._lhd, self._lhd))

        self._oper_mat[0][0] = np.eye(self._lhd)
        self._oper_mat[-1][-1] = np.eye(self._lhd)
        for idx, (name1, name2, strength) in\
                enumerate(self._neighbor_interaction):
            self._oper_mat[idx+1][0] = self._operator[name1]
            self._oper_mat[-1][1+idx] = self._operator[name2]*strength
        for name, strength in self._onsite_interaction:
            self._oper_mat[-1][0] += self._operator[name]*strength

        o = OperatorNode([self._oper_mat[-1]])
        o.rename('right', 2*L+1)
        o.rename('down' , L)
        o.rename('up'   , -L)
        self.append(o)

        for i in range(L-2):
            o = OperatorNode(self._oper_mat)
            o.rename('left' , 2*L+i+1)
            o.rename('right', 2*L+i+2)
            o.rename('down' , L+i+1)
            o.rename('up'   , -L-i-1)
            self.append(o)
        o = OperatorNode([[i[0]] for i in self._oper_mat])
        o.rename('left', 2*L+L-1)
        o.rename('down', 2*L-1)
        o.rename('up'  , -2*L+1)
        self.append(o)
