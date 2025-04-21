import numpy as np
from collections import namedtuple
import linalg as lg

Leg = namedtuple('Leg', ['dimension', 'label'])


class TensorNode:
    """ single tensor node
        the basic unit of tensor network
    """

    def __init__(self, *args, **kargs):
        """ input param can be data, legs or omit
            data        : array type, data of tensor node
            legs        : list of 'Leg' object, the order of legs
                          must match with data
        """
        if len(kargs.keys( )):
            self.data = kargs['data']
            self.legs = kargs['legs']

    def gen_random_data(self, legs):
        """ if you omit the parameters of constructor,
            you can use this method to set legs and
            generate random data.
            param legs : list of 'Leg' object
        """
        self.legs = tuple(legs)
        if '_legs' in dir(self):
            assert self.total_dims(legs) == self.total_dims(self.legs)
        data = np.random.randn(*self.dims)
        self.data = data

    def copy(self):
        """ return copy of self
        """
        data = np.copy(self.data)
        legs = self.legs
        return TensorNode(data=data, legs=legs)

    def total_dims(self, legs=None):
        """ calculate the total dimension of input legs
            return value : int, total dimension
        """
        if legs is None:
            legs = self.legs
        dims = 1
        for i in legs:
            dims *= i.dimension
        return dims

    def rename(self, old_name, new_name):
        """ find the label whose name is old_name
            and rename it to new_name
        """
        idx = [i.label for i in self.legs].index(old_name)
        dim = self.legs[idx].dimension
        self._legs[idx] = Leg(dim, new_name)

    def _getdata(self):
        """ get data, and reshape it into self.dims
        """
        return self._data.reshape(self.dims)

    def _setdata(self, data):
        """ set data
            check the size of new data matches with
            old data and legs
        """
        data = np.asarray(data)
        if '_data' in dir(self):
            assert np.size(self._data) == np.size(data)
        if '_legs' in dir(self):
            tot_dim = 1
            for i in self.dims:
                tot_dim *= i
            assert np.size(data) == tot_dim
        self._data = data.reshape(-1, )

    data = property(_getdata, _setdata)

    def _getlegs(self):
        """ get legs
        """
        return self._legs

    def _setlegs(self, legs):
        """ set legs
            check the product of new leg's dimension
            matches with data and old legs
        """
        for l in legs:
            assert isinstance(l, Leg)
        if '_data' in dir(self):
            tot_dim = 1
            dims = [i.dimension for i in legs]
            for i in dims:
                tot_dim *= i
            assert np.size(self._data) == tot_dim
        if '_legs' in dir(self):
            tot_dim1 = 1
            tot_dim2 = 1
            dims1 = [i.dimension for i in legs]
            dims2 = [i.dimension for i in self._legs]
            for i in dims1:
                tot_dim1 *= i
            for i in dims2:
                tot_dim2 *= i
            assert tot_dim1 == tot_dim2

        # verify the label difference between legs
        n = len(set(i.label for i in legs))
        if n != len(legs):
            raise NameError("label name must be different")
        self._legs = list(legs)

    legs = property(_getlegs, _setlegs)

    @property
    def dims(self):
        """ get the dimension of legs
        """
        return tuple(i.dimension for i in self.legs)

    @property
    def labels(self):
        """ get the label of legs
        """
        return tuple(i.label for i in self.legs)

    def transpose(self, func=None):
        """ return transpose of self
            if func is not None,
            new tensor node's legs will given
            by func(old_leg_name)
        """
        legs = []
        for l in self.legs:
            if func is not None:
                legs.append(func(l))
            else:
                legs.append(l)
        data = np.conjugate(np.copy(self.data))
        return TensorNode(data=data, legs=legs)

    @property
    def T(self):
        """ return transpose of self
        """
        return self.transpose( )

    def _get_cform(self):
        """ return canonical form
            return value : str, 'L': left canonical
                                'R': right canonical
                                'O': others
        """
        if '_cform' in dir(self):
            cform = self._cform
        else:
            return 'O'
        return cform

    def _set_cform(self, s):
        """ set canonical form
            param s : str, 'L': left canonical
                           'R': right canonical
                           'O': others
        """
        if s in ['L', 'R', 'O']:
            self._cform = s
        else:
            raise TypeError(
                "node type does not support setting canonical form")

    cform = property(_get_cform, _set_cform)

    def __mul__(self, other):
        """ multiply self with another node.
            a * b equivalent to a.__nul__(b).
            param other  : TensorNode, the node multiply to self
            return value : TensorNode, multiply result
        """
        legs1 = list(self.legs)
        legs2 = list(other.legs)
        common_label = list(set(self.labels).intersection(other.labels))
        assert len(common_label) > 0
        idx1 = [self.labels.index(l) for l in common_label]
        idx2 = [other.labels.index(l) for l in common_label]
        data = np.tensordot(self.data, other.data, axes=(idx1, idx2))
        idx1.sort(reverse=True)
        idx2.sort(reverse=True)
        [legs1.pop(i) for i in idx1]
        [legs2.pop(i) for i in idx2]
        legs = legs1+legs2
        return TensorNode(data=data, legs=legs)

    def __lshift__(self, other):
        """ connect self with another node or chain.
            a << b equivalent to a.__lshift__(b).
            param other  : TensorNode or TensorChain,
                           the node or chain connect with self
            return value : TensorChain, connect result
        """
        if isinstance(other, TensorNode):
            return TensorChain(self, other)
        elif isinstance(other, TensorChain):
            return TensorChain(self, *other)
        else:
            raise TypeError('right object must be TensorNode or TensorChain')


class TensorChain(list):
    """ list of tensor node
        can be used to represent MPS and MPO
    """

    def __init__(self, *args):
        """ connect all input objects to create TensorChain
            input parameters must be TensorNode
        """
        super( ).__init__(args)

    def __lshift__(self, other):
        """ connect self with another node or chain.
            a << b equivalent to a.__lshift__(b).
            param other  : TensorNode or TensorChain,
                           the node or chain connect with self
            return value : TensorChain, connect result
        """
        if isinstance(other, TensorNode):
            return TensorChain(*self, other)
        elif isinstance(other, TensorChain):
            return self+other
        else:
            raise TypeError('right object must be TensorNode or TensorChain')

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
        if i is None:
            for i in range(len(self)-1, 0, -1):
                self.right_canonical(i)
        else:
            if i < 0:
                i = i+len(self)
            assert 0 < i < len(self)
            t1, t2 = self[i-1], self[i]
            t1, t2 = right_canonical(t1, t2)
            self[i-1], self[i] = t1, t2

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
        if i is None:
            for i in range(len(self)-1):
                self.left_canonical(i)
        else:
            if i < 0:
                i = i+len(self)
            assert 0 <= i < len(self)-1
            t1, t2 = self[i], self[i+1]
            t1, t2 = left_canonical(t1, t2)
            self[i], self[i+1] = t1, t2

    def transpose(self, func=None):
        """ return transpose of self
            if func is not None,
            new tensor chain's legs will given
            by func(old_leg_name)
        """
        return TensorChain(*(i.transpose(func) for i in self))

    @property
    def T(self):
        """ return transpose of self
        """
        return self.transpose( )

    @property
    def simplify(self):
        """ simplify self to single node
            return value : TensorNode
        """
        assert len(self)
        T0 = self[0]
        for T in self[1:]:
            T0 = T0*T
        return T0

    @property
    def data(self):
        """ return data of simplified self
        """
        return self.simplify.data

    @property
    def legs(self):
        """ return legs of simplified self
        """
        return self.simplify.legs


def left_canonical(t1, t2):
    """ use QR algorithm, transform t1 to left canonical
        the adjoint tensor will absorbed in t2
        param t1 : TensorNode, will be transform to
                   left canonical
        param t2 : TensorNode, Tensor node connect with t1
        return value : (Q, R) both are Tensor node. Q and R
                       are transformed t1 and t2.
    """
    assert isinstance(t1, TensorNode)
    assert isinstance(t2, TensorNode)
    assert len(t1.legs) in (2, 3)
    common_label = list(set(t1.labels).intersection(t2.labels))
    assert len(common_label) == 1
    label = common_label[0]
    assert t1.labels.index(label) == len(t1.legs)-1
    index2 = t2.labels.index(label)
    leg1 = t2.legs[index2]
    t2.rename(label, 'temp')
    leg2 = t2.legs[index2]

    data1 = t1.data
    if len(t1.legs) == 3:
        a, b, c = np.shape(data1)
        data1 = data1.reshape(a*b, c)
    Q, R = lg.qr(data1)
    Q = TensorNode(data=Q, legs=t1.legs)
    R = TensorNode(data=R, legs=[leg1, leg2])
    R = R*t2
    Q.cform = 'L'
    return Q, R


def right_canonical(t1, t2):
    """ use LQ algorithm, transform t2 to right canonical
        the adjoint tensor will absorbed in t1
        param t1 : TensorNode, Tensor node connect with t2
        param t2 : TensorNode, will be transform to
                   right canonical
        return value : (L, Q) both are Tensor node. L and Q
                       are transformed t1 and t2.
    """
    assert isinstance(t1, TensorNode)
    assert isinstance(t2, TensorNode)
    assert len(t2.legs) in (2, 3)
    common_label = list(set(t1.labels).intersection(t2.labels))
    assert len(common_label) == 1
    label = common_label[0]
    assert t2.labels.index(label) == 0
    index1 = t1.labels.index(label)
    leg2 = t1.legs[index1]
    t1.rename(label, 'temp')
    leg1 = t1.legs[index1]

    data2 = t2.data
    if len(t2.legs) == 3:
        a, b, c = np.shape(data2)
        data2 = data2.reshape(a, b*c)
    L, Q = lg.lq(data2)
    L = TensorNode(data=L, legs=[leg1, leg2])
    Q = TensorNode(data=Q, legs=t2.legs)
    L = t1*L
    Q.cform = 'R'
    return L, Q


def gen_random_node(legs):
    """ generate random tensor node
        with input legs
    """
    t = TensorNode( )
    t.gen_random_data(legs)
    return t
