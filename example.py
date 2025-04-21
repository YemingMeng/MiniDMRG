from tensor import *
from mps import random_tensor_chain
from mpo import *
from dmrg import Dmrg


def example1( ):
    """ basic tensor node and tensor chain usage
    """
    print('example 1:')
    legs1 = [Leg(dimension=2, label=0),
             Leg(dimension=2, label=1),
             Leg(dimension=4, label=2)]
    legs2 = [Leg(dimension=4, label=2),
             Leg(dimension=8, label=4)]
    t1  = gen_random_node(legs1)
    t2  = gen_random_node(legs2)
    t3  = t1*t2
    tc1 = t1 << t2
    tc2 = TensorChain(t1, t2)
    print('type of t1, t2, t3')
    print(type(t1))
    print(type(t2))

    print('type of tc1, tc2')
    print(type(t3))
    print(type(tc1))
    print(type(tc2))

    print('legs of t1 and t2')
    print(t1.legs)
    print(t1.legs)
    print('legs of tc1 and tc2')
    print(tc1.legs)
    print(tc2.legs)

    print('verify data consistency')
    print(np.allclose(t3.data, tc1.data))
    print(np.allclose(tc1.data, tc2.data))


def example2( ):
    """ basic MPS usage
        define local hilbert dimension, total site number
        and truncation dimension, generate random MPS
    """
    print('example 2:')
    lhd   = 2
    nsite = 20
    tc = random_tensor_chain(lhd, nsite, truncation=40)
    tc.right_canonical( )
    print([i.dims for i in tc])


def example3( ):
    """ basic MPO definition
    """
    print('example 3:')
    # define parameter
    h  = 1
    j  = 1
    jz = 1

    # define operator
    s0 = np.eye(2)
    sp = np.asarray([[0, 1],
                     [0, 0]])
    sm = np.asarray([[0, 0],
                     [1, 0]])
    sz = np.asarray([[1, 0],
                     [0, -1]])/2
    o = np.zeros((2, 2))
    M1 = [[h/2*(sp+sm), j/2*sm, j/2*sp, jz*sz, s0]]
    N1 = OperatorNode(M1)
    M2 = [[s0, o, o, o, o],
          [sp, o, o, o, o],
          [sm, o, o, o, o],
          [sz, o, o, o, o],
          [h/2*(sp+sm), j/2*sm, j/2*sp, jz*sz, s0]]
    N2 = OperatorNode(M2)
    M3 = [[s0],
          [sp],
          [sm],
          [sz],
          [h/2*(sp+sm)]]
    N3 = OperatorNode(M3)
    print(np.shape(N1.data))
    print(N1.legs)
    print(np.shape(N2.data))
    print(N2.legs)
    print(np.shape(N3.data))
    print(N3.legs)


def example4( ):
    """ another way to generate MPO
    """
    print('example 4:')
    # define parameter
    h  = 1
    j  = 1
    jz = 1

    # define operator
    sp = np.asarray([[0, 1],
                     [0, 0]])
    sm = np.asarray([[0, 0],
                     [1, 0]])
    sz = np.asarray([[1, 0],
                     [0, -1]])/2

    H = LocalHamiltonianChain( )
    H.define_operator('sp', sp)
    H.define_operator('sm', sm)
    H.define_operator('sz', sz)
    H.add_neighbor_interaction('sp', 'sm', j/2)
    H.add_neighbor_interaction('sm', 'sp', j/2)
    H.add_neighbor_interaction('sz', 'sz', jz)
    H.add_onsite_interaction('sp', 1/2*h)
    H.add_onsite_interaction('sm', 1/2*h)
    H.gen_data(L=20)

    print(np.shape(H[ 0].data))
    print(H[ 0].legs)
    print(np.shape(H[ 1].data))
    print(H[ 1].legs)
    print(np.shape(H[-1].data))
    print(H[-1].legs)


def example5( ):
    """ variational MPS algorithm (finite DMRG)
    """
    print('example 5:')
    # set parameter
    j  = 1
    jz = 1
    h  = 0
    L  = 20

    # define operator
    sp = np.asarray([[0, 1], [0, 0]])
    sm = np.asarray([[0, 0], [1, 0]])
    sz = np.asarray([[1/2, 0], [0, -1/2]])

    # generate MPO
    H = LocalHamiltonianChain( )
    H.define_operator('sp', sp)
    H.define_operator('sm', sm)
    H.define_operator('sz', sz)
    H.add_neighbor_interaction('sp', 'sm', j/2)
    H.add_neighbor_interaction('sm', 'sp', j/2)
    H.add_neighbor_interaction('sz', 'sz', jz)
    H.add_onsite_interaction('sp', 1/2*h)
    H.add_onsite_interaction('sm', 1/2*h)
    H.gen_data(L=L)

    # generate random MPS
    tc = random_tensor_chain(2, L, truncation=40)

    # dmrg progress
    d = Dmrg(tc, H, eps=1e-6, visual=True)
    d.start( )

    # print energy
    print('Energy per site = {:.12f}'.format(d.energy))


if __name__ == "__main__":
    example1( )
    print(  )
    example2(  )
    print(  )
    example3(  )
    print(  )
    example4(  )
    print(  )
    example5( )
