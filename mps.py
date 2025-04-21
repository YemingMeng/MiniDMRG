from tensor import Leg, TensorChain, gen_random_node


def random_tensor_chain(lhd, nsite, truncation=10000):
    """ generate random 1 dimension MPS with open boundary condition
        param lhd   : dimension of local hilbert space
        param nsite : total site number of chain
        truncation  : truncation dimension of MPS
    """
    hlabels = list(range(1, nsite))
    temp  = [lhd**i for i in hlabels]
    hdims = tuple(min(min(i), truncation)
                  for i in zip(temp, temp[::-1]))
    hlegs = [Leg(d, l) for d, l in zip(hdims, hlabels)]
    vlabels = list(range(nsite, nsite+nsite))
    vdims = [lhd for i in vlabels]
    vlegs = [Leg(d, l) for d, l in zip(vdims, vlabels)]

    Legs = [[vlegs[0], hlegs[0]]]
    for i in range(nsite-2):
        Legs.append([hlegs[i], vlegs[i+1], hlegs[i+1]])
    Legs.append([hlegs[-1], vlegs[-1]])
    return TensorChain(*(gen_random_node(l) for l in Legs))
