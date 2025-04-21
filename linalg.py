import numpy as np

''' This file is a wrapper to the numpy.linalg.qr and numpy.linalg.svd
'''


def qr(data):
    """ QR decomposition method
    """
    return np.linalg.qr(data)


def lq(data):
    """ LQ decomposition method,
    """
    qat, rat = np.linalg.qr(data.T)
    la, qa   = rat.T, qat.T
    return la, qa


def svd(data, truncation=0):
    """ SVD decomposition method,
        param truncation : int, if truncation > 0,
                           truncation u, s, v with
                           max weight of s
    """
    u, s, v = np.linalg.svd(data, full_matrices=False)
    if truncation > 0:
        idx = s.argsort( )[::-1]
        u = u[:, idx]
        s = s[idx]
        v = v[idx, :]
        u = u[:, :truncation]
        s = s[:truncation]
        v = v[:truncation, :]
    return u, s, v
