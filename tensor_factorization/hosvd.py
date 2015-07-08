#!/usr/bin/env python
# -*- coding: ascii -*-
from __future__ import division
"""Higher order singular value decomposition routines

as introduced in:
    Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,
    'A multilinear singular value decomposition',
    SIAM J. Matrix Anal. Appl. 21 (4), 2000, 1253-1278

implemented by Jiahao Chen <jiahao@mit.edu>, 2010-06-11

Disclaimer: this code may or may not work.
"""
from numpy.linalg import inv
__author__ = 'Jiahao Chen <jiahao@mit.edu>'
__copyright__ = 'Copyright (c) 2010 Jiahao Chen'
__license__ = 'Public domain'

try:
    import numpy as np
except ImportError:
    print "Error: HOSVD requires numpy"
    raise ImportError



def unfold(A,n):
    """Computes the unfolded matrix representation of a tensor

    Parameters
    ----------

    A : ndarray, shape (M_1, M_2, ..., M_N)

    n : (integer) axis along which to perform unfolding,
                  starting from 1 for the first dimension

    Returns
    -------

    Au : ndarray, shape (M_n, M_(n+1)*M_(n+2)*...*M_N*M_1*M_2*...*M_(n-1))
         The unfolded tensor as a matrix

    Raises
    ------
    ValueError
        if A is not an ndarray

    LinAlgError
        if axis n is not in the range 1:N

    Notes
    -----
    As defined in Definition 1 of:

        Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,
        "A multilinear singular value decomposition",
        SIAM J. Matrix Anal. Appl. 21 (4), 2000, 1253-1278
    """

    if type(A) != type(np.zeros((1))):
        print "Error: Function designed to work with numpy ndarrays"
        raise ValueError

    if not (1 <= n <= A.ndim):
        print "Error: axis %d not in range 1:%d" % (n, A.ndim)
        raise np.linalg.LinAlgError

    s = A.shape

    m = 1
    for i in range(len(s)):
        m *= s[i]
    m //= s[n-1]

    #The unfolded matrix has shape (s[n-1],m)
    Au = np.zeros((s[n-1],m))

    index = [0]*len(s)

    for i in range(s[n-1]):
        index[n-1] = i
        for j in range(m):
            Au[i,j] = A[tuple(index)]

            #increment (n-1)th index first
            index[n-2] += 1

            #carry over: exploit python's automatic looparound of addressing!
            for k in range(n-2,n-1-len(s),-1):
                if index[k] == s[k]:
                    index[k-1] += 1
                    index[k] = 0

    return Au



def fold(Au, n, s):
    """Reconstructs a tensor given its unfolded matrix representation

    Parameters
    ----------

    Au : ndarray, shape (M_n, M_(n+1)*M_(n+2)*...*M_N*M_1*M_2*...*M_(n-1))
         The unfolded matrix representation of a tensor

    n : (integer) axis along which to perform unfolding,
                  starting from 1 for the first dimension

    s : (tuple of integers of length N) desired shape of resulting tensor

    Returns
    -------
    A : ndarray, shape (M_1, M_2, ..., M_N)

    Raises
    ------
    ValueError
        if A is not an ndarray

    LinAlgError
        if axis n is not in the range 1:N

    Notes
    -----
    Defined as the natural inverse of the unfolding operation as defined in Definition 1 of:

        Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,
        "A multilinear singular value decomposition",
        SIAM J. Matrix Anal. Appl. 21 (4), 2000, 1253-1278
    """

    m = 1
    for i in range(len(s)):
        m *= s[i]
    m //= s[n-1]

    #check for shape compatibility
    if Au.shape != (s[n-1], m):
        print "Wrong shape: need", (s[n-1], m), "but have instead", Au.shape
        raise np.linalg.LinAlgError

    A = np.zeros(s)

    index = [0]*len(s)

    for i in range(s[n-1]):
        index[n-1] = i
        for j in range(m):
            A[tuple(index)] = Au[i,j]

            #increment (n-1)th index first
            index[n-2] += 1

            #carry over: exploit python's automatic looparound of addressing!
            for k in range(n-2,n-1-len(s),-1):
                if index[k] == s[k]:
                    index[k-1] += 1
                    index[k] = 0

    return A


def HOSVD(A, threshold=1.0):
    """Computes the higher order singular value decomposition of a tensor

    Parameters
    ----------

    A : ndarray, shape (M_1, M_2, ..., M_N)

    Returns
    -------
    U : list of N matrices, with the nth matrix having shape (M_n, M_n)
        The n-mode left singular matrices U^(n), n=1:N

    S : ndarray, shape (M_1, M_2, ..., M_N)
        The core tensor

    D : list of N lists, with the nth list having length M_n
        The n-mode singular values D^(n), n=1:N

    Raises
    ------
    ValueError
        if A is not an ndarray

    LinAlgError
        if axis n is not in the range 1:N

    Notes
    -----
    Returns the quantities in Equation 22 of:

        Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,
        "A multilinear singular value decomposition",
        SIAM J. Matrix Anal. Appl. 21 (4), 2000, 1253-1278
    """

    Transforms = []
    Transforms2 = []
    NModeSingularValues = []

    #--- Compute the SVD of each possible unfolding
    for i in range(len(A.shape)):
        U, D, V = np.linalg.svd(unfold(A, i+1))
        Transforms.append(np.asmatrix(U))
        # U2 = U.copy()
        # U2[:, 2] = 0
        # print "U,U2: ", U, U2
        # Transforms2.append(np.asmatrix(U2))
        NModeSingularValues.append(D)

    param = []
    for j in range(len(A.shape)):
        value_list = list(NModeSingularValues[j])
        # print "value_list:", value_list
        length = len(value_list)
        value_sum = 0
        for value in value_list:
            value_sum += value
        index = length
        temp = 0
        while index > 1:
            temp += value_list[index-1]
            # print "temp:", temp
            if temp > value_sum * (1.0 - threshold):
                break
            index -= 1

        param.append(index)

    for k in range(len(A.shape)):
        U2 = Transforms[k].copy()
        # print "col: ", Transforms[k].shape[1]
        if param[k] < Transforms[k].shape[1]:
            for col in range(param[k], Transforms[k].shape[1]):
                U2[:, col] = 0
        Transforms2.append(np.asmatrix(U2))

    # print "Transforms:", Transforms
    # print "Transforms2:", Transforms2

    # print "param: ", param
    #--- Compute the unfolded core tensor
    axis = 1 #An arbitrary choice, really
    Aun = unfold(A, axis)

    #--- Computes right hand side transformation matrix
    B = np.ones((1,))
    # print "B:", type(B)
    # print A.ndim
    for i in range(axis-A.ndim, axis-1):
        # print "i:", i
        B = np.kron(B, Transforms2[i])

    #--- Compute the unfolded core tensor along the chosen axis
    # S(n) =U ?A(n)? U ?U ?????U ?U ?U ?????U . (p. 12  ??23)
    # print "shape1: ", Transforms2[axis-1].transpose().conj().shape
    # print "shape2: ", Aun.shape
    # print "shape3: ", B.shape
    Sun = Transforms2[axis-1].transpose().conj() * Aun * B

    # print Sun.shape
    S = fold(Sun, axis, A.shape)

    return Transforms2, S, NModeSingularValues


def recon(tensor, matrix, mode):
    Asize = np.asarray(tensor.shape)

    Asize[mode-1] = matrix.shape[0]

    An = unfold(tensor, mode)
    T = fold(np.dot(matrix, An), mode, Asize)
    return T


def reconstruct(S, U):
    axis = 1
    Sun = unfold(S, axis)

    B = np.ones((1,))
    for i in range(axis-S.ndim, axis-1):
        B = np.kron(B, U[i])
    Aun = U[axis-1] * Sun * (B.transpose().conj())

    A = fold(Aun, axis, S.shape)

    return A


def frobenius_norm(A):
    Aun = unfold(A, 1)
    #print "Aun: ", Aun
    a, b = Aun.shape
    sum = 0
    for i in range(0, a):
        for j in range(0, b):
            sum += Aun[i, j] ** 2

    return sum ** 0.5


if __name__ == '__main__':
    print
    print "Higher order singular value decomposition routines"
    print
    print "as introduced in:"
    print "    Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,"
    print "    'A multilinear singular value decomposition',"
    print "    SIAM J. Matrix Anal. Appl. 21 (4), 2000, 1253-1278"

    print
    print "Here are some worked examples from the paper."

    print
    print
    print "Example 1 from the paper (p. 1256)"

    A = np.zeros((3,2,3))

    A[0,0,0]=A[0,0,1]=A[1,0,0]=1
    A[1,0,1]=-1
    A[1,0,2]=A[2,0,0]=A[2,0,2]=A[0,1,0]=A[0,1,1]=A[1,1,0]=2
    A[1,1,1]=-2
    A[1,1,2]=A[2,1,0]=A[2,1,2]=4
    #other elements implied zero

    #test: compute unfold(A,1)
    print
    print "The input tensor is:"
    print A
    print
    print "Its unfolding along the first axis is:"
    print unfold(A,1)

    """
    print
    print
    print "Example 2 from the paper (p. 1257)""

    A = np.zeros((2,2,2))
    A[0,0,0] = A[1,1,0] = A[0,0,1] = 1
    #other elements implied zero
    """

    """
    print
    print
    print "Example 3 from the paper (p. 1257)""

    A = np.zeros((2,2,2))
    A[1,0,0] = A[0,1,0] = A[0,0,1] = 1
    #other elements implied zero
    """
    print
    print
    print "Example 4 from the paper (pp. 1264-5)"
    A = np.zeros((3,3,3))

    A[:,0,:] = np.asmatrix([[0.9073, 0.8924, 2.1488],
                            [0.7158, -0.4898, 0.3054],
                            [-0.3698, 2.4288, 2.3753]]).transpose()

    A[:,1,:] = np.asmatrix([[1.7842, 1.7753, 4.2495],
                            [1.6970, -1.5077, 0.3207],
                            [0.0151, 4.0337, 4.7146]]).transpose()

    A[:,2,:] = np.asmatrix([[2.1236, -0.6631, 1.8260],
                            [-0.0740, 1.9103, 2.1335],
                            [1.4429, -1.7495,-0.2716]]).transpose()

    print "The input tensor has matrix unfolding along axis 1:"
    print unfold(A, 1)
    print

    U, S, D = HOSVD(A, 1.0)

    print "The left n-mode singular matrices are:"
    print U[0]
    print
    print U[1]
    print
    print U[2]
    print

    print "The core tensor has matrix unfolding along axis 1:"
    print unfold(S, 1)
    print

    print "The n-mode singular values are:"
    print list(D[0])
    print D[1]
    print D[2]

    s1 = unfold(S, 1)
    print "core tensor unfold: ", s1
    print "orthogonal: ", s1[1].dot(s1[2])

    # print np.array([1, 2]).dot(np.array([2, 3]))

    A2 = reconstruct(S, U)
    print "reconstruct tensor: ", A2

    print frobenius_norm(A-A2)