#!/usr/bin/env python
# -*- coding: ascii -*-

"""Higher order singular value decomposition routines

as introduced in:
    Lieven de Lathauwer, Bart de Moor, Joos Vandewalle,
    'A multilinear singular value decomposition',
    SIAM J. Matrix Anal. Appl. 21 (4), 2000, 1253-1278

implemented by Jiahao Chen <jiahao@mit.edu>, 2010-06-11

Disclaimer: this code may or may not work.
"""

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
    m /= s[n-1]

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
    m /= s[n-1]

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



def HOSVD(A):
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
    NModeSingularValues = []

    #--- Compute the SVD of each possible unfolding
    for i in range(len(A.shape)):
        U,D,V = np.linalg.svd(unfold(A,i+1))
        Transforms.append(np.asmatrix(U))
        NModeSingularValues.append(D)

    #--- Compute the unfolded core tensor
    axis = 1 #An arbitrary choice, really
    Aun = unfold(A,axis)

    #--- Computes right hand side transformation matrix
    B = np.ones((1,))
    for i in range(axis-A.ndim,axis-1):
        B = np.kron(B, Transforms[i])

    #--- Compute the unfolded core tensor along the chosen axis
    Sun = Transforms[axis-1].transpose().conj() * Aun * B

    S = fold(Sun, axis, A.shape)

    return Transforms, S, NModeSingularValues


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
    print unfold(A,1)
    print

    U, S, D = HOSVD(A)

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
    print D[0]
    print D[1]
    print D[2]
