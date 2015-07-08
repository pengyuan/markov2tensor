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
from mobility import matrix_sn_nn
from tensor_old import nmodmult

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
        print "value_list:", value_list
        length = len(value_list)
        value_sum = 0
        for value in value_list:
            value_sum += value
        index = length
        temp = 0
        while index > 1:
            temp += value_list[index-1]
            print "temp:", temp
            if temp > value_sum * (1.0 - threshold):
                break
            index -= 1

        param.append(index)

    for k in range(len(A.shape)):
        U2 = Transforms[k].copy()
        print "col: ", Transforms[k].shape[1]
        if param[k] < Transforms[k].shape[1]:
            for col in range(param[k], Transforms[k].shape[1]):
                U2[:, col] = 0
        Transforms2.append(np.asmatrix(U2))

    print "Transforms:", Transforms
    print "Transforms2:", Transforms2

    print "param: ", param
    #--- Compute the unfolded core tensor
    axis = 1 #An arbitrary choice, really
    Aun = unfold(A, axis)

    #--- Computes right hand side transformation matrix
    B = np.ones((1,))
    print "B:", type(B)
    print A.ndim
    for i in range(axis-A.ndim, axis-1):
        print "i:", i
        B = np.kron(B, Transforms2[i])

    #--- Compute the unfolded core tensor along the chosen axis
    # S(n) =U ?A(n)? U ?U ?????U ?U ?U ?????U . (p. 12  ??23)
    print "shape1: ", Transforms2[axis-1].transpose().conj().shape
    print "shape2: ", Aun.shape
    print "shape3: ", B.shape
    Sun = Transforms2[axis-1].transpose().conj() * Aun * B

    print Sun.shape
    S = fold(Sun, axis, A.shape)

    return Transforms2, S, NModeSingularValues


def none_negative_hosvd(tensor, threshold=1.0):
    # mode 1
    shape1 = tensor.shape
    matrix_mode_one = unfold(tensor, 1)
    u, s, v = np.linalg.svd(matrix_mode_one, full_matrices=True)
    print u.shape, s.shape, v.shape
    singular_matrix = np.zeros(matrix_mode_one.shape)
    l = s.shape[0]
    singular_matrix[:l, :l] = np.diag(s)
    print "is all close: ", np.allclose(matrix_mode_one, u.dot(singular_matrix).dot(v))
    result_u, sigma, param_matrix = matrix_sn_nn(u)

    print "result: ", result_u
    print "sigma: ", sigma
    print "param_matrix: ", param_matrix


    remain_matrix = inv(param_matrix).dot(inv(sigma)).dot(singular_matrix).dot(v)

    shape2 = (result_u.shape[1], shape1[1], shape1[2])
    tensor2 = fold(remain_matrix, 1, shape2)
    print "tensor2: ", tensor2

    # mode 2
    matrix_mode_two = unfold(tensor2, 2)
    u, s, v = np.linalg.svd(matrix_mode_two, full_matrices=True)
    print u.shape, s.shape, v.shape
    singular_matrix = np.zeros(matrix_mode_two.shape)
    l = s.shape[0]
    singular_matrix[:l, :l] = np.diag(s)
    result_v, sigma, param_matrix = matrix_sn_nn(u)

    remain_matrix = inv(param_matrix).dot(inv(sigma)).dot(singular_matrix).dot(v)

    shape3 = (result_u.shape[1], result_v.shape[1], shape1[2])
    tensor3 = fold(remain_matrix, 2, shape3)
    print "tensor3: ", tensor3

    # mode 3
    matrix_mode_three = unfold(tensor3, 3)
    u, s, v = np.linalg.svd(matrix_mode_three, full_matrices=True)
    print u.shape, s.shape, v.shape
    singular_matrix = np.zeros(matrix_mode_three.shape)
    l = s.shape[0]
    singular_matrix[:l, :l] = np.diag(s)
    result_w, sigma, param_matrix = matrix_sn_nn(u)

    remain_matrix = inv(param_matrix).dot(inv(sigma)).dot(singular_matrix).dot(v)

    shape4 = (result_u.shape[1], result_v.shape[1], result_w.shape[1])
    r = fold(remain_matrix, 3, shape4)
    # print "tensor4: ", r

    print "u: ", result_u
    print "v: ", result_v
    print "w: ", result_w
    print "r: ", r

    # print r.dot(result_u).dot(result_v).dot(result_w)

    x = recon(r, result_u, 1)
    x = recon(x, result_v, 2)
    x = recon(x, result_w, 3)
    print "reconstruct tensor: ", x
    print "tensor: ", tensor

    return x, result_u, result_v, result_w, r


def none_negative_svd(matrix, threshold=1.0):
    # mode 1
    shape1 = matrix.shape
    matrix_mode_one = unfold(matrix, 1)
    u, s, v = np.linalg.svd(matrix_mode_one, full_matrices=True)
    print u.shape, s.shape, v.shape
    singular_matrix = np.zeros(matrix_mode_one.shape)
    l = s.shape[0]
    singular_matrix[:l, :l] = np.diag(s)
    print "is all close: ", np.allclose(matrix_mode_one, u.dot(singular_matrix).dot(v))
    result_u, sigma, param_matrix = matrix_sn_nn(u)

    print "result: ", result_u
    print "sigma: ", sigma
    print "param_matrix: ", param_matrix


    remain_matrix = inv(param_matrix).dot(inv(sigma)).dot(singular_matrix).dot(v)

    shape2 = (result_u.shape[1], shape1[1])
    tensor2 = fold(remain_matrix, 1, shape2)
    print "tensor2: ", tensor2

    # mode 2
    matrix_mode_two = unfold(tensor2, 2)
    u, s, v = np.linalg.svd(matrix_mode_two, full_matrices=True)
    print u.shape, s.shape, v.shape
    singular_matrix = np.zeros(matrix_mode_two.shape)
    l = s.shape[0]
    singular_matrix[:l, :l] = np.diag(s)
    result_v, sigma, param_matrix = matrix_sn_nn(u)

    remain_matrix = inv(param_matrix).dot(inv(sigma)).dot(singular_matrix).dot(v)

    shape3 = (result_u.shape[1], result_v.shape[1])
    r = fold(remain_matrix, 2, shape3)

    print "u: ", result_u
    print "v: ", result_v
    print "r: ", r

    return result_u, result_v, r


def none_negative_svd2(matrix, threshold=1.0):
    # mode 1
    shape1 = matrix.shape
    matrix_mode_one = unfold(matrix, 1)
    u, s, v = np.linalg.svd(matrix_mode_one, full_matrices=True)
    print u.shape, s.shape, v.shape
    singular_matrix = np.zeros(matrix_mode_one.shape)
    l = s.shape[0]
    singular_matrix[:l, :l] = np.diag(s)
    print "is all close: ", np.allclose(matrix_mode_one, u.dot(singular_matrix).dot(v))
    result_u, sigma, param_matrix = matrix_sn_nn(u)
    result_v, sigma2, param_matrix2 = matrix_sn_nn(v)

    print "result: ", result_u
    print "sigma: ", sigma
    print "param_matrix: ", param_matrix
    print "sigma2: ", sigma2
    print "param_matrix2: ", param_matrix2

    r = inv(param_matrix).dot(inv(sigma)).dot(singular_matrix).dot(inv(sigma2).transpose()).dot(inv(param_matrix2).transpose())

    return result_u, result_v, r


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

    aaa = matrix_sn_nn(U[1])
    print "before:", U[1]
    print "after:", aaa

    x, result_u, result_v, result_w, r = none_negative_hosvd(A)
    n_a = result_u.shape[1]
    n_b = result_v.shape[1]
    n_c = result_w.shape[1]

    print A[2][1][0]
    print n_a, n_b, n_c, result_u.shape, result_v.shape, result_w.shape, r.shape, np.array(result_u)[2][2]
    B = np.zeros((3, 3, 3))
    for index_i in range(0, 3):
        for index_j in range(0, 3):
            for index_k in range(0, 3):
                for i in range(0, n_a):
                    for j in range(0, n_b):
                        for k in range(0, n_c):
                            B[index_i][index_j][index_k] = np.array(result_u)[index_i][i] * np.array(result_v)[index_j][j] * np.array(result_w)[index_k][k] * np.array(r)[i][j][k]

    print "B: ", B

    matrix = [[1, 2], [3, 4]]
    u, v, r = none_negative_svd(np.array(matrix))

    C = np.zeros((2, 2))
    n_a = u.shape[1]
    n_b = v.shape[1]
    for index_i in range(0, 2):
        for index_j in range(0, 2):
            for i in range(0, n_a):
                for j in range(0, n_b):
                    C[index_i][index_j] = np.array(u)[index_i][i] * np.array(v)[index_j][j] * np.array(r)[i][j]

    print "C: ", C

    matrix = [[1/2, 1/2], [1/3, 2/3], [1/5, 4/5]]
    u, v, r = none_negative_svd2(np.array(matrix))

    D = np.zeros((3, 2))
    n_a = u.shape[1]
    n_b = v.shape[1]
    print n_a, n_b
    for index_i in range(0, 3):
        for index_j in range(0, 2):
            for i in range(0, n_a):
                for j in range(0, n_b):
                    D[index_i][index_j] = np.array(u)[index_i][i] * np.array(v)[index_j][j] * np.array(r)[i][j]

    print "D: ", D