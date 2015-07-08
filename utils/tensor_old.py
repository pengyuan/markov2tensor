#!/usr/bin/env python
# coding: UTF-8
from numpy.linalg import LinAlgError

from scipy.linalg import svd, pinv
import numpy as np


def flatten(T, n):
    """D = FLATTEN(T, N) makes a matrix out of a tensor.
    Such that the fibers along dimension N are aligned along the
    columms of D.

    SYNOPSIS:
        T - a tensor
        N - The mode in which to flatten T into.

    AUTHOR Larsson Omberg lom@larssono.com
    DATE   05-June-2009"""

    n=n-1
    if n>T.ndim | n<=0:
        raise ValueError('n has to be between 1 and the number of dimensions of T')

    order=[n,]
    order.extend(range(n+1, T.ndim))
    order.extend(range(0,n))

    nrows=T.shape[n]
    ncols=np.prod(T.shape)/nrows

    return np.reshape(np.transpose(T, order),(nrows,ncols), order='FORTRAN')


def hosvd2(T, saveSpace=False):
    """HOSVD    N-mode SVD decomposition of N-way tensor
    (Z, Un, Sn, Vn) = HOSVD(T) decomposes the N-way tensor D into N
    orthogonal matrices stored in Un and a core tensor Z.  Also
    returns the n Tucker1 results in Sn and Vn if saveSpace=False (default)

    Author: Larsson Omberg <lom@larssono.com>
    Date:   11-NOV-2007, 28-February-2008, 5-June-2009, 4-Nov-2010"""
    #Find orthogonal matrices
    Un=[]; Vn=[]; Sn=[]
    for n in range(T.ndim):
        Tn = flatten(T, n+1);  #FIX
        if Tn.shape[1] < Tn.shape[0]:
            [U,S,V] = svd(Tn,0);
            V=V.T
        else:
            [V,S,U] = svd(Tn.T,0);
            U=U.T
        Un.append(U);
        if not saveSpace:
            Vn.append(V);
            Sn.append(S);
    Z=T.copy()
    for n in range(len(Un)):
        Z = nmodmult(Z, Un[n].T, n+1)
    for i in range(len(Un)):
        new_T = nmodmult(Z, Un[n], n+1)



    if not saveSpace:
        return [new_T, Z, Un, Sn, Vn]
    return [Z, Un]


def unflatten(T, sizes, n):
    """D = UNFLATTEN(T, SIZES, N) remakes a tensor out of a matrix.
    Such that the mode N flattening of the tensor will return the
    matrix T.

    SYNOPSIS:
       T - a matrix of size mxn
       SIZES - size of the output matrix prod(sizes) must equal mxn
       N - The mode in which to unflatten t into.

    AUTHOR Larsson Omberg lom@larssono.com
    DATE   21-January-2005, 5-june-2009"""

    if np.prod(sizes) != np.prod(T.shape) or T.ndim !=2:
        raise ValueError('matrix and output tensor must have same number of elements')
    if n>len(sizes) | n<=0 :
        raise ValueError('n has to be between 1 and the number of dimensions of T')

    order=[n-1,]
    order.extend(range(n, len(sizes)))
    order.extend(range(n-1))

    #ndx = [order ones(1,ndims(b)-length(order))];
    #ndx(ndx) = 1:max(length(order),ndims(b));  % Inverse permutation order

    ndx = np.arange(len(order))
    ndx[order] = np.arange(len(order))  # Inverse permutation order

    d=np.reshape(T,sizes[order], order='FORTRAN');
    return np.transpose(d, ndx);  #np.transpose(...)


def nmodmult(A, B, n):
    """NMODMULT  peforms the n-mode multiplication of a tensor and matrix
    T = NMODMULT(A, B, N)
      A is a tensor of order < N
      B is a matrix to be multiplied
      N is the order of the multiplication
      return:  tensor product of A x_N B"""
    Asize = np.asarray(A.shape);

    Asize[n-1] = B.shape[0];

    An=flatten(A,n)
    T = unflatten(np.dot(B,An), Asize, n);
    return T


def gsvd(A,B):
    """Computes the generalized singular value decomposition
              U1,U2,V,S1,S2 = GSVD(A,B)
       such that A=U1 * S1 * V'
                 B=U2 * S2 * V'
                 S1'*S1 + S2'*S2 = I
       using the method described by Paige and Saunders in 1981
    Arguments:
    - `A`: a matrix of dimensions m x n
    - `B`: a matrix of dimensions p x n

    Returns: unitary matrices U1 and U2, a square matrix V, and
      nonnegative diagonal matrices S1 and S2.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    m,n = A.shape
    p,n1 = B.shape
    if n1!=n:
        raise LinAlgError('The number of columns in the two matrices A and B don\'t match')
    #First Step of algorithm
    P, R0, Q = svd(np.concatenate([A,B]))
    k = sum(R0>1e-12)
    R0=np.diag(R0)
    #Second Step part a
    P11 = P[:m,:k]
    U1, SA, W1=svd(P11)
    S1=np.zeros_like(P11)
    np.fill_diagonal(S1, SA)
    if S1.shape[0] > len(SA):  #Workaround for bug #1953 in fill_diagonal
        S1[len(SA):, :] = 0
    #Second Step part b
    P21 = P[m:m+p,:k]
    VB, SB, W=svd(P21)
    kr = min(p, k)
    S2 = np.zeros((p,k))
    S2[p-kr:, k-kr:] =   np.diag(SB[kr::-1])
    U2=reduce(np.dot, (P21, W1.T, pinv(S2)))

    Z=np.zeros((k,n-k));
    V=np.dot(np.hstack((np.dot(W1, R0[:k,:k]), Z)), Q).T

    return U1, U2, S1, S2, V


def test():
    T=np.reshape(np.arange(1,121).T, (3,4,5,2), order='FORTRAN')
    Z,Un,Sn,Vn = hosvd(T)
    x=nmodmult(Z, Un[0],1)
    x=nmodmult(x, Un[1],2)
    x=nmodmult(x, Un[2],3)
    x=nmodmult(x, Un[3],4)
    assert np.all(x-T<1e-12)

    A=[[1,    6,   11],
       [2,    7,   12],
       [3,    8,   13],
       [4,    9,   14],
       [5,   10,   15]]
    B=[[8,    1,    6],
       [3,    5,    7],
       [4,    9,    2]]
    U1, U2, S1, S2, V = gsvd(A,B)
    assert (np.all(reduce(np.dot, (U1,S1,V.T))-A < 1e-8) and np.all(reduce(np.dot, (U2,S2,V.T))-B < 1e-8))
    A=[[1,      4,     7,    10,    13],
       [2,      5,     8,    11,    14],
       [3,      6,     9,    12,    15]]
    B=[[17,    24,     1,     8,    15],
       [23,     5,     7,    14,    16],
       [ 4,     6,    13,    20,    22],
       [10,    12,    19,    21,     3]]

    U1, U2, S1, S2, V = gsvd(A,B)
    assert (np.all(reduce(np.dot, (U1,S1,V.T))-A < 1e-8) and np.all(reduce(np.dot, (U2,S2,V.T))-B < 1e-8))


if __name__ == '__main__':
    test()