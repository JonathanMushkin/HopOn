# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:33:23 2019

@author: Jonathan Mushkin (Weizmann Institute of Science)

General unitlity fucntions.

"""

import numpy  as np
from numba import njit

@njit('float64[:](float64[:],float64[:])')
def my_cross(x,y):
    z = np.empty_like(x)
    z[0] = x[1]*y[2] - x[2]*y[1]
    z[1] = x[2]*y[0] - x[0]*y[2]
    z[2] = x[0]*y[1] - x[1]*y[0]
    return z

@njit('int32[:,:](int32)')
def create_triplets(N):
    """
    Create all possible triplets for an N body system. The triplet refer to a 
    pair + single configuration.
    This means that the triplets [1,2,3] and [2,1,3] are the same. 
    But the triplets [1,2,3] and [1,3,2] are different. 
    We create only unique triplets. [1,2,3] will be included, but [2,1,3] will 
    not be.
    
    INPUT: 
        N : int32, number of bodies 
    
    OUTPUT: 
        t : int32 2d array, of triplets. t[i,j] containt the index of the j-th
        body in the i-th triplet.
        
        triplets are arrange by ascending first index, ascending second index, 
        then ascending third index.      
    """
    triplets = np.zeros((int(N*(N-1)*(N-2)/2),3),dtype=np.int32)
    i=int(0)
    for n in range(N-1):
        for nn in range(n+1,N):
            for k in range(N):
                if (k!=n)& (k!=nn):
                    triplets[i] = [n,nn,k]
                    i+=1
                        
    return triplets


@njit('float64[:](float64[:,:],float64[:])')
def weighted_average(X,W):
    N = int(W.shape[0])
    Y = np.zeros(X.shape[1])
    
    for i in range(N):
        Y += X[i,:]*W[i]
    
    return Y/np.sum(W)
            
@njit('float64(float64[:],float64[:])')
def enod(x,y):
    """
    Compute the Eucleadian Norm Of the Difference (enod) between two 3d vectors
    """
    return ((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2 )**(0.5)

@njit('float64[:](float64[:,:])')
def pairwise_enod(R):
    """
    Compute all Eucleadina Norm of Differences between all possible 3D vectors 
    pairs in a vector array
    INPUTS: 
        R : N x 3 dimensional np.array-s of positions
    OUTPUTS: 
        D : 1d array of length N*(N-1)/2 
    
    D[i] contain the distnace between 2 members of the i-th pair
    pairs are ordered by ascending first index, then ascending second index
    meaning: (0,1), (0,2), (0,3), ... (1,2), (1,3),... (N-1,N)
    
    """

    N = int(R.shape[0]) # number of bodies    
    pair_index = 0
    D = np.zeros(int( N*(N-1)/2 ))
    for n in range(N-1):
        for nn in range(n+1,N):
             D[pair_index] = enod(R[n,:],R[nn,:])
             pair_index = pair_index+1
    return D

@njit('float64[:](float64[:,:],float64[:])')
def ps_enod(R,M):
    """
    Pair-Single Eucleadian Norm of Difference.
    Return the Eucleadian norm of difference between the weighted average (set 
    by vectors in R and weights in M) of each pair and the vector of a single 
    body. 
    For example: used to find the distance between a body and the center of 
    mass of a bianry.
    
    
    INPUTS: 
        R : N x 3 dimensional np.array-s of positions
        M : np.array of length N  of masses

            
    OUTPUTS: 
        dbs : 1d array of length N*(N-1)*(N-2)/2 of pair-signles distances.
        triplets are ordered by ascending first index (first pair member), 
        then ascending second index (second pair member),
        then ascending third index (single body). Meaning:
        ((0,1),2),((0,1),3),..., 
        ((0,2),1),((0,2),3),...,    
        ((1,2),0),((1,2),3),...
        ((N-1,N),0), ((N-1,N),1)...((N-1,N),N-2)
    """
    N = R.shape[0]
    dbs = np.zeros(int(N*(N-1)*(N-2)/2 )) # binary-single distance
    triplet_index = 0
    for n in range(N-1):
        for nn in range(n+1,N):
            for k in range(N):
                if (k != n) & (k!= nn):
                    dbs[triplet_index]= enod((M[n]*R[n,:] + M[nn]*R[nn,:])/(M[n]+M[nn]) ,R[k,:]   )
                    triplet_index= triplet_index+ 1
    
    return dbs    



