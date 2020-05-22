# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:33:23 2019

#@author: Jonathan Mushkin (Weizmann Institute of Science)

All functions related to the Newtonian Mechanics of  3-body motion simulation
When possible, this code is made generic for N-bodies

"""

import numpy  as np
from numba import njit
import hopon.util as util

@njit('float64[:](float64[:,:],float64[:,:],float64[:],float64)')
def pairwise_energy(R,V,M,G):
    """
    return the energies of each pair of particles, 
    calculated in their own c.o.m rest frame. 
    
    INPUTS: 
        R : 2d array of shape N x 3 of positions
        V : 2d array of shape N x 3 of velocities
        M : 1d array of lenght N of masses
        G : Universal Gravitational Constant
        
    OUTPUTS: 
        E : 1d array of length N*(N-1)/2 of pair energies.
    pairs are ordered by ascending first index, then ascending second index
    meaning: (0,1), (0,2), (0,3), ... (1,2), (1,3),... (N-1,N)
    
    """
    d = util.pairwise_enod(R) # relative distances between all bodies
    v = util.pairwise_enod(V) # relative velocities between all bodies
    
    N = R.shape[0]
    E = np.zeros(int(N*(N-1)/2))
    pair_index = 0
    for n in range(0,N-1):
        for nn in range (n+1, N):
            E[pair_index] = -(G*M[n]*M[nn] / d[pair_index]) + 0.5 * (M[n]*M[nn])/(M[n]+M[nn]) * (v[pair_index]**2)
            pair_index = pair_index+1    
    return E

@njit('float64[:](float64[:,:],float64[:,:],float64[:],float64)')
def binary_single_energy(R,V,M,G):
    """
    Calcualtes the energy of the 2 body system, composed of one single body and
    a the center of mass of a pair treated. (In the language of 3-body systems, 
    this is the outer-orbit / outer binary energy).
    
    INPUTS: 
        R : N x 3 dimensional np.array-s of positions
        V : N x 3 dimensional np.array-s of velocities
        M : np.array of length N  of masses
        G : Universal Gravitational Constant
            
    OUTPUTS: 
        Ebs, 1d array of length N*(N-1)*(N-2)/2 of pair-signles energies.
        triplets are ordered by ascending first index (first pair member), 
        then ascending second index (second pair member),
        then ascending third index (single body). Meaning:
        ((0,1),2),((0,1),3),..., 
        ((0,2),1),((0,2),3),...,    
        ((1,2),0),((1,2),3),...
        ((N-1,N),0), ((N-1,N),1)...((N-1,N),N-2)
    """
    N = R.shape[0]
    Ebs = np.zeros(int(N*(N-1)*(N-2)/2))
    
    triplet_index = int(0)
    for n in range(N-1):
        for nn in range(n+1,N):
            for k in range(N):
                if (k != n) & (k!= nn):
                    mu = M[k]*(M[nn]+M[n])/ (M[n]+M[nn]+M[k])
                    r = util.enod((M[n]*R[n,:] + M[nn]*R[nn,:])/(M[n]+M[nn]), R[k,:])
                    v = util.enod((M[n]*V[n,:] + M[nn]*V[nn,:])/(M[n]+M[nn]), V[k,:])
                    Ebs[triplet_index] = -G*mu*(M[n]+M[nn]+M[k])/r + 0.5 * mu*v**2
                    
                    triplet_index= triplet_index+ 1
        
    return Ebs

@njit('float64(float64[:,:],float64[:,:],float64[:],float64)')
def total_energy(R,V,M,G):
    """
    Calculate total energy of an N body system, 
    w.r.t the center-of-mass rest frame
    INPUTS:
        R : 2d array of shape Nx3, of positions
        V : 2d array of shape Nx3, of velocities
        M : 1d array of length N, of masses
        G : Universal Gravitational Constant,
    OUTPUT:
        E : total energy
        
    """
    N = R.shape[0] # number of bodies
    E = 0 # initialize energy
    V0 = np.zeros(3) # initialize center of mass velocity,
    for n in range(N):
        V0 = V0 + V[n,:]*M[n]
    V0 = V0/np.sum(M)
    # find overall energy
    for n in range(N):
        # add kinetic energy of n-th body
        E = E + ((util.enod(V[n,:],V0))**2) * 0.5 * M[n]
        # add potential energy of n-th body relative to other bodies, without double counting
        for nn in range(n+1,N):
             E = E - G*M[n]*M[nn] / util.enod(R[n,:],R[nn,:])
    return E

@njit('float64(float64[:,:],float64[:])')
def total_kinetic_energy(V,M):
    """
    Find the total kinetic energy of an N body system,
    w.r.t the center of mass rest frame.
    INPUTS:
        V : 2d array of shape Nx3, of velocities
        M : 1d array of length N, of masses
    OUTPUT:
        K : total kinetic energy
    """
    N = V.shape[0] # number of bodies
    K = 0 # initialize kinetic energy
    V0 = np.zeros(3) # initialize center of mass velocity
    # find refernce velocity
    for n in range(N):
        V0 = V0 + V[n,:]*M[n]
    V0 = V0/np.sum(M)
    # find kinetic energy
    for n in range(N):
        K = K + ((util.enod(V[n,:],V0))**2)* 0.5 * M[n]
   
    return K

@njit('float64(float64[:,:],float64[:],float64)')
def total_potential_energy(R,M,G):
    """
    Find the total gravitational potential energy of N body system
    INPUTS:
        R : 2d array of shape Nx3, of positions
        M : 1d array of length N, of masses
        G : Universal Gravitational Constant,
    OUTPUT:
        U : total potential energy
    """
    U = 0
    N = R.shape[0] 
  
    for n in range(N):
        for nn in range(n+1,N):
             U = U - G*M[n]*M[nn] / util.enod(R[n,:],R[nn,:])
    
    return U


        
@njit('float64[:,:](float64[:,:],float64[:],float64)')
def acceleration(R,M,G):
    """
    Calculate gravitational acceleration applied to each body by the other bodies
    INPUTS: 
        R : N x 3 dimensional np.array of positions
        M : np.array of length N of masses
        G : Universal Gravitational Constant, scalar
        
    OUTPUT:
        a : N x 3 dimensional np.array of accelerations
    """
    N = R.shape[0]
    a = np.zeros((N,3)) # initialize accelerations
    for n in range(N):
        for nn in range(N):
            if n!=nn:
                a[n,:] += G*M[nn] * (R[nn,:]-R[n,:]) / util.enod(R[nn,:],R[n,:])**(3)    
    return a

@njit('float64(float64,float64,float64)')
def leapfrog_dt(dt0,U,U0):
    """
    Calculate the adaptive timestep for a leapfrog 3-body integrator. 
    INPTUS:
        dt0 : referece time step size 
        U : potential energy at current leapfrog position
        U0 : reference potential energy
    OUTPUT:
        dt : adaptive timestep
    
    NOTE: This function is nice to have, but even nicer not to use
    """
    return dt0 * ( np.abs(U/ U0)) **(-3/2)

@njit(\
      'Tuple((float64[:,:], float64[:,:],float64[:,:],float64))(float64[:,:],float64[:,:],float64[:],float64,float64,float64,float64)'\
      )
def leapfrog_step(R,V,M,G,dt0,U0,E0 ):
    """
    Perfrom a leapfrog step for N body motion, in the Kick-Drift method
    INPUTS:
        R : N x 3 dimensional np.array of leapfrog positions
        V: N x 3 dimensional np.array of leapfrog velocities
        M: np.array of length N  of masses
        G : Universal Gravitational Constant, scalar
    OUTPUTS:
        Rn : Next leapfrog positions
        R_cotemp : Next positions, co-temporal with velocities. 
        Vn : next leapfrog velocities
        dt : time increment (related to the kick)
        
    NOTE: LEAPFROG vs COTEMPORAL POSITIONS
    The leapfrog integrator is powerful because it takes positions and 
    velocities at half-step shift. This also means that R at step i and V at 
    step i refer to quantities at different moments in time.
    Actual physical quantities that are derived from R,V should use the values 
    at a specific moment.
    This is why this functions returns both the LEAPFROG values of the positions,
    and the values co-temporal with the velocities.
    """
    # kick 
    U = total_potential_energy(R,M,G)
    acc = acceleration(R,M,G)
    Vn = V + acc*dt0*(U/U0)**(-3/2) # new velocity
    
    # drift
    K = total_kinetic_energy(Vn,M)

    R_cotemp = R + Vn * dt0/2 * ((E0-K)/U0)**(-3/2) # new R, cotemporal with V
    Rn = R_cotemp + Vn * dt0/2 * ((E0-K)/U0)**(-3/2) # new R, half-step from V

    return Rn, R_cotemp, Vn, dt0*(U/U0)**(-3/2)