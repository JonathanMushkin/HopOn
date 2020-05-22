# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:33:23 2019

#@author: Jonathan Mushkin (Weizmann Institute of Science)

All functions related to the Keplerian Mechcnics of 3-body motion simulation
When possible, this code is made generic for N-bodies

"""

import numpy  as np
from numpy.linalg import norm
from numba import njit
import hopon.util as util
import hopon.newton as newton

@njit('float64(float64,float64,int32)')
def inverse_kepler_eq(MeanAnom,ecc,n):
    """
    Inverse Kepler Equation solver. Find eccentric anomaly from the mean 
    anomaly and eccentricity
    INPUT:    
        MeanAnom : Mean anomaly (np array)
        e: eccentricity (magnitude) (np array)
        n: number of iteration to perform (int)     
    OUTPUT: 
        EccAnom: Eccentric anomaly (np array)
        
    Testing showed n of about 10 is enough, meaning  the difference 
    between outputs obtained from n=10 and n=10000 are below float64 
    precision, for a large random sample.

    Method:
        Use Halley's method Fixed Point solver
        equation: want to solve eq. f(x) = 0.
        For us, 
        f(E) = E-e*sin(E)-M = 0.
        define df = df/dE, ddf = d^2f / dE^2 
        E(n+1) = E(n) - 2*f*df/(2*df^2 - f*ddf)
         
    """    
    EccAnom = MeanAnom

    for i in range(n):
        f = EccAnom - ecc * np.sin(EccAnom) - MeanAnom
        df = 1 - ecc * np.cos(EccAnom)
        ddf = ecc * np.sin(EccAnom)
        EccAnom = EccAnom - 2 * f * df / (2*df**2 - f *ddf  )
        
    return EccAnom

@njit('float64(float64,float64)')
def mean_to_ecc_anomaly(MeanAnom,ecc):
    """
    Transform mean anomaly to eccentric anomaly. 
    INPUTS:
        MeanAnom: mean-anomaly of a binary. 
        ecc: scalar eccentricity of a binary
    OUTPUTS:
        eccentric anomaly of a binary
    """
    return inverse_kepler_eq(MeanAnom,ecc,50)

@njit('float64(float64,float64)')
def ecc_to_mean_anomaly(EccAnom,ecc):
    return EccAnom-ecc*np.sin(EccAnom)

@njit('float64(float64,float64)')
def true_to_ecc_anomaly(f,e):
    """
    get the eccentric anomaly (E) of a keplerian orbit from the true anomaly (f)
    and scalar eccentricity (e).
    INPUTS: 
        f : true anomaly,
        e : scalar eccentricity
    OUTPUT:
        E : eccentric anomaly
        
    """
    C = ( e+ np.cos(f) )/ (1+e*np.cos(f))
    S = (np.sin(f)*(1-e**2)**(0.5)) / (1+e*np.cos(f))
    return   np.mod(np.arctan2(S,C),2*np.pi)
    
@njit('float64(float64,float64)')
def ecc_to_true_anomaly(E,e):
    """
    get the true anomaly (f) of a keplerian orbit from the eccentric anomaly (E)
    and scalar eccentricity (e).
    INPUTS: 
        E : eccentric anomaly
        e : scalar eccentricity
    OUTPUT:
        f : true anomaly
        
    """    
    return np.mod(2*np.arctan( ( (1+e)/(1-e) )**0.5 * np.tan(E/2)  ), np.pi*2)

@njit(\
      'Tuple((float64[:], float64[:]))(float64,float64,float64,float64,float64,float64,float64)')
def rv_from_keplerian_elements(ecc_mag,a,f,inc,Omega,om,GM):
    """
    Find position and velocity (vectors) of a body, given the orbital parameters and phase.
    INPUTS:
        ecc_mag : scalar eccentricity
        a : semi-major axis
        f : true anomaly
        inc : inclination
        Omega : longitude of ascending node 
        om : (lowe case omega) argument of periapsis
        GM : gravitational parameter, acceleration = -GM/r^2
    OUTPTS:
        r : position vector at given parameters
        v : velocity vector at given parameters
    """
    
    
    h_mag = ((1-ecc_mag**2) * (GM*a))**(1/2) # r-cross-v
    J_hat = np.array([np.sin(inc)*np.sin(Omega) ,\
                          -np.sin(inc)*np.cos(Omega) ,\
                          np.cos(inc)])
    
    asc_node_hat =  np.array([np.cos(Omega),np.sin(Omega),0.0])
    t_hat = util.my_cross(J_hat,asc_node_hat)
    ecc_hat = asc_node_hat * np.cos(om) + t_hat*np.sin(om)
    
    q_hat = util.my_cross(J_hat, ecc_hat)
    # no need to normalize it, as asc_node_hat and third_hat are both
    # orthogonal to J_hat
    
    radius = a*(1-ecc_mag**2)/ (1+ecc_mag*np.cos(f))
        
    vf = h_mag /radius # in phi direction
    vr = ecc_mag * np.sin(f)*h_mag/ a / (1-ecc_mag**2)
    
    v = ecc_hat  * (vr*np.cos(f) - vf*np.sin(f)) + q_hat*(vr*np.sin(f) + vf*np.cos(f))
        
    r = ecc_hat * radius*np.cos(f) + radius*np.sin(f)*q_hat
    
    return r, v

@njit(\
      'Tuple((float64[:],float64,float64,float64,float64,float64))(float64[:],float64[:],float64)'\
      )
def  keplerian_elements_from_rv(r,v,GM):
    """
    Find orbital parameters and phase of an orbit, given position and velocity (vectors).
    INPUTS:
        r : position vector
        v : velocity vector
        GM : gravitational parameter, acceleration = -GM/r^2
    OUTPUTS:
        ecc_vec : eccentricity vector
        f : true anomaly
        a : semi-major axis  
        inc : inclination 
        Omega : longitude of ascending node 
        om : argument of periapsis
    """
    eps = 1e-15 # small number
    r_mag = norm(r)
    E =  (np.sum(v**2)/2 - GM/r_mag) # energy per unit reduced mass
    if E==0.0:
        a = np.inf
    else:
        a = -GM / (2*E) # a> 0 for elliptic / circular orbits, a<0 for hyperbolic
        
    r_hat = r/r_mag
    
    h = util.my_cross(r,v) # angular momentum per unit reduced mass
    J_hat = h/norm(h)
    inc = np.arccos(J_hat[2])

    x_hat = np.array([1.0,0.0,0.0])
    z_hat = np.array([0.0,0.0,1.0])
	# if the inclination is extremely close to 0 or pi, there is no ascending node. We set the asc_node to the x axis then.
    if (np.abs(inc)<eps) | (np.abs(inc-np.pi)<eps):
        asc_node_hat = x_hat
    else:
        asc_node_hat = util.my_cross(z_hat, J_hat)
        asc_node_hat = asc_node_hat / norm(asc_node_hat)
        
    Omega = np.arctan2(h[0],-h[1])
    Omega = np.mod(Omega,2*np.pi)

    
    ecc_vec = util.my_cross(v,h)/GM - r_hat
    
    if norm(ecc_vec)>eps:
        ecc_hat = ecc_vec / norm(ecc_vec)
        om = np.arctan2(norm(util.my_cross(asc_node_hat,ecc_hat)), np.sum(asc_node_hat*ecc_hat) )
    else:
        ecc_hat = asc_node_hat
        om = 0.0
        
    if ecc_hat[2]<0:
        om = np.pi*2 - om
    
    q_hat = util.my_cross(J_hat,ecc_hat)
    f = np.mod(np.arctan2(np.sum(r_hat*q_hat), np.sum(r_hat*ecc_hat)  ),np.pi*2)
     
    return ecc_vec, f, a, inc, Omega, om




@njit('float64(float64,float64,float64,float64)')
def advance_true_anomaly(f0, e, mean_motion, t ):
    """
    Find the true anomaly of the orbit after time t
    INPUTS:
        f0 : current true anomaly (scalar)
        e : scalar eccentricity (scalar)
        mean_motion : mean-anomaly change rate (scalar)
        t : scalar, time to advance the system by
    OUTPUT:
        ft : true anomaly of the orbit after time t (scalar)
    """
    # initial eccentric and mean anomalies
    ea = true_to_ecc_anomaly(f0,e)
    ma = ecc_to_mean_anomaly(ea,e) 
    
    # tareget mean anomaly
    ma_t = np.float64(np.mod(ma + t * mean_motion, 2*np.pi))
    
    # target eccentric anomaly
    ea_t = mean_to_ecc_anomaly(ma_t,e)

    return ecc_to_true_anomaly(ea_t,e)

@njit('Tuple((float64[:,:],float64[:,:],float64))(float64[:,:], float64[:,:], int32[:],float64[:],float64)')
def double_keplerian_motion(R,V,i,M,G):
    """
    Given 3 bodies in an hierarchical triplet, advance the outer orbit to the 
    point reflected by semi-major axis (f -> 2*pi-f), in an unpertubed way (i.e.,
    not interaction between inner and outer orbits). Advance inner orbit according 
    to the time passes.
    INPUTS : 
        R : numpy array of size 3x3, positions of bodies
        V : numpy array of size 3x3, velocities of bodies
        i : int array of length 3. 
			i[0] = index of first inner binary member.
            i[1]  = index of second inner binary member.
            i[2] = index of tertiary, memeber of the outer binary
        M : 1d array of length 3, masses of bodies
        G : Universal Gravitational Constant
                      
    OUTPUTS: 
        R_new: 2d array of size 3x3, positions of bodies after motion
        V_new: 2d array of size 3x3, velocities of bodies after motion
        delta_t : time elapsed during motion
        
    HOW TO READ:
        suffix _i, _o stand for inner, outer orbits
        suffix _t stand for target value, after the motion
        
        e stand for scalar eccentricity
        ea, ma stand for eccentric, mean anomaly
        f stand for true anomaly
        delta_t is the time passed during the Keplerian motion 
    """
    # find total and reduces mass of inner orbit
    M_i = M[i[0]]+M[i[1]]
    mu_i = M[i[0]]*M[i[1]]/M_i
    # find total and reduced mass of outer orbit
    M_o = np.sum(M)
    mu_o = M[i[2]]*M_i/(M_o)
    
	# find relative positions and center of masses of the two binaries
    R_i = (R[i[0],:]*M[i[0]] + R[i[1],:]*M[i[1]])/M_i # inner binary's C.O.M
    r_i = R[i[0],:]-R[i[1]] # inner binary's relative position
    V_i = (V[i[0],:]*M[i[0]] + V[i[1],:]*M[i[1]])/M_i # inner binary's C.O.M velocity
    v_i = V[i[0],:]-V[i[1],:] # inner binary's relative velocity
    
    R_o = (R[i[0],:]*M[i[0]] + R[i[1],:]*M[i[1]] + R[i[2],:]*M[i[2]])/M_o # Outer binary's C.O.M
    V_o = (V[i[0],:]*M[i[0]] + V[i[1],:]*M[i[1]] + V[i[2],:]*M[i[2]])/M_o # Outer binary's C.O.M 
    r_o = R_i-R[i[2],:] # outer binary's relative position    
    v_o = V_i-V[i[2],:] # outer binary's relative velocity
	
    # find Keplerian parameters of both orbits
    ecc_vec_i,f_i,a_i,inc_i,Omega_i,w_i = keplerian_elements_from_rv(r_i,v_i,G*M_i)
    ecc_vec_o,f_o,a_o,inc_o,Omega_o,w_o = keplerian_elements_from_rv(r_o,v_o,G*M_o)
    
    # find magnitude of ecc. vector
    e_i = norm(ecc_vec_i)
    e_o = norm(ecc_vec_o)
    
    # advance the outer orbit from true anom. f to 2*pi-f
    f_o_t = 2*np.pi - f_o	

    # find eccentric and mean anomalies (ea,ma) of outer orbit
    ea_o = true_to_ecc_anomaly(f_o,e_o) # current outer eccentric anomaly
    ea_o_t = true_to_ecc_anomaly(f_o_t,e_o) # target outer eccentric anoamly
    ma_o = ecc_to_mean_anomaly(ea_o,e_o) # current outer mean anomaly
    ma_o_t =ecc_to_mean_anomaly(ea_o_t,e_o) # target outer mean anoamly
    mean_motion_o = (G*M_o)**(0.5) * a_o**(-3/2) # outer mean_motion =  = 2*pi / Orbital Period
    delta_t = (ma_o_t-ma_o)/mean_motion_o #  time duration of keplerian motion
    mean_motion_i = (G*M_i)**(0.5) * a_i **(-3/2) # inner mean motion  
    f_i_t = advance_true_anomaly(f_i,e_i,mean_motion_i ,delta_t ) # target inner mean anomaly
    
    # find position of inner & outer orbits at target time
    r_o_t, v_o_t = rv_from_keplerian_elements(e_o, a_o, f_o_t, inc_o, Omega_o, w_o, G*M_o)
    r_i_t, v_i_t = rv_from_keplerian_elements(e_i, a_i, f_i_t, inc_i, Omega_i, w_i, G*M_i)
	
    R_t = np.zeros((3,3), dtype = np.float64) 
    V_t = np.zeros((3,3), dtype = np.float64)
    R_t[i[0],:] = R_o + mu_o/M_i * r_o_t + r_i_t*mu_i/M[i[0]]
    R_t[i[1],:] = R_o + mu_o/M_i * r_o_t - r_i_t*mu_i/M[i[1]]
    R_t[i[2],:] = R_o - mu_o/M[i[2]] * r_o_t
    
    V_t[i[0],:] = V_o + mu_o/M_i * v_o_t + v_i_t*mu_i/M[i[0]]
    V_t[i[1],:] = V_o + mu_o/M_i * v_o_t - v_i_t*mu_i/M[i[1]]
    V_t[i[2],:] = V_o - mu_o/M[i[2]] * v_o_t
    
    return R_t, V_t, delta_t

@njit('Tuple((int32,int32))(float64[:,:],float64[:,:],float64[:],float64,float64,float64,float64,int32[:,:])')
def choose_NKT(R,V,M,G,h1,h2,lengthscale, triplets):
    """
    Choose whether to perform the next simulation step by (N)ewtonian dynamics,
    (K)eplerian calculation, or to (T)erminate the simulation.
        
    INPUTS: 
        R : N x 3 dimensional np.array-s of positions
        V : N x 3 dimensional np.array-s of velocities
        M : np.array of length N  of masses
        G : Universal Gravitational Constant, scalar
        h1 :    Separation criterion. K and T are selected only if one of the 
                bodies is at least h1 times farther away from the two closest bodies 
                than they are from each other.
        h2 :    Heirarchy criterion. We treat the motion as well separated 
                into inner and outer orbit if the outer radius axis
                is h2 times bigger than the inner semi-major-axis. (h1 and h2 do not have to be the same, but it is a good practice).
        lengthscale:    additional criterion for heirarchy. Useful in cases of 
                        small separation between two bodies, in such a way that
                        separation criterion is met (h1), and the energy has 
                        deviations, which makes errors in the inner orbit's 
                        semi-major-axis calculation. To avoid those errors, 
                        we also demand that the outer radius is h2 times 
                        larger than the lengthscale. ( A good selection is the initial inner orbit's semi-major axis). 
        triplets : a list specifing the exact indexing and members of each triplet. 
            For example, if the indexing means
            0 : ((0,1),2). 1: ((0,2),1). 2: ((1,2),0)
            then triplet index = [[0,1,2],[0,2,1],[1,2,0]]
    OUTPUTS:
            Tuple of 2 outputs.
            output[0] : 'N', 'K' or 'T'. (char)
            output[1] : triplet index. (int)
    """
    # conditions
    is_hierarchical = 0
    triplet_index = 0
    outer_orbit_bound = 0
    positive_drodt = 0
    
    D = util.pairwise_enod(R) # pairwise distances of bodies
    DD = sorted(D)[0:2]       # sorted
    # inner separation (r_i) = DD[0]
    # outer separation (r_o) >= DD[1]. 
    triplet_index = np.argmin(D) # see util.pairwise_enod() and util.create_triplets() for indexing scheme
    Ebs = newton.binary_single_energy(R,V,M,G)
    Epw = newton.pairwise_energy(R,V,M,G)
    E_i = Epw[triplet_index]
    # decide if triplet is heirarchical. 
    # only if relative distances fit, 
    # add tests to make sure no numerical errors due to phase of inner binary,
    # and to some absolute and relative length scales       
        
    # the three bodies are instantaneously separated, there is a point 
    # figuring if they are hierarchical. 
    # for example: if the inner orbit has SMA = 1 and e = 0.999...
    # and the outer orbit has SMA = 3 and e = 0.5, 
    # in could be that r_i = 1e-3, r_o = 1, r_i<<r_o
    # But making an analytical keplerian timestep would be wrong, as during
    # the outer orbit, the inner orbit will reach r_i = 1.9999...
    
    # Epw[triplet_index] = energy of the inner orbit
            
    i0 = int(triplets[triplet_index,0])
    i1 = int(triplets[triplet_index,1])
    i2 = int(triplets[triplet_index,2])
            
    a_i = - G * M[i0]*M[i1]/2 / E_i
    
    r_o = -R[i2,:]+(M[i0]*R[i0,:]+M[i1]*R[i1,:])/(M[i0]+M[i1])
    v_o = -V[i2,:]+(M[i0]*V[i0,:]+M[i1]*V[i1,:])/(M[i0]+M[i1])
    
    # reason for tests below:
    # 	1. DD[0]*h1 < DD[1] : makes sure there is instantaneous separation
    # 	
	#	2. a_i*h2 < DD[1] : makes sure the instantaneous hierarchy, relative to the 
    # 	inner orbit's size. For example, if e_i = 0.999, a_i = 1 and r_o = 3, 
    # 	it might happen that r_i << r_o, even though r_i and r_o will be 
    # 	compareable within a short time. Test 2 is precaution against energy calculations errors that will fool test 1.
    # 
	# 	3. lengthscale*h2< DD[1] : precaution, compare not only to instantaneous separations but also to a constant lengthscale.
    # 
	# 	4. a_i*0.9 < DD[0] : make sure the inner binary is not too close to pericenter. At high e_i (0.999999....), the energy calculations may 
    # 	deviate significantlly due to the small separations.
    # 
	# 	5. E_i<0 : inner binary is bound
    
    if (DD[0]*h1 < DD[1]) & ((a_i*h2) < DD[1]) & (lengthscale * h2 < DD[1]) & ((a_i*0.9) < DD[0]) & ( E_i<0 ) :
        is_hierarchical = 1
            
    # check if outer orbit is bound or not                    
    if Ebs[triplet_index]<0 :
        outer_orbit_bound = 1
    # make sure if outer orbit is going towards or away from pericenter
    if np.dot(r_o,v_o)>0 :
        positive_drodt = 1
    
	# Determine what to do in the next stetp: N, K or T
    if (is_hierarchical==0) | (positive_drodt==0) :
        next_step = 0 # Newtonian step
    if (is_hierarchical==1) &(outer_orbit_bound ==1) &  (positive_drodt==1):
        next_step = 1 # Keplerian step
    if (is_hierarchical==1) &(outer_orbit_bound ==0) &  (positive_drodt==1):
        next_step = 2 # Terminate
                 
    return next_step, triplet_index
 
class orbital_elements(object):
# obejct of orbital elements, calculable from position, velocity and masses

    def __init__(self,r,v,M,mu,G ):
        
        ecc_vec, f, a, inc, Omega, omega = keplerian_elements_from_rv(r,v,M*G)
        
        self.r = r
        self.v = v
        self.ecc_vec = ecc_vec
        self.ecc_mag = np.sum(ecc_vec**2)**(0.5)
        self.E = -G*M*mu/2/a
        self.a = a
         
        if self.ecc_mag > 1.0 and a>0.0: # for hyperbolic orbits
            self.E = np.abs(self.E)
            self.a = -np.abs(a)
        
        self.f = f
        self.inc = inc
        self.Omega = Omega
        self.omega = omega
        self.M = M
        self.mu = mu
        self.G = G