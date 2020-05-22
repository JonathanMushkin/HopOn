# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:33:23 2019

#@author: Jonathan Mushkin (Weizmann Institute of Science)

Functions and object classes that perform, or support performing,3-body 
simulations.
"""

from numba import njit
import numba as nb
import numpy as np
import hopon.util as util
import hopon.newton as newton
import hopon.kepler as kepler

class InitialConditions(object):
    """
    This object contains the bare minimum to set the initial conditions of a simulation.
    """
    def __init__(self,R,V,M,G ):
        self.M = M
        self.R = R
        self.V = V
        self.G = G

class Results(object):
    """
    Results object, to store simulations results. [maybe not really needed]
    """
    def __init__(self, t, d_min, kepler_steps, auto_terminated, last_ind,
                 steps_limit, ic, final_triplet_index, orbit_record):
        """
        INPUTS:
            fill in everything. vectors? scalars? what is it? maybe need to delete this?
        OUTPUTS:
            A Results class object.
        """
        self.M = ic.M # masses
        self.t = t # time
        self.d_min = d_min # minimal reached distance
        self.kepler_steps = kepler_steps # number of keplerian time steps used during the simulation
        self.auto_terminated = auto_terminated # 1/0, if the simulation was terminated automatically (1) or by reaching the steps limit(0).
        self.last_ind = last_ind # last iteration index
        self.steps_limit = steps_limit 
        self.G = ic.G # gravitational constant
        self.triplet_index = final_triplet_index
        self.orbit_record = orbit_record # 
        self.energy_error_norm = 0.0 
        
        if np.size(self.orbit_record)>0:
            E0 = newton.total_energy(ic.R,ic.V,ic.M,ic.G)
            norm_error = 0.0
            for j in range(np.size(self.orbit_record)):
                norm_error =  np.max([norm_error,np.abs((self.orbit_record[j].E - E0)/E0)])
            
            self.energy_error_norm = norm_error
        # count the number of exchanges that happen
        aux_triplet = 0
        N_exchanges = 0
        for i in range(1,np.size(orbit_record)):
            if (orbit_record[i].triplet_index != aux_triplet):
                N_exchanges = N_exchanges +1
                aux_triplet = orbit_record[i].triplet_index
        self.N_exchanges = N_exchanges
       
        
class orbits_param_record(object):
    """
    A orbital-parameters record. Used to store the parameters of the inner and outer binary's during snapshots of the simulation.
    """
    def __init__(self, R, V, M, G, triplet_index, t, step, record_type, d_min):
        """
    	INPUTS:
    	
    	OUTPUTS:
    		an orbital_param_record object.
    	"""
        self.triplet_index = triplet_index
        triplet = util.create_triplets(3)[triplet_index,:]        
        i0 = triplet[0]
        i1 = triplet[1]
        i2 = triplet[2]
        
        
        M_o = M.sum()
        M_i = M[i0]+M[i1]
        mu_i = M[i0]*M[i1]/M_i
        mu_o = M[i2]*M_i/M_o
        
        r_o = (R[i0,:]*M[i0]+R[i1,:]*M[i1])/M_i - R[i2,:]
        v_o = (V[i0,:]*M[i0]+V[i1,:]*M[i1])/M_i - V[i2,:]
        r_i = R[i0,:]-R[i1,:]
        v_i = V[i0,:]-V[i1,:]
        
        ecc_vec_o,f_o,a_o,inc_o,Omega_o,omega_o = kepler.keplerian_elements_from_rv(r_o,v_o,G*M_o,mu_o)
        ecc_vec_i,f_i,a_i,inc_i,Omega_i,omega_i = kepler.keplerian_elements_from_rv(r_i,v_i,G*M_i,mu_i)
        
        self.outer = kepler.orbital_elements(r_o,v_o,M_o,mu_o,G)
        self.inner = kepler.orbital_elements(r_i,v_i,M_i,mu_i,G)

        self.t = t
        self.step_index = step
        if record_type =='kepler':
            self.kepler_step = 1
        else:
            self.kepler_step = 0
        
        self.E = newton.total_energy(R,V,M,G)
        # keep track of the closest any two bodies got so far
        self.d_min = d_min
        
def create_empty_initial_conditions():
    """
    Create an empty InitialConditions object, to be filled later.
    INPUTS: 
    	None
    OUTPUTS:
    	InitialConditions object
    """
    M = np.zeros(3)
    R = np.zeros((3,3))
    V = np.zeros((3,3))
    G = 1.0
    return InitialConditions(R,V,M,G)

def create_empty_results():
    """
    Creates an empty Results object, to be filled later.
    INPUTS: 
    	None
    OUTPUTS:
    	Results object
    """
    R = np.zeros((3,3))
    V = np.zeros((3,3))
    t_ar = np.zeros(1)
    t = np.zeros(1)
    d_min = nb.float64(0)
    kepler_steps = np.int32(0)
    auto_terminated = np.int32(0)
    last_ind = np.int64(0)
    steps_limit = nb.int64(0)
    
    ic = create_empty_initial_conditions()
    R_ar = np.zeros((1,3,3))
    V_ar = np.zeros((1,3,3))
    triplet_index = nb.int64(0)
    
    return Results( R,V,t,d_min,kepler_steps,auto_terminated,last_ind, 
                   steps_limit,ic, R_ar, V_ar,t_ar, triplet_index )

def initialize_simulation_elliptic(M,G,\
                                   a_i, f_i, e_i, inc_i, Omega_i, w_i,\
                                   a_o, f_o, e_o, inc_o, Omega_o, w_o):
    
    """
    initialize a simulation with elliptic outer orbit and inner orbits
    INPUTS:
        M : Masses of 3 bodies
        G : Univeral gravitational constant
        a_i : inner semi-major axis
        f_i : inner true anomaly
        e_i : inner scalar eccentricity
        inc_i : inner inclination, w.r.t plane set by outer plane of motion
        Omega_i : inner Omega, w.r.t. plane set by outer plane of motion
        w_i : inner w (lowercase omega), w.r.t plane set by outer plane of motion
        a_o,f_o,e_o,inc_o,Omega_o,w_o : same, for outer orbit
        
    OUTPUTS:
        ic : Initial conditions object
    """
    mu_i = M[0]*M[1]/(M[0]+M[1])
    M_i = M[0]+M[1]
    
    mu_o = (M[0]+M[1])*M[2]/(M[0]+M[1]+M[2])
    M_o = M[0]+M[1]+M[2]
    
    r_o,v_o = kepler.rv_from_keplerian_elements(e_o,a_o,f_o,inc_o,Omega_o,w_o,G*M_o) 
    r_i,v_i = kepler.rv_from_keplerian_elements(e_i,a_i,f_i,inc_i,Omega_i,w_i,G*M_i) 
    
    R = np.zeros((3,3))
    V = np.zeros((3,3))
    R[0,:] = +mu_i/M[0] * r_i + r_o * mu_o/M_i
    V[0,:] = +mu_i/M[0] * v_i + v_o * mu_o/M_i
    R[1,:] = -mu_i/M[1] * r_i + r_o * mu_o/M_i    
    V[1,:] = -mu_i/M[1] * v_i + v_o * mu_o/M_i
    
    R[2,:] = -r_o * mu_o/M[2]
    V[2,:] = -v_o * mu_o/M[2]
    
    return InitialConditions(R,V,M,G)

def perform_simulation(ic, Nsteps, kep_sep_crit,kep_hier_crit, hier_lengthscale, eval_NKT_every, dt00):
    """
    Perform 3-body simulation. 
    Use a leapfrog Kick-Drift method, with Keplerian dynamics where possible to
    save run-time. 
    
    INPUTS:
        ic : simulator.IniticalConditions class object. contians the initial 
        conditions for the simulation
        Nsteps : upper bound on the number of steps to make before terminating 
        simulations
        kep_sep_crit : declare separation if r_o/ri > kep_sep_crit
        kep_hier_crit : delcare hierarchy ro/a_i > kep_hier_crit
        hier_lengthscale : constant lengthscale, to outer orbit separation 
        when deciding if triplet is heirarchical or not
        hierarhcy if ro/hier_lengthscale > kep_hier_crit
        eval_NKT_every : number of time-steps between NKT checks
        dt00 : basic time step, in fraction of inner orbit period
        
    OUTPUTS:
        res: simulator.Results class object. contain all the informaiton about 
        the simulation. 
        
    """ 
	##########################################
	####### prepare for integration ##########
	##########################################
	
    t=0.0
    
    kepler_steps = nb.int32(0)
    auto_terminated = nb.int32(0)
    G = ic.G
    M = ic.M
    R = ic.R
    V = ic.V
    E0 = newton.total_energy(R,V,M,G)
    U0 = newton.total_potential_energy(R,M,G)
    
    E_i = newton.pairwise_energy(R,V,M,G)[0] 
    a_i = - ic.G * M[0]*M[1] / 2 / E_i # inner orbit energy    
    dt0 = dt00 * (a_i**3 / G/ (M[0]+M[1]))**(0.5)   # basic time step, physical units
    
    R_lf = R + \
        0.5 * V*dt0*( (E0-newton.total_kinetic_energy(V,M))/U0) **(-3/2) # Leapfrog position. See hopon.leapfrog_step() for details.
    
    d_min = np.min(util.pairwise_enod(R) ) # minimal distance between any two bodies. updated during simulations.
    triplets = util.create_triplets(nb.int32(3)) 
    i=nb.int64(0)
    triplet_index = nb.int64(0)
    
    Nrec =   nb.int64(100) # number of records to allow. updates during simulation
    # record_flag = True marks that the current Orbital Parameters are to be recorded.
	# It is switched to False after each record.
	# It is sitched to true by the function record_decision()
    record_flag =True
    orbital_record = [None]*Nrec # initialize orbital record object array
    rec_ind = nb.int64(0) # record index
    
	#####################
    # start integration #
	#####################
	
    if record_flag:
        orbital_record[rec_ind] = orbits_param_record(R,V,M,G,0,t,i,'initial', d_min)
        rec_ind = rec_ind+1
    
    for i in range(1,Nsteps):
			# Start of NKT evaluation
            # select (N)ewtonian (0), (K)eplerian (1) or (T)ermination (2)
        if np.mod(i,eval_NKT_every)==0:
            NKT_selection = kepler.choose_NKT(R,V,M ,G, kep_sep_crit,kep_hier_crit, hier_lengthscale, triplets)
            # how to read kepler.choose_NKT output:
            # NKT_selection[0] = 0,1,2 for Newtoian step / Keplerian step / termination
            # NKT_selection[1] = triplet index 
            
        else :
            NKT_selection = (0,0)
        # end of NKT evaluation. 
        
		# Perform step according to NKT selection.
        if (NKT_selection[0]==2): # TERMINATION : record orb. param then terminate simulation
            # record orbital parameters
            orbital_record[rec_ind] = orbits_param_record(R,V,M,G,NKT_selection[1],t,i,'terminate', d_min)
            rec_ind = rec_ind+1
            if rec_ind>= np.shape(orbital_record)[0]:
                orbital_record = np.append(orbital_record,[None]*Nrec)
            # terminate simulation
            auto_terminated = 1
            triplet_index = NKT_selection[1]
            return Results(t=t, d_min=d_min, kepler_steps=kepler_steps, auto_terminated=auto_terminated,
                           last_ind=i,steps_limit=Nsteps, ic=ic, final_triplet_index=triplet_index,
                           orbit_record=orbital_record[:rec_ind])
                
        if (NKT_selection[0]==1) : # KEPLERIAN : record orbital parameters then perform Keplerian motion
            # record before the keplerian motion
            if record_flag:
                orbital_record[rec_ind] = orbits_param_record(R,V,M,G,NKT_selection[1],t,i,'kepler', d_min)
                rec_ind = rec_ind+1
                if rec_ind>= np.shape(orbital_record)[0]:
                    orbital_record = np.append(orbital_record,[None]*Nrec)
            record_flag=True
            # perform double-keplerian motion, analytically
            kepler_steps = kepler_steps+1
            R, V, dt = kepler.double_keplerian_motion(\
                R,V,triplets[NKT_selection[1]],M,G)
            t = t+dt
            # re-calculate reference energy and potential energy, as precaution against numerical deviations.
            E0 = newton.total_energy(R,V,M,G)
            U0 = newton.total_potential_energy(R,M,G)

            E_i = newton.pairwise_energy(R,V,M,G)[NKT_selection[1]] 
            a_i = - ic.G * M[0]*M[1] / 2 / E_i
            i0 = triplets[NKT_selection[1],0]
            i1 = triplets[NKT_selection[1],1]
            dt0 = dt00 * (a_i**3 / G/ (M[i0]+M[i1]))**(0.5)   
            
            K = newton.total_kinetic_energy(V,M)
            
            R_lf = R + \
                V * dt0/2 * ((E0-K)/U0)**(-3/2)
            
        if(NKT_selection[0]==0):  # NEWTONIAN: Decide if to record, then perform leapfrog step
            
            # decide & record orbital patameters
            record_now, record_flag, triplet_index = record_decision(R,V,M,G,record_flag,triplets)
            if record_now:
                orbital_record[rec_ind] = orbits_param_record(R,V,M,G,triplet_index,t,i,'leapfrog', d_min)
                rec_ind = rec_ind+1
            if rec_ind>= np.shape(orbital_record)[0]:
                orbital_record = np.append(orbital_record,[None]*Nrec)
            # end recording
            
            # perform step
            R_lf, R, V, dt =newton.leapfrog_step(R_lf, V, M ,G, dt0, U0, E0)
            t=t+dt
            
            # decide if new minimal distance reached
            if d_min > np.min(util.pairwise_enod(R) ):
                d_min = np.min(util.pairwise_enod(R) )
  
	########################
    ## end of integration ##
	########################
	
    # record the final orbital state of the system
    record_now, record_flag, triplet_index = record_decision(R,V,M,G,record_flag,triplets)
    orbital_record[rec_ind] = orbits_param_record(R,V,M,G,triplet_index,t,i,'terminate', d_min)
    rec_ind = rec_ind+1
            

    return Results(t=t, d_min=d_min,
                   kepler_steps=kepler_steps, auto_terminated=auto_terminated,
                   last_ind=i, steps_limit=Nsteps,
                   ic=ic, final_triplet_index=triplet_index,
                   orbit_record=orbital_record[:rec_ind])
                                    

@njit('Tuple((boolean,boolean,int32))(float64[:,:],float64[:,:],float64[:],float64,boolean,int32[:,:])')
def record_decision(R,V,M,G,record_flag,triplets):
    """
    Decide whether to record the current orbital parameters of the inner and 
    outer orbits.
    INPUTS:
        R : 3x3 positions 2d-array 
        V : 3x3 velocities 2d-array
        M : 3x1 massses 1d-array
        G : Universal gravitational constant
        record_flag : logical True/False. True of the simulation is ready to 
        record the following configuration, False if not. 
        triplets : 3x3 array of indices 0-2. see MyPackage.util.create_triplets
        
    OUTPUTS:
        record_flag : logical True/False. RECORD_DECISION function may turn 
        True to False if it suggests to record now (thus preventing recording
        same orbit twice), or turn False to True (if the orbit's phase changed, 
        from 0-pi to pi-2*pi)
        if_to_record : logical True/False. True of the current position is to 
        be recorded NOW, and False if not.        
    """
    rec_eps = 1e-4
    if_to_record = False
#    D  = sorted(util.pairwise_enod(R))
    D = util.pairwise_enod(R)
    triplet_index = nb.int32(np.argmin(D) )
    j0 = triplets[triplet_index,0]
    j1 = triplets[triplet_index,1]
    j2 = triplets[triplet_index,2]
    
        # get SMA of all possible outer orbits
    mu = (M[j0]+M[j1])*M[j2]/M.sum()
    Ebs = newton.binary_single_energy(R,V,M,G)[triplet_index]
    if np.abs(Ebs)<1e-10:
        a_o = np.inf
    else:
        a_o = np.abs( G * M.sum()*mu/2 * Ebs**(-1)) 
    
    ro = (R[j0,:]*M[j0]+R[j1,:]*M[j1])/(M[j0]+M[j1])-R[j2,:]
    vo = (V[j0,:]*M[j0]+V[j1,:]*M[j1])/(M[j0]+M[j1])-V[j2,:]    
    ro = (R[j0,:]*M[j0]+R[j1,:]*M[j1])/(M[j0]+M[j1])-R[j2,:]
    vo = (V[j0,:]*M[j0]+V[j1,:]*M[j1])/(M[j0]+M[j1])-V[j2,:]
        
        # (r dot v) / |r cross v|
    rdv_rcv = np.sum(ro*vo)/ np.linalg.norm(util.my_cross(ro,vo))
        # rdv_rcv > 0 implies the outer orbit is in to (0,pi) phase
        # |rdv_rcv| < eps implies the orbit is near periapsis / apoapsis
        # add condition |r_o| > a_o to make sure it records the 
        # part of the orbit
        
    if (rdv_rcv >0) & (rdv_rcv < rec_eps) & (np.linalg.norm(ro) > a_o) & (record_flag):
        record_flag = False
        if_to_record = True
        
    if (~record_flag) & (rdv_rcv<0.0):
        record_flag =True
    
    return (if_to_record, record_flag, triplet_index)