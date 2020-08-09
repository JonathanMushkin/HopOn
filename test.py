# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:25:36 2018

@author: Jonathan Mushkin (Weizmann Institute of Science)
"""
#%% import
import sys
import hopon.simulator as sim
import hopon.constants as con
import numpy as np
import matplotlib.pyplot as plt
#%% initialize 

M= np.array([con.Msolar]*3)
G = con.G
a_i= con.AU * 1.0
e_i = 0.5
f_i = 0.0
inc_i = np.pi * 0.9
Omega_i = 0.0
w_i = 0.0
r_p_o = a_i * 5
e_o = 0.8
h1 = 100
h2 = 100
hls = a_i # hierarchical lengths-scale, 
dt00= 0.003
eval_nkt_every = 100
Nsteps = np.int64(5e6)

a_o = r_p_o / (1-e_o)

mu_o = M[2]*(M[1]+M[0])/(M[0]+M[1]+M[2])
ic = sim.initialize_simulation_elliptic(M=M,G=G,
										a_i=a_i, f_i=f_i, e_i=e_i, inc_i=inc_i, Omega_i=Omega_i, w_i=w_i,
										a_o=a_o, f_o=3*np.pi/2, e_o=e_o, inc_o=0.0, Omega_o=0.0, w_o=0.0)


print('initial conditions created successfully')

#%% perform simulation
sr = sim.perform_simulation(ic=ic, Nsteps=Nsteps,kep_sep_crit=h1,kep_hier_crit=h2,
                                   hier_lengthscale=hls ,eval_NKT_every=eval_nkt_every,
                                   dt00=dt00 )       
print('simulation performed successfully')

#%% plot inner eccentricity evolution with time

ei =np.array( [sr.orbit_record[i].inner.ecc_mag for i in range(len(sr.orbit_record))])
t = np.array( [sr.orbit_record[i].t for i in range(len(sr.orbit_record))])

plt.plot(t/con.yr,ei)
plt.xlabel('t (years)')
plt.ylabel('inner eccentricity')
plt.ylim((0,1))