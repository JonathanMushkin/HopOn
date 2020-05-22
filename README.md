# HopOn
A Python 3 code for direct integration of the hierarchical 3-body problem.

HopOn is a PYTHON 3.6 package, tailored for the hierarchical 3-body problem. 
The package provides tools to create 3-body systems, evolve them numerically and record the systems parameters through time. 
The core of the code is an Drift-Kick Leapfrog symplectic integrator, from the family of integrators suggested by Preto &
Tremaine (1999).
To minimize performance time, HopOn employs the Numba Just-in-Time compiler (Lam et al. 2015).
At sufficiently large separations, the integrator can use the analytical solution to the motion. 
To reduce the output data size, HopOn records the orbital parameters once every outer orbit.
