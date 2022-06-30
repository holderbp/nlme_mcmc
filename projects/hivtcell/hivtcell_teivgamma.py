import sys
import math
import numpy as np
import scipy.integrate as spi

class pars:
    #=== Uninfected timeseries pars
    N = 3e5 # initial number of cells
    tauT = 6.0 # average time to death in cell culture, h
    nT = 5
    s = 0.0 # 1/h
    dD = 1e-3 # h

def setpars(**kwdargs):
    """Allows user to set parameter values for ODEand ICs"""
    for kw in kwdargs.keys():
        setattr(pars,kw,float(kwdargs[kw]))
    checkpars()

def checkpars():
    # round off nT and cast to integers
    pars.nT = max(1, int(np.round(pars.nT)))

def ode_uninf(t, X):
    dT = pars.nT / pars.tauT
    dXdt = np.zeros(pars.nT+1)
    X[X<0] = 0.0
    # live cells 
    dXdt[0] = pars.s*X[0] - dT*X[0]
    for i in np.arange(1,pars.nT):
        dXdt[i] = dT*(X[i-1] - X[i])
    # dead cells
    dXdt[-1] = dT*X[-2] - pars.dD*X[-1]
    return dXdt

def evolve_uninf(evolve_type, tvals):
    Tinit = np.zeros(pars.nT+1)
    Tinit[0] = pars.N
    if (evolve_type == 'solve_ivp'):
        sol = spi.solve_ivp(ode_uninf, [tvals[0], tvals[-1]], Tinit,
                            t_eval = tvals)
    # sol.y has shape ( len(Tinit), len(tvals) )
    dead = sol.y[-1,:]  # last row are dead cells
    tot = np.sum(sol.y, axis=0)
    return [tot, dead]
