import numpy as np
import pandas as pd
import hivtcell_teivgamma as hivtcell

#=======================================================
#                   Parameters
#=======================================================
#--- Evolve-model and Error-model pars
#    (imported from model_hyp module at each log-like calc)
indiv_pop = None     # dataframe (rows are individuals)
indiv_nopop = None   # dataframe (rows are individuals)
other = None         # dictionary
#--- Data set pars
datasets_for_indiv = None
yvals_type_for_indiv = None

def run_evolve_model(em, indiv_name, X):
    """
    Called by get_evolve_model_solutions() method.  
    
    Returns dictionary of all data sets produced by
    the evolve-model run, with y_data_type as key.
    """
    if (em == 'virus-decay-moi'):
        #
        #=== VIRUS DECAY
        #
        # First find the parameters among the parameter objects
        A = find_par_in_par_objects('A', indiv_name)
        B = find_par_in_par_objects('B', indiv_name)
        thalf = find_par_in_par_objects('thalf', indiv_name)
        c = np.log(2)/thalf
        # Then calculated y-values
        Y = A * (1.0 - np.exp( - B * np.exp( -c * X) ) )
        return {'count_inf': np.array(Y)}
    elif (em == 'uninfected-timeseries'):
        #
        #=== UNINFECTED TIMESERIES
        #
        # Find the parameters among the parameter objects
        N = find_par_in_par_objects('N', indiv_name)
        tauT = find_par_in_par_objects('tauT', indiv_name)
        nT = find_par_in_par_objects('nT', indiv_name)
        s = find_par_in_par_objects('s', indiv_name)
        dD = find_par_in_par_objects('dD', indiv_name)
        # Set the parameters in the evolve-model module
        hivtcell.setpars(N=N, tauT=tauT, nT=nT, s=s, dD=dD)
        # Run the evolvemodel
        tot, dead = hivtcell.evolve_uninf('solve_ivp', X)
        return {
            'count_tot': tot,
            'count_dead': dead
        }
    elif (em == 'infected-timeseries'):
        #
        #=== INFECTED TIMESERIES
        #
        # placeholder until prepped
        return {
            'count_tot': None,
            'count_dead': None,
            'count_inf': None,
            'count_inf_and_dead': None,
        }
        
def get_evolve_model_solutions(indiv_name, em, xvals):
    """
    Public method called by mod-hyp module's log-likelihood.
    Must call set_pars() first to set current parameter values.
    
    Runs an evolve-model for a given individual, using their own
    parameters (plus the "other", if necessary) and get back a
    dictionary numpy vectors of yvals each data type:

         soln =   {
                    ydata_type_1: [y1(x1), y1(x2), ... , y1(xn)],
                    ydata_type_2: [y2(x1), y2(x2), ... , y2(xn)],
                                  ...
                    ydata_type_m: [ym(x1), ym(x2), ... , ym(xn)]
                  }

    Then put the soln values into dictionary with dataset as key
    """
    soln = run_evolve_model(em, indiv_name, xvals)
    ymod = {}
    for d in datasets_for_indiv[indiv_name][em]:
        ymod[d] = soln[yvals_type_for_indiv[indiv_name][em][d]]
    return ymod

#======================================================================#
#                                                                      #
#                No adjustments should be needed below                 #
#                                                                      #
#======================================================================#

def initialize(mh_dsfi, mh_yvtfi, mh_parconst):
    """
    Public method called by model hypothesis module to set these
    variables (which were created by the data module)
    """
    global datasets_for_indiv, yvals_type_for_indiv, par_constants
    datasets_for_indiv = mh_dsfi
    yvals_type_for_indiv = mh_yvtfi
    par_constants = mh_parconst

def set_pars(mh_indiv_pop, mh_indiv_nopop, mh_other, output_info=False, indentstr=None):
    """
    Public method used by model hypothesis module to transfer current parameters
    """
    global indiv_pop, indiv_nopop, other
    #--- Set the parameter objects
    #
    #        - individual fit-model parameters governed by pop distribution (df)
    #        - individual fit-model parameters not governed by pop dist (df)
    #        - other (e.g., error-model) parameters (dict)
    #
    indiv_pop = mh_indiv_pop
    indiv_nopop = mh_indiv_nopop    
    other = mh_other

def find_par_in_par_objects(parkey, indiv_name):
    """
    Parameters needed for evolve_model must be
    in one (and only one) of three places. 
    """
    if (parkey in par_constants):
        return par_constants[parkey]
    elif (parkey in indiv_pop.columns):
        return indiv_pop.at[indiv_name, parkey]
    elif (parkey in indiv_nopop.columns):
        return indiv_nopop.at[indiv_name, parkey]
    elif (parkey in other):
        return other[parkey]
    else:
        print("***Error (evolvemodel):", parkey, "not in parameter objects.")

