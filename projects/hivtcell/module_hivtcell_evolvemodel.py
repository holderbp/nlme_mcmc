import numpy as np
import pandas as pd


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

def initialize(mh_dsfi, mh_yvtfi):
    """
    Public method called by model hypothesis module to set these
    variables (which were created by the data module)
    """
    global datasets_for_indiv, yvals_type_for_indiv
    datasets_for_indiv = mh_dsfi
    yvals_type_for_indiv = mh_yvtfi

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
    if (parkey in indiv_pop.columns):
        val = indiv_pop.at[indiv_name, parkey]
    elif (parkey in indiv_nopop.columns):
        val = indiv_nopop.at[indiv_name, parkey]
    else:
        val = other[parkey]
    return val

def run_evolve_model(em, indiv_name, X):
    """
    Called by get_evolve_model_solutions() method.  
    
    Returns dictionary of all data sets produced by
    the evolve-model run, with y_data_type as key.
    """
    if (em == 'linear'):
        # First find the parameters among the parameter objects
        m = find_par_in_par_objects('m', indiv_name)
        b = find_par_in_par_objects('b', indiv_name)
        # Then calculated y-values
        Y = m*X + b
        return {'reaction-time_ms': np.array(Y)}
        
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
