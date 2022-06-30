import numpy as np
import pandas as pd
import datetime as dt
import functools
import scipy.optimize as spo
import scipy.stats as sps
import scipy.special as spsp
import scipy.spatial as spspat
import emcee
import corner
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mplbp
import seaborn as sns
# make default style for seaborn
sns.set()
import multiprocessing as mp

#==============================================================
#            
#   This module contains parameters and methods to be used by
#   both the MCMC run script (mixed-effects_mcmc.py), and the
#   associated plotting routines (mixed-effects_plot-data.py
#   and mixed-effects_cornerplot.py) and the model-hypothesis
#   module(s).
#
#   This module is imported independently by each, and uses
#   non-overlapping parts, which are blocked out here.  The
#   values of all parameters are generally set within the
#   file that is using this module.
#
#==============================================================

# flush the print output by default (for running remotely)
print = functools.partial(print, flush=True)

#==============================================================
#            
#      Parameters used by the mixed-effects_mcmc.py
#
#==============================================================

#=== Set-up parameters
start_time = None
ncpu = mp.cpu_count()

#=== Maximum posterior/likelihood
do_maximum_likelihood = None
minimize_tol = None
minimize_method = None

#=== The model hypothesis log-likelihood and prior function
mod_log_like_and_prior = None

#=== MCMC run and output parameters
mcmc_do_backend = None
mcmc_existing_chain_file = None
mcmc_do_multiprocessing = None
mcmc_ncpus_to_use = None
mcmc_show_progress = None
mcmc_run_to_convergence = None
mcmc_run_to_convergence_check_steps = None
mcmc_run_to_convergence_N_autocorrs = None
mcmc_run_to_convergence_delta_tau_frac = None
mcmc_run_to_convergence_verbose = None
mcmc_run_to_convergence_plot_autocorr = None
mcmc_run_to_convergence_get_marginal_likelihood = None
mcmc_run_to_convergence_marg_like_checks_per_conv = None
mcmc_run_to_convergence_plot_chains = None
mcmc_run_to_convergence_save_chains = None
mcmc_run_to_convergence_chains_max_taus = None
mcmc_run_to_convergence_save_samples = None
mcmc_run_to_convergence_plot_sample_histograms = None
mcmc_save_samples_max = None
mcmc_skip_initial_state_check = None
mcmc_N_walkers = None
mcmc_N_steps_max = None
mcmc_N_steps_restart = None
mcmc_initial_bundle_fractional_width = None
mcmc_burnin_drop = None
mcmc_thin_by = None

#=== Marginal likelihood calculation
ml_plot_only_GD = None
ml_fac_to_make_smooth = None
ml_fac_to_reduce_sigma = None

#
#=== Info on parameters
#
par_names_print = None
par_names_plot = None
par_names_mcmc = None
par_names_print_short = None
par_names_plot_short = None
par_names_mcmc_short = None
#  important (non-nuisance) parameters
ind_important_pars = None
imp_par_names_print = None
imp_par_names_plot = None
imp_par_names_mcmc = None
N_imp_pars = None
#  discrete parameters
ind_discrete_pars = None
discrete_par_names = None

#=== Figure/Plotting options
cornerplot_plot_maxvals = None
cornerplot_make_allplot = None
histograms_plot_maxvals = None

#=== File and output
file_chain = None
autocorr_fig_file = None
autocorr_data_file = None 
marglike_fig_file = None
marglike_data_file = None
chains_fig_file = None
chains_data_file = None 
samples_fig_file = None
samples_hist_file = None
samples_data_file = None
samples_logprob_data_file = None
samples_loglike_data_file = None
samples_logprior_data_file = None
cornerplotall_fig_file = None
cornerplotimp_fig_file = None 



#==============================================================
#            
#      Parameters used by the model-hypothesis module
#
#==============================================================
#
#=== Number of individuals and individual names
#     (to be provided by model hypothesis call to data module)
Nindiv = None
indiv_names = None
#
#
#=== Some data module parameter objects
#      ( mostly used for plotting model over data)
#
yvals_for_indiv = None
yvals_type_for_indiv = None
evolve_models_for_indiv = None
evolve_models_xvals_for_indiv = None
datasets_for_indiv = None
xindices_for_indiv = None
#=== Descriptive names of parameters
#
descrip_name = {}
#
#=== Priors
#
#    Prior density types ("prior_dist") can be:
#
#      normal(real pars): mean, stdev
#      lognormal (pos pars): mean, stdev (see below)
#      uniform (any pars): min, max
#
#    Each of the dictionaries below will have an entry for each parameter
#    in the [indiv, pop, other] dictionaries (so the keys in those
#    dictionaries must be distinct).
#
prior_dist = {}
prior_mean = {}
prior_stdev = {}
prior_min = {}
prior_max = {}
#
# For lognormal priors, the "width factor" specifies the width
# of the distribution, i.e., 
#
#    log(X) ~ N(mean, width^2)  w/  width = log(widthfac)
#
# means that the 68% region of the normal distribution over
# log(X) is:
#
#        log(mean / widthfac) < log(X) < log(mean * widthfac)
#
# When no further information is known, this might be a good
# value for "width factor"
#
#=== Parameters transformation
#
#    For evolution with MCMC, parameters can be transformed to preserve
#    bounds.  Possible transformation types (with respect to the
#    untransformed parameters stored in the [indiv, pop, other] dictionaries
#    are:
#
#         'none'
#         'log'  (pos pars)
#
#    They can also be transformed or not (independently) for printing or
#    plotting.
#
#    Note: For model hypotheses requiring transformations involving
#    multiple variables --- e.g., for "w/cov", where we mcmc-evolve the
#    matrix elements [ell, log(D1), log(D2)], but are interested in
#    [var_m, var_b, cov_mb] --- some special manipulation of the functions
#    for moving parameters back and forth from mcmc is required (see below).
#
#    Each of the dictionaries below has an entry for each parameter
#    in the [indiv_nopop, pop, other] dictionaries (so the keys in those
#    dictionaries must be distinct).
#
transf_mcmc = {}
transf_plot = {}
transf_print = {}
#
#=== INDIV_POP (Individual parameters controlled by population distribution)
#
#    Each indiv_pop parameter will be given a conditional prior with
#    respect to the population distribution.  For each parameter in
#    an individuals dictionary here, there are two associated "pop"
#    parameters governing the population distribution. So, we do not
#    set any explicit prior information for these parameters (in
#    prior_dist, prior_mean, etc).
#
#    The transformations of these parameters (for transf_mcmc,
#    transf_plot, etc) are set by their associated POP parameters below.
#
#    If there are parameters associated to individuals that are not
#    governed by a population distribution, they should appear in
#    the indiv_nopop dictionary instead.
#
#    In this model hypothesis the indiv_pop pars for each individual are:
#
#             m (slope) and b (intercept)
#
# Use lists of parameters to preserve ordering when writing to
# the numpy array used by MCMC, particularly for the dictionaries below.
list_indiv_pop_pars = None
# for the indiv_pop and indiv_nopop pars, create a pandas dataframe
# with individual names as indices
indiv_pop = None
#
#=== INDIV_NOPOP (Individual parameters not distributed by population)
#
#    For this model hypothesis, the indiv_nopop pars for each individual are:
#
#                        <none>
#
list_indiv_nopop_pars = None
indiv_nopop = None
#
#    If there were parameters in indiv_nopop, we would also define
#    elements of these dictionaries for each parameter
#
#        [descrip_name,
#         prior_dist, prior_mean, prior_min, prior_max,
#         tranf_mcmc, tranf_plot, transf_print]
#
#=== POP (Random effects population distribution parameters)
#
#    Population parameters can come from these possible
#    two-parameter distributions:
#
#      normal (real pars): mean, stdev
#      lognormal (pos parameters): mean, stdev (log(X) ~ N(mean, stdev^2))
#      gamma (pos pars): mean (k*theta), stdev (sqrt[k*theta^2])
#
pop_dist = {}
pop = {}
#
#    Dictionaries should be specified with the key-naming
#    convention shown below.  For every list_indiv_pop_pars
#    parameter there should be a par_mean and par_stdev.
#
#
#=== OTHER (other pars, e.g., those associated to error-model)
#
#    In this model hypothesis we assume:
#
#            'sigma': common error stdev for all individuals
#
#    With prior:
#
#            'sigma': log-normally-distributed
#
#    And transformation for MCMC:
#
#            'sigma': log
#
list_other_pars = None
other = {}

#==============================================================
#            
#      Methods used by the model-hypothesis module
#
#==============================================================

def get_normal_loglike(yvals, ymod, var):
    return -0.5 * np.sum( (yvals - ymod)**2 / var + np.log(2.0 * np.pi * var) )

def get_lognormal_loglike(yvals, ymod, var):
    # exclude zero values
    good = ( (yvals > 0) & (ymod > 0) )
    return -0.5 * np.sum( (np.log(yvals[good]) - np.log(ymod[good]) )**2
                          / var + np.log(2.0 * np.pi * var) )

def get_poisson_loglike(yvals, ymod):
    #
    # Use special degenerate poisson if model (lambda par) is zero
    #
    #              /  1   (i=0)
    #    pmf(i) = {
    #              \  0   (otherwise)
    #
    # which follows from 0^0 = 1
    #
    zero = (ymod == 0)
    if np.any(yvals[zero] > 0):
        return -np.inf
    #
    # Otherwise, use normal if model values are large
    #
    retval = 0.0
    large = (ymod > 1000)
    retval += get_normal_loglike(yvals[large], ymod[large], ymod[large])
    #
    # and use regular poisson when model values are smaller
    #
    #    pmf(i) = exp(-lambda) * lambda^i / i!
    #
    # where we can write i! = gamma(i+1)
    #
    small = ~large & ~zero 
    yv = np.round(yvals[small]).astype(int)
    retval += np.sum( yv * np.log(ymod[small]) - ymod[small] - spsp.gammaln(yv + 1) )
    # return the result
    return retval

def print_list_w_commas(thelist):
    first=True
    for p in thelist:
        if first:
            print(p, end='')
            first=False
        else:
            print(", " + p, end='')

def print_mixed_model_pars(indentstr="", dividerstr=""):
    """
    Called by the model hypothesis module to print out the pars
    """
    # individual parameters (controlled by pop dist)
    print(indentstr + "Individual-specific parameters controlled by pop dist are:")
    if (len(indiv_pop) == 0):
        print(2*indentstr + "<none>", end='')
    else:
        for p in indiv_pop:
            print(2*indentstr + descrip_name[p] + ", " + p)
        print(indentstr + "with initial values for each individual:", end='')
        for index, row in indiv_pop.iterrows():
            print("\n" + 2*indentstr + index + ": ", end='')
            for c in indiv_pop:
                print(c + " = " + str(indiv_pop.at[index, c]) + ", ", end='')
    # individual parameters (not controlled by pop dist)
    print("\n" + indentstr + "Individual-specific parameters NOT controlled by pop dist are:")
    if (len(indiv_nopop) == 0):
        print(2*indentstr + "<none>", end='')
    else:
        for p in indiv_nopop:
            print(3*indentstr + descrip_name[p] + ", " + p)
        print(indentstr + "with initial values for each individual:", end='')
        for index, row in indiv_nopop.iterrows():
            print("\n" + 3*indentstr + index + ": ", end='')
            for c in indiv_nopop:
                print(c + " = " + str(indiv_nopop.at[index, c]) + ", ", end='')
            print("\n")
        # prior values
        for p in indiv_nopop:
            print(3*indentstr + "prior dist: " + prior_dist[p] + " w/")
            if (prior_dist[p] in ['normal', 'lognormal']):
                print(4*indentstr + p + "(prior_mean, prior_stdev) = ", end='')
                print(f"({prior_mean[p]:.3e}, {prior_stdev[p]:.3e})")
            elif (prior_dist[p] == 'poisson'):
                print(4*indentstr + p + "prior_mean = ", end='')
                print(f"{prior_mean[p]:.3e}")
            elif (prior_dist[p] == 'uniform'):
                print(4*indentstr + p + "(prior_min, prior_max) = ", end='')
                print(f"({prior_min[p]:.3e}, {prior_max[p]:.3e})")
            # transformation maps
            print(3*indentstr + "transformations:")
            print(4*indentstr + p + "[mcmc-transformation]: ", end='')
            print(transf_mcmc[p])
            print(4*indentstr + p + "[print-transformation]: ", end='')
            print(transf_print[p])
            print(4*indentstr + p + "[plot-transformation]: ", end='')
            print(transf_plot[p])
    # population parameters
    print("\n" + indentstr + "Parameters governing the population distributions:")
    if (len(indiv_pop) == 0):
        print(2*indentstr + "<none>")
    else:
        for p in pop_dist:
            # par name and initial value
            print(2*indentstr + descrip_name[p] + ", " + p + ", from a two-parameter "
                  + pop_dist[p] + " distribution")
            print(3*indentstr + "parameters, with initial values:")
            print(4*indentstr + p + '_mean' + f" = {pop[p+'_mean']:.3e}")
            print(4*indentstr + p + '_stdev' + f" = {pop[p+'_stdev']:.3e}")   
            print(3*indentstr + "prior specification:")
            # print info for mean, then for stdev 
            for s in ['_mean', '_stdev']:
                # prior values
                print(4*indentstr + p + s + " prior distribution: " + prior_dist[p+s])
                if (prior_dist[p+s] in ['normal', 'lognormal']):
                    print(4*indentstr + p + s + "(prior_mean, prior_stdev) = ", end='')
                    print(f"({prior_mean[p+s]:.3e}, {prior_stdev[p+s]:.3e})")
                elif (prior_dist[p+s] == 'poisson'):
                    print(4*indentstr + p + s + "prior_mean = ", end='')
                    print(f"{prior_mean[p+s]:.3e}")
                elif (prior_dist[p+s] == 'uniform'):
                    print(4*indentstr + p + s + "(prior_min, prior_max) = ", end='')
                    print(f"({prior_min[p+s]:.3e}, {prior_max[p+s]:.3e})")
            print(3*indentstr + "transformations of parameters:")
            for s in ['_mean', '_stdev']:                
                # transformation maps
                print(4*indentstr + p + s + "[mcmc-transformation]: ", end='')
                print(transf_mcmc[p+s])
                print(4*indentstr + p + s + "[print-transformation]: ", end='')
                print(transf_print[p+s])
                print(4*indentstr + p + s + "[plot-transformation]: ", end='')
                print(transf_plot[p+s])
    # other parameters
    print(indentstr + "Other parameters")
    for p in other:
        print(2*indentstr + descrip_name[p] + ", " + p)
        # name and initial value
        print(3*indentstr + "initial value")
        print(4*indentstr + p + " = " + str(other[p]))
        # prior values
        print(3*indentstr + "prior dist: " + prior_dist[p] + " w/")
        if (prior_dist[p] in ['normal', 'lognormal']):
            print(4*indentstr + p + "(prior_mean, prior_stdev) = ", end='')
            print(f"({prior_mean[p]:.3e}, {prior_stdev[p]:.3e})")
        elif (prior_dist[p] == 'poisson'):
            print(4*indentstr + p + "prior_mean = ", end='')
            print(f"{prior_mean[p]:.3e}")
        elif (prior_dist[p] == 'uniform'):
            print(4*indentstr + p + "(prior_min, prior_max) = ", end='')
            print(f"({prior_min[p]:.3e}, {prior_max[p]:.3e})")
        # transformation maps
        print(3*indentstr + "transformations:")
        print(4*indentstr + p + "[mcmc-transformation]: ", end='')
        print(transf_mcmc[p])
        print(4*indentstr + p + "[print-transformation]: ", end='')
        print(transf_print[p])
        print(4*indentstr + p + "[plot-transformation]: ", end='')
        print(transf_plot[p])
    # summary of parameters
    num_nan_pop = int(indiv_pop.isna().sum().sum())
    num_nan_nopop = int(indiv_nopop.isna().sum().sum())
    num_mcmc_pars = Nindiv*(len(list_indiv_pop_pars) + len(list_indiv_nopop_pars)) + len(pop) \
        + len(other) - num_nan_pop - num_nan_nopop
    print(indentstr + f"In summary, there are {num_mcmc_pars:d} parameters for MCMC evolution:")
    print(2*indentstr + f"{Nindiv*len(list_indiv_pop_pars):d} indiv_pop parameters: (", end='')
    print_list_w_commas(list_indiv_pop_pars)
    print(") * " + f"{Nindiv:d} individuals")
    print(3*indentstr + f"[minus {num_nan_pop:d} parameters not set (\"nan\") for some individuals]")
    print(2*indentstr + f"{Nindiv*len(list_indiv_nopop_pars):d} indiv_nopop parameters: (", end='')
    print_list_w_commas(list_indiv_nopop_pars)
    print(") * " + f"{Nindiv:d} individuals")
    print(3*indentstr + f"[minus {num_nan_nopop:d} parameters not set (\"nan\") for some individuals]")    
    print(2*indentstr + f"{len(indiv_pop.columns)*2:d} pop parameters: (", end='')
    print_list_w_commas(list_indiv_pop_pars)
    print(") * " + "2 distribution pars")
    print(2*indentstr + f"{len(other):d} other parameters: (", end='')
    print_list_w_commas(other)
    print(")")
    num_prior_pars = Nindiv*len(indiv_nopop) + 2*len(pop) + 2*len(other)
    print(indentstr + f"And there are {num_prior_pars:d} fixed prior meta-parameters.")

def check_all_names_distinct():
    """
    Verify that all parameter names are distinct
    """
    i = list_indiv_nopop_pars
    p = [k for k in pop]
    o = list_other_pars
    tot = len(i) + len(p) + len(o)
    # make sure that the set of unique keys is equal to tot
    if (len(np.unique(np.concatenate((i,p,o)))) != tot):
        print("***Error: The set of all parameter keys are not distinct.")
        print("indiv:", i)
        print("pop:", p)
        print("other:", o)            
        exit(0)

def check_Nindiv_correct():
    if ( (len(list_indiv_pop_pars) > 0)
         & (len(indiv_pop) != Nindiv) ):
        print("***Error: The number of individuals in the indiv_pop dataframe ("
              + str(len(indiv_pop)) + ") is not equal to Nindiv ("
              + str(Nindiv) + ")")
        exit(0)
    if ( (len(list_indiv_nopop_pars) > 0)
         & (len(indiv_nopop) != Nindiv) ):
        print("***Error: The number of individuals in the indiv_nopop dataframe ("
              + str(len(indiv_nopop)) + ") is not equal to Nindiv ("
              + str(Nindiv) + ")")
        exit(0)
              
def get_val_w(transf_func, parname, val, reverse=False):
    """
    Return a (potentially transformed) parameter value 
    from the stored parameters
    """
    if reverse:
        if (transf_func[parname] == 'none'):
            return val
        elif (transf_func[parname] == 'log'):
            return np.exp(val)
        elif (transf_func[parname] == 'tan'):
            ymin = prior_min[parname]
            ymax = prior_max[parname]
            return (ymax-ymin) *( ( np.arctan(val) + np.pi/2 )/ np.pi ) + ymin
        else:
            print("***Error: reverse transform " + transf_func[parname] + " not found")                
            exit(0)
    else:
        if (transf_func[parname] == 'none'):
            return val
        elif (transf_func[parname] == 'log'):
            return np.log(val)
        elif (transf_func[parname] == 'tan'):
            ymin = prior_min[parname]
            ymax = prior_max[parname]
            return np.tan( (val - ymin) * np.pi/(ymax-ymin) - np.pi/2 )
        elif (transf_func[parname] == 'round'):
            return np.round(val)
        else:
            print("***Error: transform " + transf_func[parname] + " not found")                
            exit(0)

def get_name_w(transf_func, parname, i=None):
    """
    Return a (potentially transformed) parameter name
    """
    if i is None:
        pn = parname
    else:
        pn = parname.split("_")[0] + "[" + i + "]"
    if (transf_func[parname] == 'none'):
        return pn
    elif (transf_func[parname] == 'log'):
        return "log(" + pn + ")"
    elif (transf_func[parname] == 'tan'):
        return "tan(... " + pn + " ...)"
    elif (transf_func[parname] == 'round'):
        return pn
    else:
        print("***Error: transform " + transf_func[parname] + " not found")
    exit(0)

def give_vec_for(transf_func, justnames=False, shortnames=False):
    """
    Returns the numpy array of all parameters to be used by:
               
        [MCMC, printing, or plotting]
    
    Note: For model hypotheses requiring transformations 
    involving multiple variables --- e.g., for "w/cov", 
    where we evolve matrix elements [ell, D1, D2], but are 
    interested in [var_m, var_b, cov_mb] --- we will need to 
    catch and transform those variables separately.
    """
    if justnames:
        arr = []
        for p in list_indiv_pop_pars:
            ind = 0
            for i in indiv_pop.index:
                if shortnames:
                    iname = str(ind)
                else:
                    iname = i
                # only send the non-NaN values into arr
                if ~np.isnan(indiv_pop.at[i,p]):
                    arr = np.concatenate(
                        (arr, [ get_name_w(transf_func, p + "_mean", i=iname) ])
                    )
                ind += 1
        for p in list_indiv_nopop_pars:
            ind = 0
            for i in indiv_nopop.index:
                if shortnames:
                    iname = str(ind)
                else:
                    iname = i
                # only send the non-NaN values into arr                
                if ~np.isnan(indiv_nopop.at[i,p]):                
                    arr = np.concatenate(
                        (arr, [ get_name_w(transf_func, p, i=iname) ])
                    )
                ind += 1
        for p in list_indiv_pop_pars:
            arr = np.concatenate( (arr, [get_name_w(transf_func, p + "_mean")]) )
            arr = np.concatenate( (arr, [get_name_w(transf_func, p + "_stdev")]) )
        for p in list_other_pars:
            arr = np.concatenate( (arr, [get_name_w(transf_func, p)]) )
        return arr
    #
    #--- Ordering of parameters in the numpy vector sent to MCMC
    #
    #    (1) Place the INDIV_* values in the numpy array sorted by:
    #
    #          parameter-name (in order of indiv_*_pars)
    #              then individual-name (index of dataframe)
    #
    #        This way transforms can be done on the entire array
    #        of the same parameter over multiple individuals.
    #
    #    (2) Then place the POP pars in the order of list_indiv_pop_pars
    #        with par_mean first, then par_stdev.
    #
    #    (3) Then place OTHER in the order of other_pop_pars
    #
    arr = []
    ind = 0
    # INDIV_POP:  for each par, append *vectors* over individuals
    for p in list_indiv_pop_pars:
        # only send non-NaNs into arr
        vec = indiv_pop[p].to_numpy()
        vec = vec[~np.isnan(vec)]
        arr = np.concatenate(
            (arr, get_val_w(transf_func, p + '_mean', vec ) )
        )
    # INDIV_NOPOP:  for each par, append *vectors* over individuals            
    for p in list_indiv_nopop_pars:
        # only send non-NaNs into arr        
        vec = indiv_nopop[p].to_numpy()
        vec = vec[~np.isnan(vec)]
        arr = np.concatenate(
            (arr, get_val_w(transf_func, p, vec ) )
        )
    # POP: for each par, append par_mean then par_stdev
    for p in list_indiv_pop_pars:
        arr = np.concatenate( (arr, [get_val_w(transf_func, p + '_mean', pop[p + '_mean'])]) )
        arr = np.concatenate( (arr, [get_val_w(transf_func, p + '_stdev', pop[p + '_stdev'])]) )
    # OTHER: in order of list_other_pars
    for p in list_other_pars:
        arr = np.concatenate( (arr, [get_val_w(transf_func, p, other[p])]) )
    return arr

def take_vec_from_mcmc(arr):
    """
    Takes the numpy vector evolved by emcee and imports the
    parameters to their dictionaries, with transformation if
    necessary.
        
    For transformations involving multiple variables, this
    will need to be adjusted to catch those parameters and
    handle them separately. (see give_vec_for_mcmc())
    """
    #
    #=== For INDIV_* transfer full columns from the vector to df
    #

    #
    arrind = 0
    for p in list_indiv_pop_pars:
        #
        # First check for any NaNs in the dataframe
        #  (these positions with NaNs should persist... see below)
        #
        vec = indiv_pop[p].to_numpy()
        if np.any(np.isnan(vec)):
            # if there are NaNs (individuals that don't use that par)
            # then get the positions of the non-NaN for placement
            okpos = ~np.isnan(vec)
            okvals = okpos.sum()
            newarr = np.zeros(len(indiv_pop))
            newarr[okpos] = arr[arrind:(arrind+okvals)]
            # and keep the other positions NaN-valued
            newarr[~okpos] = np.nan
            # then put NaN-expanded array into dataframe
            indiv_pop[p] = get_val_w(transf_mcmc, p + '_mean', newarr, reverse=True)
            arrind += okvals
        else:
            # if no NaNs, just put that Nindiv segment of arr into dataframe
            indiv_pop[p] = get_val_w(transf_mcmc, p + '_mean',
                                     arr[arrind:(arrind+Nindiv)], reverse=True)
            arrind += Nindiv
    for p in list_indiv_nopop_pars:
        # first check for any NaNs in the dataframe
        vec = indiv_nopop[p].to_numpy()
        if np.any(np.isnan(vec)):
            # if there are NaNs (individuals that don't use that par)
            # then get the positions of the non-NaN for placement
            okpos = ~np.isnan(vec)
            okvals = okpos.sum()
            newarr = np.zeros(len(indiv_nopop))
            newarr[okpos] = arr[arrind:(arrind+okvals)]
            # and keep the other positions NaN-valued
            newarr[~okpos] = np.nan
            # then put NaN-expanded array into dataframe
            indiv_nopop[p] = get_val_w(transf_mcmc, p, newarr, reverse=True)
            arrind += okvals
        else:
            # if no NaNs, just put that Nindiv segment of arr into dataframe
            indiv_nopop[p] = get_val_w(transf_mcmc, p,
                                     arr[arrind:(arrind+Nindiv)], reverse=True)
            arrind += Nindiv
    # For POP and OTHER, transfer element by element from vec to dictionary
    for p in pop:
        pop[p] = get_val_w(transf_mcmc, p, arr[arrind], reverse=True)
        arrind += 1
    for p in other:
        other[p] = get_val_w(transf_mcmc, p, arr[arrind], reverse=True)
        arrind += 1

def get_normal_logprior(pr_mean, pr_var, val):
    return -0.5 * np.sum( (val - pr_mean)**2 / pr_var + np.log(2*np.pi*pr_var) )

def get_poisson_logprior(pr_mean, val):
    if (pr_mean > 1000):
        # for large values, just use normal
        return get_normal_logprior(pr_mean, pr_mean, val)
    if hasattr(val, "__len__"):
        retval = 0.0
        v = np.round(val).astype(int)
        #
        # Negative values have zero probability
        #
        if np.any(np.array(val) < 0):
            return -np.inf
        #
        # Otherwise, poisson:
        #
        #     f(i) = e^(-lambda) * lambda^i / i!
        #
        #     -> log(f) = - lambda + i * log(lambda) - log(i!)
        #
        # where
        #
        #     i! = gamma(i + 1)
        #
        return np.sum( v * np.log(pr_mean) - pr_mean - spsp.gammaln(v + 1) )
    else:
        if val < 0:
            return -np.inf
        return ( -pr_mean + np.round(val)*np.log(pr_mean)
                 - spsp.gammaln(np.round(val) + 1) )
    
def get_uniform_logprior(pr_min, pr_max, val):
    if ( np.all(val > pr_min) & np.all(val < pr_max) ):
        return 0.0
    else:
        return -np.inf

def get_log_prior():
    log_prior = 0.0
    # get conditional prior values of INDIV_POP pars wrt population dist
    for p in list_indiv_pop_pars:
        # get the non-NaN values only
        vec = indiv_pop[p].to_numpy()
        vec = vec[~np.isnan(vec)]
        if (pop_dist[p] == 'normal'):
            log_prior += get_normal_logprior(pop[p+'_mean'], pop[p+'_stdev']**2, vec)
        elif (pop_dist[p] == 'lognormal'):
            log_prior += get_normal_logprior(np.log(pop[p+'_mean']),
                                             pop[p+'_stdev']**2, np.log(vec))
    # get prior values of INDIV_NOPOP pars
    for p in list_indiv_nopop_pars:
        # get the non-NaN values only
        vec = indiv_nopop[p].to_numpy()
        vec = vec[~np.isnan(vec)]
        if (prior_dist[p] == 'normal'):
            log_prior += get_normal_logprior(prior_mean[p], prior_stdev[p]**2, vec)
        elif (prior_dist[p] == 'lognormal'):
            log_prior += get_normal_logprior(np.log(prior_mean[p]),
                                             prior_stdev[p]**2, np.log(vec))
        elif (prior_dist[p] == 'poisson'):
            log_prior += get_poisson_logprior(prior_mean[p], vec)
        elif ( prior_dist[p] == 'uniform'):
            log_prior += get_uniform_logprior(prior_min[p], prior_max[p], vec)
    # get prior values of POP parameters
    for p in list_indiv_pop_pars:
        for s in ['_mean', '_stdev']:
            if (prior_dist[p+s] == 'normal'):
                log_prior += get_normal_logprior(prior_mean[p+s],
                                                 prior_stdev[p+s]**2, pop[p+s])
            elif (prior_dist[p+s] == 'lognormal'):
                log_prior += get_normal_logprior(np.log(prior_mean[p+s]),
                                                 prior_stdev[p+s]**2,
                                                 np.log(pop[p+s]))
            elif (prior_dist[p+s] == 'poisson'):
                log_prior += get_poisson_logprior(prior_mean[p+s], pop[p+s])
            elif (prior_dist[p+s] == 'uniform'):
                log_prior += get_uniform_logprior(prior_min[p+s],
                                                   prior_max[p+s], pop[p+s])
    # get prior values of OTHER parameters
    for p in list_other_pars:
        if (prior_dist[p] == 'normal'):
            log_prior += get_normal_logprior(prior_mean[p], prior_stdev[p]**2,
                                             other[p])
        elif (prior_dist[p] == 'lognormal'):
            log_prior += get_normal_logprior(np.log(prior_mean[p]),
                                             prior_stdev[p]**2,
                                             np.log(other[p]))
        elif (prior_dist[p] == 'poisson'):
            log_prior += get_poisson_logprior(prior_mean[p], other[p])
        elif (prior_dist[p] == 'uniform'):
            log_prior += get_uniform_logprior(prior_min[p], prior_max[p], other[p])
    return log_prior

def get_CI_vec(ymodels, smooth=False):
    """
    Return [2.5, 16, 50, 84, 97.5] percentiles
    """
    Npoints = len(ymodels[0])
    y = np.zeros([5, Npoints])
    pctile_values = [2.5, 16.0, 50.0, 84.0, 97.5]
    for i in range(Npoints):
        pctiles = np.percentile(ymodels[:,i], pctile_values)
        for j in range(5):
            y[j, i] = pctiles[j]
    return y

def get_evolvemodels_for_plotting(ppars, em_module):
    plot_pars = ppars.copy()
    #
    #--- Load the sample data
    #
    samples = np.loadtxt(plot_pars['sample_file'])
    if plot_pars['Nsamp'] is not None:
        samples = samples[0:plot_pars['Nsamp']]
    #
    #
    #--- For each sample, get the evolve-model for all datasets
    #
    # Info on samples
    Nsamp, Npars = samples.shape
    # Points per function for each evolve-model dataset
    Npoints = plot_pars['Npoints']  
    # Create dictionary [i][em][ds] to place the model evolutions
    xmodels = {}
    ymodels = {} 
    for i in indiv_names:
        ymodels[i] = {}
        xmodels[i] = {}
        for em in evolve_models_for_indiv[i]:
            ymodels[i][em] = {}
            xmodels[i][em] = {}
            for ds in datasets_for_indiv[i][em]:
                # to be a list of np arrays, one for each sample
                ymodels[i][em][ds] = [] 
    # loop over samples and run the evolve-model
    print("Evolving models in sample: ")
    for s in range(Nsamp):
        # put theta vector into the mm par objects
        take_vec_from_mcmc(samples[s])
        # send the parameters to the evolve-model
        em_module.set_pars(indiv_pop, indiv_nopop, other)            
        # loop over individuals, evolve-models, datasets and get model values
        if ( ((s % 100) == 0) & (s != 0) ):
            print(str(s))
        elif ( (s % 10) == 0 ):
            print(str(s) + " ", end='')            
        for i in indiv_names:
            for em in evolve_models_for_indiv[i]:
                # create a smooth function (with Npoints)
                x = evolve_models_xvals_for_indiv[i][em]
                xmodels[i][em] = np.linspace( x.min(), x.max(), num=Npoints)
                ymod = em_module.get_evolve_model_solutions(i, em, xmodels[i][em])
                for ds in datasets_for_indiv[i][em]:
                    # add the evolve-model solution for dataset to the list
                    ymodels[i][em][ds].append(ymod[ds])
    print("\n", end='')
    plot_pars['ymodels'] = ymodels
    plot_pars['xmodels'] = xmodels        
    #
    #--- Get 68 and 95 CI median lines for each indiv, ev-model, dataset
    #
    CIbounds = {} # dictionary [i][em][ds] to place the CI bounds fcns
    for i in indiv_names:
        CIbounds[i] = {}
        for em in evolve_models_for_indiv[i]:
            CIbounds[i][em] = {}
            for ds in datasets_for_indiv[i][em]:
                # returns a 5-row, Npoint-column np array where the rows
                # are the [2.5, 16.0, 50.0, 84.0, 97.5] percentile functions
                CIbounds[i][em][ds] = get_CI_vec(np.array(ymodels[i][em][ds]))
    plot_pars['CIbounds'] = CIbounds
    return plot_pars

#==============================================================
#            
#      Methods used by mixed-effects_mcmc.py
#
#==============================================================

def create_backend():
    #
    #  I tried to do this file handling to create backups and make new
    #  files, but the new files did not actually reset the size.  So
    #  I still had 8GB files that only had a few new iterations inside.
    #
    #  *** I should try to make a minimal example file and post a
    #      question to the emcee github page ***
    #
    #  But for now I'm just going to do file handling manually:
    #
    #     - make a new backend file for each run (unique timestamp)
    #
    #     - delete files frequently from "chains" directory (or
    #       wait for computer to freeze up... I currently only have
    #       28GB available on my macbook... luckily I have about
    #       1.5TB on maria!)
    #  (see prior versions and search for "reset_chainfile")
    backend = emcee.backends.HDFBackend(file_chain)
    return backend

def get_maximum_a_posteriori_estimate(negloglik, initial_theta):
    # perform the minimization
    if (minimize_method == 'avg'):
        # do a bespoke averaging over all methods
        sols = []
        for m in ['BFGS', 'Nelder-Mead', 'TNC', 'COBYLA']:
            s = spo.minimize(negloglik, initial_theta, tol=minimize_tol, method=m)
            sols.append(s.x)
        sols = np.array(sols)
        theta_ml = np.mean(sols, axis=0)
    else:
        soln = spo.minimize(negloglik, initial_theta, tol=minimize_tol, method=minimize_method)
        theta_ml = soln.x
    return theta_ml

def print_maximum_a_posteriori_estimate(theta_transformed):
    if do_maximum_likelihood:
        print("Maximum likelihood estimate (using minimize_method = " + minimize_method + "): ")
    else:
        print("Maximum posterior estimates (using minimize_method = " + minimize_method + "): ")
    for i in range(len(par_names_print)):
        print('\t' + par_names_print[i] + f" = {theta_transformed[i]:0.4e}")

def log_posterior_probability(theta):
    """ 
    Get the posterior density from the model hypothesis module
       (up to the marginal likelihood normalizing constant)
    """
    # evaluate likelihood and prior
    ll, lp = mod_log_like_and_prior(theta)
    # update (2022-05-26): add "blobs" for likelihood and prior individually
    return lp + ll, ll, lp

def print_percentiles(flat_samples_transformed, pctile_values):
    print("The " + str(pctile_values) + " percentiles are:")
    for i in range(len(par_names_print)):
        pctiles = np.percentile(flat_samples_transformed[:,i], pctile_values)
        print('\t' + par_names_print[i] + \
              f" = {pctiles[1]:0.4e} [{pctiles[0]:0.4e}:{pctiles[2]:0.4e}]")

def get_mcmc_N_steps(restart):
    if restart:
        return mcmc_N_steps_restart
    else:
        return mcmc_N_steps_max
    
def get_init_pos(pars_map):
    n_dim = len(pars_map)
    # tight bundle about max a posteriori
    pos = ( pars_map + np.abs(pars_map)*mcmc_initial_bundle_fractional_width
            * np.random.randn(mcmc_N_walkers, n_dim))
    # except for discrete parameters which get uniform spread over prior
    for i in range(len(ind_discrete_pars)):
        # if par name is for individual it will be saved as par[indivname]
        short_name = discrete_par_names[i].split('[')[0]
        pos[:,ind_discrete_pars[i]] = np.random.uniform(low=prior_min[short_name],
                                                        high=prior_max[short_name],
                                                        size=mcmc_N_walkers)
    return pos, n_dim

def print_mcmc_convergence_step(Nsteps, tau_max, tau_frac_max):
    time_elapsed = dt.datetime.now() - start_time
    print("\n[" + str(time_elapsed)
          + f"]\n  Checking convergence at ({Nsteps:d}) steps:")
    outstring = f"tau_max * N_autocorrs = {tau_max:.1f} * "
    outstring += f"{mcmc_run_to_convergence_N_autocorrs:d} = "
    outstring += f"{tau_max*mcmc_run_to_convergence_N_autocorrs:.0f}  ?< {Nsteps:d}"
    print("\t" + outstring)
    outstring = f"tau_frac = abs(tau[i-1] - tau[i]) / tau[i] = "
    outstring += f"{tau_frac_max:.4f} ?< {mcmc_run_to_convergence_delta_tau_frac}"
    print("\t" + outstring)

def check_sampler_convergence(Nsteps, tau, old_tau):
    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau_max = np.max(tau)
    tau_min = np.min(tau)
    tau_mean = np.mean(tau)
    tau_frac_max = np.max(np.abs(old_tau - tau)/tau)
    # Check convergence
    converged = np.all(tau * mcmc_run_to_convergence_N_autocorrs < Nsteps)
    converged &= np.all(np.abs(old_tau - tau) / tau < mcmc_run_to_convergence_delta_tau_frac)
    old_tau = tau
    return tau_min, tau_max, tau_mean, tau_frac_max, converged, old_tau

def plot_and_save_autocorrelation(index, tau_v_steps_min, tau_v_steps_max,
                                  tau_v_steps_mean):
    # Plot evolution of autocorrelation times
    n_steps = mcmc_run_to_convergence_check_steps * np.arange(1, index + 2)
    t_autocorr_min = np.concatenate( ([0], tau_v_steps_min[:index+1]) )
    t_autocorr_mean = np.concatenate( ([0], tau_v_steps_mean[:index+1]) )
    t_autocorr_max = np.concatenate( ([0], tau_v_steps_max[:index+1]) )
    n_steps = np.concatenate(([0],n_steps))
    plt.close('all')
    fig, ax = plt.subplots()    
    ax.plot(n_steps, n_steps / mcmc_run_to_convergence_N_autocorrs, "--k")
    ax.plot(n_steps, t_autocorr_min, '-bo')
    ax.plot(n_steps, t_autocorr_mean, '-ko')
    ax.plot(n_steps, t_autocorr_max, '-ro')            
    ax.set_xlim(0, n_steps.max())
    ax.set_ylim(0, t_autocorr_max.max()
                + 0.1 * (t_autocorr_max.max() - t_autocorr_min.min()))
    ax.set_xlabel("number of steps")
    ax.set_ylabel(r"$\tau_{\rm max,\,mean,\,min}$")
    #print("  Saving autocorrelation figure to:\n\t" + autocorr_fig_file)
    #print("  Saving autocorrelation figure")
    fig.savefig(autocorr_fig_file)
    # save data to file
    np.savetxt(autocorr_data_file,
               np.array([n_steps, t_autocorr_min,
                         t_autocorr_mean, t_autocorr_max]).T)
                
def conv_check_get_marginal_likelihood(flat_samples, flat_loglike, flat_logprior):
    mvn = sps.multivariate_normal(np.mean(flat_samples, axis=0),
                                  np.cov(flat_samples, rowvar=0)/
                                  ml_fac_to_reduce_sigma**2)
    hm, gd = calc_marg_like(flat_samples, flat_loglike, flat_logprior,
                            mvn, convergence_check=True)
    return hm, gd

def plot_and_save_chains(sampler, tau_max):
    """
    Plot the chains of each "important" parameter
    """
    # Get the chains for the last few autocorr times of the chain
    Npoints = int(mcmc_run_to_convergence_chains_max_taus * tau_max)
    samples = sampler.get_chain()[-Npoints:] # [npoints, nwalkers, npars]
    #=== make a multi-page PDF figure showing chains of all parameters
    plt.close('all')    
    figx = 8
    figy = 10.5
    plots_per_page = 5 # we'll do 5 rows
    nrows = plots_per_page
    ncols = 1
    npars = len(samples[0,0,:])
    npages = int(np.ceil(npars/plots_per_page))
    # make a set of figure labels
    figure_nums = np.arange(npages)
    figs = []
    axs = []
    for n in range(npages):
        f, a = plt.subplots(nrows, ncols, figsize = (figx, figy), num = n)
        figs.append(f)
        axs.append(a)
    # plot chains
    for p in range(npars):
        page = p // plots_per_page
        r = p % plots_per_page
        plt.figure(page)
        ax = axs[page][r]
        ax.plot(samples[:, :, p], "k", alpha=0.3)        
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(par_names_mcmc_short[p])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    for n in range(npages):
        plt.figure(n)
        axs[n][-1].set_xlabel(f"step number (of last {Npoints:d})")
    # put into multi-page pdf
    with mplbp.PdfPages(chains_fig_file) as pdf:
        for n in range(npages):
            plt.figure(n)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"  Plotted the last {Npoints:d} of the chains.")
    # save chains data to file
    if mcmc_run_to_convergence_save_chains:
        print(f"  Saved the last {Npoints:d} of the chain to file.")    
        np.save(chains_data_file, samples)

def plot_and_save_samples(sampler, tau_max):
    #=== get independent samples (and the blob values
    burnin, thin = get_burn_thin(tau_max)
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
    flat_blobs = sampler.get_blobs(discard=burnin, thin=thin, flat=True)
    flat_log_like = flat_blobs["log_like"]
    flat_log_prior = flat_blobs["log_prior"]
    nsamps = len(flat_samples)
    # but keep only the last few of them (~1000)
    if ( nsamps > mcmc_save_samples_max ):
        Npoints = mcmc_save_samples_max
    else:
        Npoints = nsamps
    #=== make a multi-page PDF figure showing samples of all parameters
    plt.close('all')    
    figx = 8
    figy = 10.5
    plots_per_page = 5 # we'll do 5 rows
    nrows = plots_per_page
    ncols = 1
    npars = len(flat_samples[0])
    npages = int(np.ceil(npars/plots_per_page))
    # make a set of figure labels
    figure_nums = np.arange(npages)
    figs = []
    axs = []
    for n in range(npages):
        f, a = plt.subplots(nrows, ncols, figsize = (figx, figy), num = n)
        figs.append(f)
        axs.append(a)
    # plot samples
    for p in range(npars):
        page = p // plots_per_page
        r = p % plots_per_page
        plt.figure(page)
        ax = axs[page][r]
        ax.scatter(np.arange(0, Npoints), flat_samples[-1*Npoints:,p], s=0.5)
        ax.set_xlim(0, Npoints)
        ax.set_ylabel(par_names_mcmc_short[p])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    for n in range(npages):
        plt.figure(n)
        axs[n][-1].set_xlabel(f"steps / thin (last {Npoints:d} samples)")
    # put into multi-page pdf
    with mplbp.PdfPages(samples_fig_file) as pdf:
        for n in range(npages):
            plt.figure(n)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"  Plotted and saved the last {Npoints:d} samples (and blobs).")
    #=== make a multi-page PDF figure showing histograms of samples of all parameters
    plt.close('all')    
    figx = 8
    figy = 10.5
    plots_per_page = 4 # we'll do 2 rows x 2 cols
    nrows = 2
    ncols = 2
    plots_per_page = nrows * ncols
    npages = int(np.ceil(npars/plots_per_page))
    # make a set of figure labels
    figure_nums = np.arange(npages)
    figs = []
    axs = []
    for n in range(npages):
        f, a = plt.subplots(nrows, ncols, figsize = (figx, figy), num = n)
        figs.append(f)
        axs.append(a)
    # plot samples
    for p in range(npars):
        page = p // plots_per_page
        r = (p % plots_per_page) // nrows
        c = (p % plots_per_page) % nrows
        plt.figure(figs[page])
        ax = axs[page][r,c]
        sns.histplot(data = flat_samples[-1*Npoints:, p], ax=ax)
        ax.set_xlabel(par_names_mcmc_short[p])
    # put into multi-page pdf
    with mplbp.PdfPages(samples_hist_file) as pdf:
        for p in figs:
            plt.figure(p)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print(f"  Plotted and saved the last {Npoints:d} samples (and blobs).")
    #=== Save the sample data to file
    np.savetxt(samples_data_file, flat_samples[-1*Npoints:,:])
    np.savetxt(samples_logprob_data_file, flat_log_prob[-1*Npoints:])
    np.savetxt(samples_loglike_data_file, flat_log_like[-1*Npoints:])
    np.savetxt(samples_logprior_data_file, flat_log_prior[-1*Npoints:])    
        
def plot_and_save_marginal_likelihood(n_steps, lml_hm, lml_gd):
    # Plot log(marg_like) vs steps
    plt.close('all')
    figml, axml = plt.subplots(figsize=(10,8))
    if ml_plot_only_GD:
        axml.plot(n_steps, lml_gd, '-bo')
        axml.set_ylabel("logML_gd (b)")
        maxval = np.max(lml_gd)
        minval = np.max(lml_gd)
    else:
        axml.plot(n_steps, lml_hm, '-ro')
        axml.plot(n_steps, lml_gd, '-bo')
        axml.set_ylabel("logML_hm (r), logML_gd (b)")
        maxval = np.concatenate((lml_hm, lml_gd)).max()
        minval = np.concatenate((lml_hm, lml_gd)).min()           
    axml.set_xlim(0, n_steps[-1])
    axml.set_ylim(minval - 0.1 * (maxval - minval), maxval + 0.1 * (maxval - minval))
    axml.yaxis.get_ticklocs(minor=True)
    axml.minorticks_on()
    axml.grid(True, which='both')
    axml.set_xlabel("number of steps")
    # save figure file
    #print("  Saving marginal likelihood figure to:\n\t" + marglike_fig_file)
    print("  Saved marginal likelihood figure.")
    figml.savefig(marglike_fig_file)
    # save data to file
    np.savetxt(marglike_data_file, np.array([n_steps, lml_hm, lml_gd]).T)
    
def get_burn_thin(tau_max):
    thin = int(mcmc_thin_by * tau_max)
    randoff = np.random.randint(low=0, high=thin-1)
    burnin = int(mcmc_burnin_drop * tau_max + randoff)
    return burnin, thin

def sample_to_convergence(sampler, init_pos, restart=False):
    """
    Adapted from:
          https://emcee.readthedocs.io/en/stable/tutorials/monitor/
    """
    if restart:
        # need to explicitly get current position to get sample() to work on restart
        #      https://github.com/dfm/emcee/issues/305
        init_pos = sampler.get_last_sample()
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    old_taus = np.inf
    first_tau_checked = False
    # keeping the autocorrelation data
    autocorr_min = np.empty(mcmc_N_steps_max)
    autocorr_max = np.empty(mcmc_N_steps_max)
    autocorr_mean = np.empty(mcmc_N_steps_max)
    # and the marginal likelihood data
    log_marg_like_hm = []
    log_marg_like_gd = []
    n_steps_ml = []
    # Sample up to mcmc_N_steps_max steps, stopping at convergence
    #   [see check_sampler_convergence()... approximately:
    #        (tau*55 < Nsteps)  &  (tau well measured) ]
    for sample in sampler.sample(init_pos, iterations=mcmc_N_steps_max,
                                 progress=mcmc_show_progress,
                                 skip_initial_state_check = mcmc_skip_initial_state_check):
        Nsteps = sampler.iteration
        if Nsteps % mcmc_run_to_convergence_check_steps:
            margsteps = mcmc_run_to_convergence_check_steps \
                / mcmc_run_to_convergence_marg_like_checks_per_conv
            if ( (Nsteps % margsteps) == 0 ):
                
                if ( mcmc_run_to_convergence_get_marginal_likelihood
                     & first_tau_checked ):
                    #--- 10 times every convergence check: do marginal likelihood check
                    burnin, thin = get_burn_thin(np.max(old_taus))
                    if (Nsteps > 2*burnin):
                        flat_samples = sampler.get_chain(discard=burnin,
                                                         thin=thin, flat=True)
                        flat_blobs = sampler.get_blobs(discard=burnin,
                                                       thin=thin, flat=True)
                        flat_loglike = flat_blobs["log_like"]
                        flat_logprior = flat_blobs["log_prior"]
                        # calculate and print ML to user
                        print(f"  ({Nsteps:d})", end="")
                        hm, gd = conv_check_get_marginal_likelihood(flat_samples,
                                                                    flat_loglike,
                                                                    flat_logprior)
                        log_marg_like_hm.append(hm)
                        log_marg_like_gd.append(gd)
                        n_steps_ml.append(Nsteps)
                        # plot marginal likelihood vs steps; save plot and data
                        plot_and_save_marginal_likelihood(n_steps_ml, log_marg_like_hm, log_marg_like_gd)
                elif mcmc_run_to_convergence_verbose:
                    # print out the acceptance frac and step prior to first marg like
                    af = sampler.acceptance_fraction
                    print(f"  ({Nsteps:d}) Current average acceptance fraction = {np.mean(af):.3f}")                    
            # Don't check convergence until "check_steps" is reached
            continue
        # print a newline for the steps printout
        print("")
        #--- Every mcmc_run_to_convergence_check_steps steps:
        #
        #   * check convergence
        #   * get the autocorrelation stats and make plot
        #   * get marginal likelihood (as above)
        #   * plot chains
        #   * save a flat_sample
        #
        first_tau_checked = True
        # Get integrated autocorrelation time for each parameter
        taus = sampler.get_autocorr_time(tol=0)
        # Get stats of autocorrelation time
        tau_min, tau_max, tau_mean, tau_frac_max, converged, old_taus = \
            check_sampler_convergence(Nsteps, taus, old_taus)
        # Save autocorrelation info for plotting
        autocorr_min[index] = tau_min
        autocorr_max[index] = tau_max
        autocorr_mean[index] = tau_mean
        if mcmc_run_to_convergence_verbose:
            # print info to user about convergence
            print_mcmc_convergence_step(Nsteps, tau_max, tau_frac_max)
            # and print out the acceptance fraction
            af = sampler.acceptance_fraction
            print(f"  Current average acceptance fraction = {np.mean(af):.3f}")
        if mcmc_run_to_convergence_plot_autocorr:
            # make a plot of the autocorrelation values vs steps
            plot_and_save_autocorrelation(index, autocorr_min, autocorr_max,
                                          autocorr_mean)
        if mcmc_run_to_convergence_plot_chains:
            # plot and save the recent time-series of "important" parameters
            plot_and_save_chains(sampler, tau_max)
        if mcmc_run_to_convergence_save_samples:
            # plot and save a set of the last few (~1000) independent samples 
            plot_and_save_samples(sampler, tau_max)
        if mcmc_run_to_convergence_get_marginal_likelihood:
            # print info about the marginal likelihood, 
            # make a plot of ML vs steps, and save data to a file
            burnin, thin = get_burn_thin(tau_max)
            if (Nsteps > 2*burnin):
                flat_samples = sampler.get_chain(discard=burnin,
                                                 thin=thin, flat=True)
                flat_blobs = sampler.get_blobs(discard=burnin,
                                               thin=thin, flat=True)
                flat_loglike = flat_blobs["log_like"]
                flat_logprior = flat_blobs["log_prior"]
                # calculate and print ML to user
                hm, gd = conv_check_get_marginal_likelihood(flat_samples,
                                                            flat_loglike,
                                                            flat_logprior)
                log_marg_like_hm.append(hm)
                log_marg_like_gd.append(gd)
                n_steps_ml.append(sampler.iteration)
                # plot marginal likelihood vs steps; save plot and data
                plot_and_save_marginal_likelihood(n_steps_ml, log_marg_like_hm, log_marg_like_gd)
        index += 1
        # if reached convergence then done
        if converged:
            break

def do_sampling(init_pos, n_dim, moves=None, backend = None, pool = None, restart=False):
    print("Creating EnsembleSampler...")
    # blobs for log_likelihood and log_prior in the log_posterior_prob return
    dtype = [("log_like", float), ("log_prior", float)]
    # define the sampler
    sampler = emcee.EnsembleSampler(
        mcmc_N_walkers, n_dim, log_posterior_probability,
        moves=moves,
        pool=pool,
        backend=backend,
        blobs_dtype=dtype
    )
    # checking convergence on the fly or just running for max steps
    if mcmc_run_to_convergence:
        print("Running sampler to convergence ", end='')
        print(f"(N_autocorr = {mcmc_run_to_convergence_N_autocorrs:d}; ", end='')
        print(f"Delta_autocorr_frac = {mcmc_run_to_convergence_delta_tau_frac:.3f})")
        print(f"with max steps {mcmc_N_steps_max:d}, ", end='')
        print(f"checking convergence every {mcmc_run_to_convergence_check_steps:d} ...\n")
        sample_to_convergence(sampler, init_pos, restart=restart)
    else:
        mcmc_N_steps = get_mcmc_N_steps(restart)
        print(f"Running sampler for {mcmc_N_steps:d} steps ... \n")
        sampler.run_mcmc(init_pos, mcmc_N_steps, progress=mcmc_show_progress,
                         skip_initial_state_check = mcmc_skip_initial_state_check)
    return sampler

def run_mcmc_sampler(pars_map, moves, backend, restart=False):
    # Initialize walkers in small bundle about MAP position
    #   (for discrete-valued pars, will give uniform distribution)
    initial_pos, n_dim = get_init_pos(pars_map)
    # but no need for initial position if a restarted chain
    if restart:
        initial_pos = None
    # do sampling with multiprocessing or not
    #  (and, within do_sampling, evolve to convergence or just do max steps)
    print(f"Running the MCMC with {mcmc_N_walkers:d} walkers and {mcmc_N_steps_max:d} steps each.")
    if mcmc_do_multiprocessing:
        with mp.Pool(mcmc_ncpus_to_use) as pool:
            if mcmc_ncpus_to_use is None:
                print(f"--> With multiprocessing: Using all of the {ncpu:d} CPUs")
            else:
                print(f"--> With multiprocessing: Using {mcmc_ncpus_to_use:d} of {ncpu:d} CPUs")
            sampler = do_sampling(initial_pos, n_dim, moves=moves,
                                  backend = backend, pool = pool, restart=restart)
    else:
        # no pool
        print(f"--> Single-thread processing")
        sampler = do_sampling(initial_pos, n_dim, moves = moves,
                              backend = backend, restart=restart)
    return sampler, n_dim

def calc_marg_like(flat_samples, flat_loglike, flat_logprior,
                   mvn, convergence_check=False):
    """
    Calculate the marginal likelihood (ML) using the harmonic mean and then
    Gelfand and Dey's corrected harmonic mean with multiplier, h(theta).
    For the h(theta) function use the bestfit multivariate normal, but w/ 
    smaller width (ml_fac_to_reduce_sigma).

    Calculate directly, but also use "logsumexp" in case the probabilities
    are too small to calculate directly (giving a log(ML) instead).
    """
    hdist = mvn.pdf
    Nsamp = len(flat_samples)
    hs = []
    for i in range(Nsamp):
        lh = np.log(hdist(flat_samples[i]))
        hs.append(lh)
    ls = flat_loglike
    ps = flat_logprior
    hs = np.array(hs)
    # Harmonic mean of likelihood estimator:
    #
    #   ML = [ 1/Nsamp sum(  1/ls[i]  ) ]^(-1)
    #
    # log(ML) = + log(Nsamp) - log( sum( 1/exp( loglike[i] ) ) )
    #         = + log(Nsamp) - log( sum( exp( - loglike[i] ) ) )
    #         = + log(Nsamp) - logsumexp( -loglike[i] )
    #
    logML_HM = np.log(Nsamp) - spsp.logsumexp( -1.0*ls )
    ML = (np.sum(1/np.exp(ls)) / Nsamp)**(-1)
    if convergence_check:
        # for printing marg likelihood while running the mcmc sampler
        print(f"  Marginal likelihood using {Nsamp:d} samples:\n"
              + f"\tlogML_HM = {logML_HM:.4e}", end='')
    else:
        print("Marginal likelihood using Harmonic Mean:\n"
          + f"\tusing logsumexp:\tlogML = {logML_HM:.4e} -->\tML = {np.exp(logML_HM):.4e}\n"
          + f"\tdirect calculation:\t\t\t\tML = {ML:.4e}")
    # Gelfand and Dey's corrected harmonic mean is the harmonic mean of:
    #
    #         likelihood * prior / h
    #
    # so:
    #
    # log(ML) = + log(Nsamp) - log( sum( 1/exp( loglike[i] ) ) )
    #         = + log(Nsamp) - log( sum( exp( - loglike[i] ) ) )
    #         = + log(Nsamp) - logsumexp( -loglike[i] )
    #
    # logML = + log(Nsamp) - log( sum( h[i] / (like[i] * prior[i]) ) )
    #       = + log(Nsamp) - log( sum( exp(log(h[i]))
    #                                   / ( exp(loglike[i]) * exp(logprior[i]) ) ) )
    #       = + log(Nsamp) - log( sum( exp[ log(h[i]) - loglike[i] - logprior[i] ] ) )
    #       = + log(Nsamp) - logsumexp( log(h[i]) - loglike[i] - logprior[i] )
    logML_GD = np.log(Nsamp) - spsp.logsumexp( hs - ls - ps )
    ML = (np.sum( np.exp(hs) / np.exp(ls) / np.exp(ps) ) / Nsamp)**(-1)
    if convergence_check:
        print(f"\tlogML_GD = {logML_GD:.4e}")
    else:
        print("Marginal likelihood using G&D's Corrected Harmonic Mean (h(theta) is MVN fit):\n"
          + f"\tusing logsumexp:\tlogML = {logML_GD:.4e} -->\tML = {np.exp(logML_GD):.4e}\n"
          + f"\tdirect calculation:\t\t\t\tML = {ML:.4e}")
    return logML_HM, logML_GD

def get_index_of_top_N_vals(arr, N):
    x = arr.copy()
    n = np.arange(len(x))
    inds = []
    for i in range(N):
       maxind = np.argmax(x)
       inds.append(n[maxind])
       x = np.delete(x, maxind)
       n = np.delete(n, maxind)
    return np.array(inds)

def get_autocorrelation_times(sampler, ignorewarning=False):
    if ignorewarning:
        taus = sampler.get_autocorr_time(tol=0)
    else:
        taus = sampler.get_autocorr_time()
    return taus

def print_autocorrelation_times(taus):
    for i in range(len(taus)):
        outstr = "\ttau_" + par_names_print[i]
        outstr += f" = {taus[i]:0.2f}"
        print(outstr)

def get_flattened_samples(taus, sampler):
    tau_max = np.max(taus)
    tau_min = np.min(taus)
    burnin, thin = get_burn_thin(tau_max)
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    flat_logprob = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
    flat_blobs = sampler.get_blobs(discard=burnin, thin=thin, flat=True)
    flat_loglike = flat_blobs["log_like"]
    flat_logprior = flat_blobs["log_prior"]
    print("Shape of flattened samples is", flat_samples.shape)
    return flat_samples, flat_logprob, flat_loglike, flat_logprior, burnin, thin

def get_top_posterior_values_and_compare(flat_logprob, flat_samples, pars_MAP):
    maxinds = get_index_of_top_N_vals(flat_logprob, 10)
    MAPs = flat_logprob[maxinds]
    print("The top 10 largest posterior values among the samples are:")
    for i in range(10):
        print(f"{MAPs[i]:.5e} with random-effect parameters:\n\t", end='')
        print(flat_samples[maxinds[i],ind_important_pars])
    tree = spspat.KDTree(flat_samples)
    dist, parind = tree.query(pars_MAP)
    print(f"The closest sample to the maximum likelihood theta has:")
    print(flat_samples[parind,ind_important_pars])
    print(f"With log(posterior) = {flat_logprob[parind]:.5e}")

def get_histograms(flat_samples_transformed_plot, pars_MAP_transformed):
    pass
    
def get_corner_plots(flat_samples_transformed_plot, pars_MAP_transformed):
    flat_samples_re = flat_samples_transformed_plot[:,ind_important_pars]
    if cornerplot_plot_maxvals:
        figCI = corner.corner(flat_samples_re, labels=imp_par_names_plot,
                              truths=pars_MAP_transformed[ind_important_pars])
        if cornerplot_make_allplot:
            figCA = corner.corner(flat_samples_transformed_plot,
                                  labels=par_names_plot,
                                  truths=pars_MAP_transformed)
            return figCI, figCA
        else:
            return figCI, None
    else:
        figCI = corner.corner(flat_samples_re, labels=imp_par_names_plot)
        if cornerplot_make_allplot:
            figCA = corner.corner(flat_samples_transformed_plot,
                                  labels=par_names_plot)
            return figCI, figCA
        else:
            return figCI, None

def get_bestfit_mvn(flat_samples):
    # find bestfit multivariate normal to full distribution (untransformed)
    mvn = sps.multivariate_normal(np.mean(flat_samples, axis=0),
                                  np.cov(flat_samples, rowvar=0)/
                                  ml_fac_to_reduce_sigma**2)
    # generate samples from mvn for plotting
    mvn_samples = mvn.rvs(size=ml_fac_to_make_smooth*flat_samples.shape[0])
    return mvn, mvn_samples

def overplot_mvn_on_cornerplots(figCI, figCA, mvn_samples_transformed, ndim):
    if (len(ind_important_pars) == 1):
        iip = np.concatenate( (ind_important_pars, [ind_important_pars[0]-1]) )
        corner.corner(mvn_samples_transformed[:,iip],
                      fig=figCI, color='red',
                      plot_datapoints=False, plot_density=False,
                      hist_kwargs = {'alpha' : 0.0})
        # reset the yrange on the 1-D histograms
        axes = np.array(figCI.axes).reshape((2,2))
        for i in range(2):
            ax = axes[i, i]
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax/ml_fac_to_make_smooth)
    else:
        corner.corner(mvn_samples_transformed[:,ind_important_pars],
                      fig=figCI, color='red',
                      plot_datapoints=False, plot_density=False,
                      hist_kwargs = {'alpha' : 0.0})
        # reset the yrange on the 1-D histograms
        axes = np.array(figCI.axes).reshape((N_imp_pars, N_imp_pars))
        for i in range(N_imp_pars):
            ax = axes[i, i]
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax/ml_fac_to_make_smooth)
        if cornerplot_make_allplot:
            corner.corner(mvn_samples_transformed,
                          fig=figCA, color='red',
                          plot_datapoints=False, plot_density=False,
                          hist_kwargs = {'alpha' : 0.0})

def get_blob_filenames(file_sample):
    fileparts = file_sample.split('_')
    # sample file is of form:
    #
    #     dirname/2022-06-18_2131_samples_this_that_the-other.dat
    #
    # and blobs are
    #
    #     dirname/2022-06-18_2131_samples_<blobname>_this_that_the-other.dat
    #
    file_prefix = fileparts[0] + '_' + fileparts[1] + '_' + fileparts[2] + '_'
    file_suffix = "_"
    for i in np.arange(3,len(fileparts)-1):
        file_suffix += fileparts[i] + '_'
    file_suffix += fileparts[-1]
    return [file_prefix + 'loglike' + file_suffix,
            file_prefix + 'logprior' + file_suffix]
