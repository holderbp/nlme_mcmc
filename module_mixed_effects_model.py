import numpy as np
import pandas as pd

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
            print("\n" + 2*indentstr + index + ": ", end='')
            for c in indiv_nopop:
                print(c + " = " + str(indiv_nopop.at[index, c]) + ", ", end='')
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
                    print(4*indentstr + p + s + f"(prior_mean, prior_stdev) = ", end='')
                    print(f"({prior_mean[p+s]:.3e}, {prior_stdev[p+s]:.3e})")
                elif (prior_dist[p+s] == 'uniform'):
                    print(4*indentstr + p + s + f"(prior_min, prior_max) = ", end='')
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
    num_mcmc_pars = Nindiv*(len(list_indiv_pop_pars) + len(list_indiv_nopop_pars)) + len(pop) + len(other)
    print(indentstr + f"In summary, there are {num_mcmc_pars:d} parameters for MCMC evolution:")
    print(2*indentstr + f"{Nindiv*len(list_indiv_pop_pars):d} indiv_pop parameters: (", end='')
    print_list_w_commas(list_indiv_pop_pars)
    print(") * " + f"{Nindiv:d} individuals")
    print(2*indentstr + f"{Nindiv*len(list_indiv_nopop_pars):d} indiv_nopop parameters: (", end='')
    print_list_w_commas(list_indiv_nopop_pars)
    print(") * " + f"{Nindiv:d} individuals")
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
        else:
            print("***Error: transform " + transf_func[parname] + " not found")                
            exit(0)
    else:
        if (transf_func[parname] == 'none'):
            return val
        elif (transf_func[parname] == 'log'):
            return np.log(val)
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
    else:
        print("***Error: transform " + transf_func[parname] + " not found")
    exit(0)

def give_vec_for(transf_func, justnames=False):
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
            for i in indiv_pop.index:
                arr = np.concatenate(
                    (arr, [ get_name_w(transf_func, p + "_mean", i=i) ])
                )
        for p in list_indiv_nopop_pars:
            for i in indiv_nopop.index:
                arr = np.concatenate(
                    (arr, [ get_name_w(transf_func, p, i=i) ])
                )
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
        arr = np.concatenate(
            (arr, get_val_w(transf_func, p + '_mean', indiv_pop[p].to_numpy() ) )
        )
    # INDIV_NOPOP:  for each par, append *vectors* over individuals            
    for p in list_indiv_nopop_pars:
        arr = np.concatenate(
            (arr, get_val_w(transf_func, p, indiv_nopop[p].to_numpy() ) )
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
    # For INDIV_* transfer full columns from the vector to df
    N_ipp = len(list_indiv_pop_pars)
    for (p, n) in zip(list_indiv_pop_pars, range(N_ipp)):
        indiv_pop[p] = get_val_w(transf_mcmc, p + '_mean',
                                 arr[n*Nindiv:(n+1)*Nindiv], reverse=True)
    N_inpp = len(list_indiv_nopop_pars)
    for (p, n) in zip(list_indiv_nopop_pars, range(N_inpp)):
        indiv_nopop[p] = get_val_w(transf_mcmc, p,
                                   arr[(N_ipp+n)*Nindiv:(N_ipp+n+1)*Nindiv],
                                   reverse=True)
    # For POP and OTHER, transfer element by element from vec to dictionary
    count = Nindiv * (N_ipp + N_inpp)
    for p in pop:
        pop[p] = get_val_w(transf_mcmc, p, arr[count], reverse=True)
        count += 1
    for p in other:
        other[p] = get_val_w(transf_mcmc, p, arr[count], reverse=True)
        count += 1

def log_like_and_prior(theta):
    """
    Public method called by mcmc_mixed-effects to return both 
    the likelihood and the prior for a given parameter vector
    """
    # put the parameters into their dictionaries
    take_vec_from_mcmc(theta)
    # set the parameters for the data module
    dat.set_pars(indiv_pop, indiv_nopop, other)
    # get the log-likelihood from the data module
    ll = dat.get_log_likelihood()
    # get the log-prior
    lp = get_log_prior()
    # return to mcmc_mixed-effects
    return ll, lp

def get_normal_logprior(pr_mean, pr_var, val):
    return -0.5 * np.sum( (val - pr_mean)**2 / pr_var + np.log(2*np.pi*pr_var) )

def get_uniform_logprior(pr_min, pr_max, val):
    if ( np.all(val > pr_min) & np.all(val < pr_max) ):
        return 0.0
    else:
        return -np.inf

def get_log_prior():
    log_prior = 0.0
    # get conditional prior values of INDIV_POP pars wrt population dist
    for p in list_indiv_pop_pars:
        if (pop_dist[p] == 'normal'):
            log_prior += get_normal_logprior(pop[p+'_mean'], pop[p+'_stdev']**2,
                                             indiv_pop[p].to_numpy())
        elif (pop_dist[p] == 'lognormal'):
            log_prior += get_normal_logprior(np.log(pop[p+'_mean']),
                                             np.log(pop[p+'_stdev'])**2,
                                             np.log(indiv_pop[p].to_numpy()) )
    # get prior values of INDIV_NOPOP pars
    for p in list_indiv_nopop_pars:
        if (prior_dist[p] == 'normal'):
            log_prior += get_normal_logprior(prior_mean[p], prior_stdev[p]**2,
                                indiv_nopop[p].to_numpy())
        elif (prior_dist[p] == 'lognormal'):
            log_prior += get_normal_logprior(np.log(prior_mean[p]),
                                             np.log(prior_stdev[p])**2,
                                             np.log(indiv_nopop[p].to_numpy()))
        elif ( prior_dist[p] == 'uniform'):
            log_prior += get_uniform_logprior(prior_min[p], prior_max[p],
                                              indiv_nopop[p].to_numpy())
    # get prior values of POP parameters
    for p in list_indiv_pop_pars:
        for s in ['_mean', '_stdev']:
            if (prior_dist[p+s] == 'normal'):
                log_prior += get_normal_logprior(prior_mean[p+s],
                                                 prior_stdev[p+s]**2, pop[p+s])
            elif (prior_dist[p+s] == 'lognormal'):
                log_prior += get_normal_logprior(np.log(prior_mean[p+s]),
                                                 np.log(prior_stdev[p+s])**2,
                                                 np.log(pop[p+s]))
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
                                             np.log(prior_stdev[p])**2,
                                             np.log(other[p]))
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
