import numpy as np
import pandas as pd
import random
import functools
import scipy.special as spsp

print = functools.partial(print, flush=True)

#======================================================================#
#                                                                      #
#       When creating a new model hypothesis, edit this section        #
#                                                                      #
#======================================================================#

#
#=== Import the mixed model module
#
import module_mixed_effects_model as mm
#
#=== Import the data PROJECT
#
import module_hivtcell_data as dat
project_name = dat.project_name 
#
#=== Define the DATA ANALYSIS SUB-PROJECT to be used
#
data_analysis_subproject = 'ecp-uninf-timeseries'
#
#=== Name this module's MODEL HYPOTHESIS
#
model_hyp = 'IPOP-none_INOPOP-N-tauT-nT-tD_COM-sigma_CONST-s'
#
#=== Load data into the data module and get the following
#    parameters and parameter objects (to be stored in the
#    mixed-effects module)
#
[ mm.Nindiv, mm.indiv_names,
  mm.yvals_for_indiv, mm.yvals_type_for_indiv,
  mm.evolve_models_for_indiv, mm.evolve_models_xvals_for_indiv,
  mm.datasets_for_indiv, mm.xindices_for_indiv ] = \
      dat.load_data(model_hyp, data_analysis_subproject)
#
#=== Import the evolve-model module
#
import module_hivtcell_evolvemodel as evo
#
#=== Optionally override the below specified transformations for
#    plotting of all variables, and make plots with the "print"
#    transformation instead. (e.g., some positive quanties might
#    usually be plotted as log("par"), but you can plot them as just
#    "par" instead)
#
plot_with_print_transformation=False
#
#=== Specify the mixed-model parameter objects
#
#--- INDIV_POP (Individual parameters distributed by population)
#
#    Individuals for uninfected-timeseries
#
#      [ act_timeseries_uninf_1, act_timeseries_uninf_2,
#        act_timeseries_uninf_3,
#        ecm_timeseries_uninf_1, ecm_timeseries_uninf_2,
#        ecm_timeseries_uninf_3,
#        ecp_timeseries_uninf_1, ecp_timeseries_uninf_2,
#        ecp_timeseries_uninf_3,
#        rest_timeseries_uninf_1, rest_timeseries_uninf_2,
#        rest_timeseries_uninf_3 ]
#
#    For this model hypothesis, the indiv_pop pars for each
#    individual are:
#
#                  []
#
widthfac = 5.0
mm.list_indiv_pop_pars = []
#mm.descrip_name['m'] = 'slope'
#init_vecs = [
#    [25, 0.0, 10, 15, 2, 0,
#     15, 25, -10, 15, 5, 10,
#     0, 5, 10, 10, 5, 5],
#    [290.0, 200.0, 226.0, 273.0, 298.0, 316.0,
#     286.0, 238.0, 263.9, 312.9, 229.0, 248.0,
#     264.0, 330.0, 267.0, 235.0, 257.0, 290.7]
#]
#d = dict(zip(mm.list_indiv_pop_pars, init_vecs))
#mm.indiv_pop = pd.DataFrame(d, index=mm.indiv_names)
d = {}
mm.indiv_pop = pd.DataFrame(d)
#
#--- INDIV_NOPOP (Individual parameters not distributed by population)
#
#    For this dasp, there are three parameters
#
mm.list_indiv_nopop_pars = ['N', 'tauT', 'nT', 'dD']
#
#   s = log(2) / t_double (not applicable for any but ACT)
#
#   dD = log(2) / t_disinteg_halflife
#
init_vecs = [
    3e5 * np.ones(3),
    [
        9.5, 11.0, 11.0,
    ],
    [
        4, 4, 3,
    ],
    [
        np.log(2)/ 4, np.log(2) / 20, np.log(2) / 10,
    ],
]
d = dict(zip(mm.list_indiv_nopop_pars, init_vecs))
mm.indiv_nopop = pd.DataFrame(d, index=mm.indiv_names)
# priors for each indiv_nopop parameter
mm.descrip_name['N'] = "initial number of cells in culture"
mm.prior_dist['N'] = 'lognormal'
mm.prior_mean['N'] = 3e5
mm.prior_stdev['N'] = np.log(1.5)
mm.transf_mcmc['N'] = 'log'
mm.transf_plot['N'] = 'none'
mm.transf_print['N'] = 'none'
mm.descrip_name['tauT'] = "average lifespan of uninf cells in culture (d)"
mm.prior_dist['tauT'] = 'lognormal'
mm.prior_mean['tauT'] = 4.0
mm.prior_stdev['tauT'] = np.log(5.0)
mm.transf_mcmc['tauT'] = 'log'
mm.transf_plot['tauT'] = 'none'
mm.transf_print['tauT'] = 'none'
mm.descrip_name['nT'] = "integer number of equations for Erlang dist of uninf"
mm.prior_dist['nT'] = 'uniform'
mm.prior_min['nT'] = 0.5
mm.prior_max['nT'] = 40.5
mm.transf_mcmc['nT'] = 'none'
mm.transf_plot['nT'] = 'round'
mm.transf_print['nT'] = 'round'
mm.descrip_name['dD'] = "log(2)/disintegration-halflife dead cells (1/d)"
mm.prior_dist['dD'] = 'lognormal'
mm.prior_mean['dD'] = 0.15
mm.prior_stdev['dD'] = np.log(5.0)
mm.transf_mcmc['dD'] = 'log'
mm.transf_plot['dD'] = 'none'
mm.transf_print['dD'] = 'none'
#
#--- POP (Random effects population distribution parameters)
#
#    For each parameter in "list_indiv_pop_pars", specify a
#    distribution, with parameters, priors, and transformations
#
#--- OTHER (other pars, e.g., those associated to error-model)
#
mm.list_other_pars = ['sigma']
mm.list_other_pars = ['sigma']
mm.descrip_name['sigma'] = 'common data lognormal-error'
mm.other['sigma'] = np.log(2.0)
# prior and transformation for sigma    
mm.prior_dist['sigma'] = 'lognormal'
mm.prior_mean['sigma'] = np.log(2.0)
mm.prior_stdev['sigma'] = np.log(5.0)
mm.prior_min['sigma'] = 0.1
mm.prior_max['sigma'] = 50.0
mm.transf_mcmc['sigma'] = 'log'
mm.transf_plot['sigma'] = 'log'
mm.transf_print['sigma'] = 'none'
#
#--- CONSTANTS
#
par_constants = {}
par_constants['s'] = 0.0
#
#=== IMPORTANT (non-nuisance) PARAMETERS
#
important_pars = ['sigma']
ind_important_pars = None  # to be set below
#
#=== DISCRETE PARAMETERS 
#
discrete_pars = ['nT']    # will seek out <parname>+'[' and <parname>+'_mean' as well
ind_discrete_pars = None  # to be set below

#
#=== Initialize the evolution model
#
evo.initialize(mm.datasets_for_indiv, mm.yvals_type_for_indiv, par_constants)

def get_log_likelihood():
    #
    #--- Give the evolve-model module the current parameters
    #
    evo.set_pars(mm.indiv_pop, mm.indiv_nopop, mm.other)
    #
    #--- Error model
    #
    # for this model-hypothesis, we assume:
    #
    var = mm.other['sigma']**2
    #
    #--- Calculate the log likelihood
    #
    log_like = 0.0
    for i in mm.indiv_names:
        for em in mm.evolve_models_for_indiv[i]:
            #
            # Run the evolve-model and receive a dictionary
            # of np arrays with Ymodel values
            #
            xvals = mm.evolve_models_xvals_for_indiv[i][em]
            y_model = evo.get_evolve_model_solutions(i, em, xvals)         
            #
            # The data module ensures that the two arrays:
            #
            #     mm.yvals_for_indiv[i][em][d]   and    y_model[d]
            #
            # are the same length
            #
            for d in mm.datasets_for_indiv[i][em]:
                # select the y-values for that data set, and slice
                # out the y-values for the x-values of that dataset
                ymod = y_model[d][mm.xindices_for_indiv[i][em][d]]
                ll = mm.get_lognormal_loglike(mm.yvals_for_indiv[i][em][d],
                                              ymod, var)
                #ll = mm.get_poisson_loglike(mm.yvals_for_indiv[i][em][d],
                #                            ymod)
                #print("[" + i + "] [" + d + "] ll = " + f"{ll:.3e}")
                log_like += ll
    return log_like

#======================================================================#
#                                                                      #
#                No adjustments should be needed below                 #
#                                                                      #
#======================================================================#

def initialize(indentstr="", dividerstr=""):
    """
    Called by mcmc_mixed-effects to print out pars
    """
    #--- Print model-hyp parameters to user/logfile
    print("Mixed model parameters:")
    print(indentstr + "project_name = " +  project_name)
    print(indentstr + "data_analysis_subproject = " + data_analysis_subproject)
    print(indentstr + "model_hyp = " + model_hyp)
    print(indentstr + "plot_with_print_transformation = " + str(plot_with_print_transformation))    
    #--- Print all of the mixed model parameters
    mm.print_mixed_model_pars(indentstr, dividerstr)
    #
    #--- Do some checks
    #
    mm.check_all_names_distinct()
    mm.check_Nindiv_correct()
    #
    #--- Return vector of MCMC parameters to user
    #
    return mm.give_vec_for(mm.transf_mcmc)

def plot_data(ppars):
    """
    Public method called by mcmc_plotdata.py to plot the data sets
    in this data analysis subproject, with a representation of the
    posterior distribution overplotted (spaghetti, lines to indicate
    confidence intervals, or, probably best: shaded regions of 68 and 
    95% confidence.

    Might not differ much from one model hypothesis to the next, but
    need to check the evo.get_evolve_model_solutions(...) part
    """
    if (ppars['plot_type'] == 'only_data'):
        #===================================================        
        #               Just make a data plot
        #===================================================        
        dat.plot_data(ppars)
    else:
        #===================================================
        #          Get model values for each sample
        #===================================================        
        plot_pars = mm.get_evolvemodels_for_plotting(ppars, evo)
        #===================================================
        #  Send to data module for plot creation and output
        #===================================================
        dat.plot_data(plot_pars)

def get_index_important_and_discrete_pars():
    """
    Public method for getting the numpy vector indices of those
    parameters that are considered "important" (non-nuisance).
    """
    global ind_important_pars, important_pars, ind_discrete_pars
    if ind_important_pars is None:
        # Find the indices of the important parameters in the numpy vector 
        parnames = list(mm.give_vec_for(mm.transf_print, justnames=True))
        max_imp = np.min([5, len(parnames)])
        if (len(important_pars) < max_imp):
            print(f"\t(well...there were only {len(important_pars):d} ", end='')
            print("given, so we'll randomly choose a few more)")
            # if less than max_imp were given, add a few randomly to make max_imp
            n = len(important_pars)
            ind_important_pars = []
            otherinds = list(range(len(parnames)))
            for p in important_pars:
                i = parnames.index(p)
                ind_important_pars.append(i)
                otherinds.remove(i)
            inds = random.sample(otherinds, max_imp - n)
            for i in range(max_imp-n):
                ind_important_pars.append(inds[i])
            ind_important_pars = np.array(ind_important_pars)
            important_pars = \
                mm.give_vec_for(mm.transf_print, justnames=True)[ind_important_pars]
        else:
            ind_important_pars = []
            for p in important_pars:
                ind_important_pars.append(parnames.index(p))
            ind_important_pars = np.array(ind_important_pars)
    if ind_discrete_pars is None:
        # Find the indices of the discrete parameters in the numpy vector
        parnames = list(mm.give_vec_for(mm.transf_print, justnames=True))
        ind_discrete_pars = []
        for p in discrete_pars:
            pindiv = p + '['
            pmean = p + '_mean'
            for i in range(len(parnames)):
                if ( (p == parnames[i]) | (pmean == parnames[i])
                     | ( pindiv in parnames[i] ) ):
                    ind_discrete_pars.append(parnames.index(parnames[i]))
        ind_discrete_pars = np.array(ind_discrete_pars)
    return ind_important_pars, ind_discrete_pars
        
def set_pars(theta):
    """ 
    Public method for sending numpy parameter vector to the mod-hyp
    module.
    """
    mm.take_vec_from_mcmc(theta)

def get_pars(transf_type, justnames=False):
    """
    Public method for getting the parameters and the parameter names
    from the mod-hyp module. (Called by mcmc_mixed-effects.py)
    """
    if ( (transf_type == 'print')
         | ( (transf_type == 'plot') & (plot_with_print_transformation) ) ):
        if justnames:
            return mm.give_vec_for(mm.transf_print, justnames=True)
        else:
            return mm.give_vec_for(mm.transf_print, justnames=False)
    elif (transf_type == 'plot'):
        if justnames:
            return mm.give_vec_for(mm.transf_plot, justnames=True)
        else:
            return mm.give_vec_for(mm.transf_plot, justnames=False)        
    elif (transf_type == 'mcmc'):
        if justnames:
            return mm.give_vec_for(mm.transf_mcmc, justnames=True)
        else:
            return mm.give_vec_for(mm.transf_mcmc, justnames=False)        

def log_like_and_prior(theta):
    """
    Public method called by mcmc_mixed-effects to return both 
    the likelihood and the prior for a given parameter vector
    """
    # put the parameters into their dictionaries
    mm.take_vec_from_mcmc(theta)
    # get the log-prior
    lp = mm.get_log_prior()
    # just return if not in prior
    if np.isneginf(lp):
        return lp, lp
    # get the log-likelihood 
    ll = get_log_likelihood()
    # return to mcmc_mixed-effects
    #print("\n", theta, "\n", ll, lp)
    return ll, lp
