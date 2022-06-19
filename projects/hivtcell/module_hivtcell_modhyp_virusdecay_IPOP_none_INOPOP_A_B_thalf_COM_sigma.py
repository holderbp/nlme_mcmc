import numpy as np
import pandas as pd

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
#       The only sleepstudy data set is:
#          response time (ms) vs days of sleep deprivation
#
data_analysis_subproject = 'virus-decay'
#
#=== Name this module's MODEL HYPOTHESIS
#
#           Random effects for m and b (single distribution, no cov)
#           common normal error (stdev) in data about model, sigma
#
model_hyp = 'no-RE-all-cells-common-lognormal-errors'
#
#=== Load data into the data module and get the following
#    parameters and parameter objects (to be stored in the
#    mixed-effects module)
#
#      * information on the individuals (Nindiv, indiv_names)
#
#      * data sets for every individual (mm.yvals_for_indiv) for
#        this data_analysis_subproject
#
#      * lists of the evolve-models to be run for each individual
#        (evolve_models_for_indiv), and the data sets to be
#        generated by each evolve-model (datasets_for_indiv), which
#        are then compared against the yvals within the log-likelihood
#        calculation.
#
[ mm.Nindiv, mm.indiv_names,
  mm.yvals_for_indiv, mm.yvals_type_for_indiv,
  mm.evolve_models_for_indiv, mm.evolve_models_xvals_for_indiv,
  mm.datasets_for_indiv, mm.xindices_for_indiv ] = \
      dat.load_data(model_hyp, data_analysis_subproject)
#
#=== Import the evolve-model module and initialize it
#
import module_hivtcell_evolvemodel as evo
evo.initialize(mm.datasets_for_indiv, mm.yvals_type_for_indiv)
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
#    For this data analysis subproject, the (ordered) individual
#    names are:
#
#      [act_virus_decay_1, act_virus_decay_2,
#       ecm_virus_decay_1, ecm_virus_decay_2,
#       ecp_virus_decay_1, ecp_virus_decay_2,
#       rest_virus_decay_1, rest_virus_decay_2]
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
mm.list_indiv_nopop_pars = ['A', 'B', 'thalf']
#
# order is [act, ecm, ecp, rest] within [A, B, thalf]
#
# guess values taken from xmgrace fits (2017-05-08 analysis)
#    (thalf = log(2) / c)
#
init_vecs = [
    [10736, 24900,
     1890, 3670,
     6817, 8508,
     204, 204],
    [9.46, 5.52,
     7.5, 8.0,
     165, 52,
     7.2, 7.2],
    [0.36, 0.42,
     0.34, 0.28,
     0.23, 0.25,
     0.23, 0.23]
]
d = dict(zip(mm.list_indiv_nopop_pars, init_vecs))
mm.indiv_nopop = pd.DataFrame(d, index=mm.indiv_names)
mm.descrip_name['A'] = "outer coeff, moi model"
mm.prior_dist['A'] = 'lognormal'
mm.prior_mean['A'] = 0.01
mm.prior_stdev['A'] = 10.0
mm.transf_mcmc['A'] = 'log'
mm.transf_plot['A'] = 'log'
mm.transf_print['A'] = 'none'
mm.descrip_name['B'] = "inner coeff, moi model"
mm.prior_dist['B'] = 'lognormal'
mm.prior_mean['B'] = 3.0
mm.prior_stdev['B'] = 5.0
mm.transf_mcmc['B'] = 'log'
mm.transf_plot['B'] = 'log'
mm.transf_print['B'] = 'none'
mm.descrip_name['thalf'] = "virion infectious halflife"
mm.prior_dist['thalf'] = 'lognormal'
mm.prior_mean['thalf'] = 0.1
mm.prior_stdev['thalf'] = 5.0
mm.transf_mcmc['thalf'] = 'log'
mm.transf_plot['thalf'] = 'none'
mm.transf_print['thalf'] = 'none'
#
#--- POP (Random effects population distribution parameters)
#
#    For each parameter in "list_indiv_pop_pars", specify a
#    distribution, with parameters, priors, and transformations
#
#          <none>
#
#--- OTHER (other pars, e.g., those associated to error-model)
#
#    In this model hypothesis we have:
#
#        'sigma': common log-normal error stdev for all individuals
#
mm.list_other_pars = ['sigma']
mm.descrip_name['sigma'] = 'common data lognormal-error'
mm.other['sigma'] = 2.0
# prior and transformation for sigma    
mm.prior_dist['sigma'] = 'lognormal'
mm.prior_mean['sigma'], mm.prior_stdev['sigma'] = 2.0, 5.0
mm.prior_min['sigma'], mm.prior_max['sigma'] = 0.1, 50.0
mm.transf_mcmc['sigma'] = 'log'
mm.transf_plot['sigma'] = 'log'
mm.transf_print['sigma'] = 'none'
#
#=== IMPORTANT PARAMETERS (non-nuisance) for plotting
#
#    For this model hypothesis, we consider the following
#    parameters to be "important"/non-nuisance:
#
#             [<pop pars>, sigma]
#
important_pars = ['sigma']
ind_important_pars = None  # to be set below

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
    #   log-normally-distributed error with
    #   common variance for all data sets
    #
    logsigmasqrd = (np.log(mm.other['sigma']) )**2
    #
    # but you could have individualized values (set within
    # the mm.indiv_names loop below):
    #
    #      sigma_sqrd = indiv_nopop.at[i, 'sigma']
    #
    # or values that depend on the particular data set
    # (set within the datasets_for_indiv[i][em] loop below):
    #
    #      if (d == 'virus-decay'):
    #          sigma_sqrd = other['sigma_virus-decay']
    #
    # or you could design a non-normal error model, that
    # is potentially dataset-dependent.
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
                log_like += \
                    -0.5*np.sum( (np.log(mm.yvals_for_indiv[i][em][d])
                                  - np.log(ymod))**2 \
                                 / logsigmasqrd
                                 + np.log(2.0 * np.pi * logsigmasqrd) )
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

def get_ind_imp_pars():
    """
    Public method for getting the numpy vector indices of those
    parameters that are considered "important" (non-nuisance).
    """
    global ind_important_pars
    if ind_important_pars is None:
        # Find the indices of the important parameters in the numpy vector 
        parnames = list(mm.give_vec_for(mm.transf_print, justnames=True))
        ind_important_pars = []
        for p in important_pars:
            ind_important_pars.append(parnames.index(p))
        ind_important_pars = np.array(ind_important_pars)
    return ind_important_pars
        
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
    # get the log-likelihood
    ll = get_log_likelihood()
    # get the log-prior
    lp = mm.get_log_prior()
    # return to mcmc_mixed-effects
    return ll, lp

