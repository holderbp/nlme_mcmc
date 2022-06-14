import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
# make default style for seaborn
sns.set()

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
import module_sleepstudy_data as dat
project_name = dat.project_name # should be 'sleepstudy'
#
#=== Import the evolvemodel module
#
import module_sleepstudy_evolvemodel as evo
#
#=== Define the DATA ANALYSIS SUB-PROJECT to be used
#
#       The only sleepstudy data set is:
#          response time (ms) vs days of sleep deprivation
#
data_analysis_subproject = 'response-time-vs-days-deprivation'
#
#=== Name this module's MODEL HYPOTHESIS
#
#           Random effects for b (single distribution),
#           common slope
#           common normal error (stdev) in data about model, sigma
#
model_hyp = 'RE-single-group-b-common-m-sigma'
#
#=== Load data into the data module and get:
#
#      * information on the individuals (Nindiv, indiv_names)
#
#      * data sets for every individual (yvals_for_indiv) for
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
  mm.evolve_models_for_indiv,  mm.evolve_models_xvals_for_indiv,
  mm.datasets_for_indiv, mm.xindices_for_indiv ] = \
      dat.load_data(model_hyp, data_analysis_subproject)
#
#=== Initialize the evolve-model module
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
#    For this model hypothesis, the indiv_pop pars for each
#    individual are:
#
#                  ['b']
#
widthfac = 5.0
mm.list_indiv_pop_pars = ['b']
mm.descrip_name['b'] = 'intercept'
init_vecs = [
    [290.0, 200.0, 226.0, 273.0, 298.0, 316.0,
     286.0, 238.0, 263.9, 312.9, 229.0, 248.0,
     264.0, 330.0, 267.0, 235.0, 257.0, 290.7]
]
d = dict(zip(mm.list_indiv_pop_pars, init_vecs))
mm.indiv_pop = pd.DataFrame(d, index=mm.indiv_names)
#
#--- INDIV_NOPOP (Individual parameters not distributed by population)
#
#    For this model hypothesis, the indiv_nopop pars for each
#    individual are:
#
#                        <none>
#
mm.list_indiv_nopop_pars = []
d = {}
mm.indiv_nopop = pd.DataFrame(d)
#
#    If there were parameters in indiv_nopop, we would use the
#    individual names as an index (as with indiv_pop):
#
#          mm.indiv_nopop = pd.DataFrame(d, index=mm.indiv_names)
#
#    and also define elements of these dictionaries for each
#    parameter
#
#        [descrip_name,
#         prior_dist, prior_mean, prior_min, prior_max,
#         tranf_mcmc, tranf_plot, transf_print]
#
#--- POP (Random effects population distribution parameters)
#
#    For each parameter in "list_indiv_pop_pars", specify a
#    distribution, with parameters, priors, and transformations
#
#--- intercept "b"
#
# population distribution type
mm.pop_dist['b'] = 'normal'
# two population parameters to be followed by MCMC
mm.pop['b_mean'], mm.pop['b_stdev'] = 300.0, 100.0
# prior and transformation for the mean    
mm.prior_dist['b_mean'] = 'normal'
mm.prior_mean['b_mean'], mm.prior_stdev['b_mean'] = 300.0, 100.0
mm.prior_min['b_mean'], mm.prior_max['b_mean'] = 200.0, 400.0
mm.transf_mcmc['b_mean'] = 'none'
mm.transf_plot['b_mean'] = 'none'
mm.transf_print['b_mean'] = 'none'
# prior and transformation for the stdev    
mm.prior_dist['b_stdev'] = 'lognormal'
mm.prior_mean['b_stdev'], mm.prior_stdev['b_stdev'] = mm.pop['b_stdev'], widthfac    
mm.prior_min['b_stdev'], mm.prior_max['b_stdev'] = 1.0, 400.0
mm.transf_mcmc['b_stdev'] = 'log'
mm.transf_plot['b_stdev'] = 'none'
mm.transf_print['b_stdev'] = 'none'
#
#--- OTHER (other pars, e.g., those associated to error-model)
#
#    In this model hypothesis we have:
#
#        'm': common slope for all individuals
#        'sigma': common normal error stdev for all individuals
#
mm.list_other_pars = ['m', 'sigma']
# prior and transformation for slope
mm.descrip_name['m'] = 'common slope'
mm.other['m'] = 8.0
mm.prior_dist['m'] = 'normal'
mm.prior_mean['m'], mm.prior_stdev['m'] = 8.0, 30.0
mm.prior_min['m'], mm.prior_max['m'] = -20.0, 40.0
mm.transf_mcmc['m'] = 'none'
mm.transf_plot['m'] = 'none'
mm.transf_print['m'] = 'none'
# prior and transformation for sigma    
mm.descrip_name['sigma'] = 'common data error variance'
mm.other['sigma'] = 20.0
mm.prior_dist['sigma'] = 'lognormal'
mm.prior_mean['sigma'], mm.prior_stdev['sigma'] = mm.other['sigma'], widthfac
mm.prior_min['sigma'], mm.prior_max['sigma'] = 1.0, 100.0
mm.transf_mcmc['sigma'] = 'log'
mm.transf_plot['sigma'] = 'none'
mm.transf_print['sigma'] = 'none'
#
#=== IMPORTANT PARAMETERS (non-nuisance) for plotting
#
#    For this model hypothesis, we consider the following
#    parameters to be "important"/non-nuisance:
#
#             [<pop pars>, sigma]
#
important_pars = ['b_mean', 'b_stdev', 'm', 'sigma']
ind_important_pars = None  # to be set below

def get_log_likelihood():
    #
    #--- Give the data module the current parameters
    #
    evo.set_pars(mm.indiv_pop, mm.indiv_nopop, mm.other)
    #
    #--- Error model
    #
    # for this model-hypothesis, we assume:
    #
    #   normally-distributed error with
    #   common variance for all data sets
    #
    sigma_sqrd = mm.other['sigma']**2
    #
    # but you could have individualized values (set within
    # the indiv_names loop below):
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
            # Run the evolve-model from the data module and
            # receive a dictionary of np arrays with Ymodel values
            #
            xvals = mm.evolve_models_xvals_for_indiv[i][em]
            y_model = evo.get_evolve_model_solutions(i, em, xvals)
            #
            # The data module ensures that the two arrays:
            #
            #     yvals_for_indiv[i][em][d]   and    y_model[d]
            #
            # are the same length
            #
            for d in mm.datasets_for_indiv[i][em]:
                # select the y-values for that data set, and slice
                # out the y-values for the x-values of that dataset
                ymod = y_model[d][mm.xindices_for_indiv[i][em][d]]
                log_like += \
                    -0.5*np.sum( (mm.yvals_for_indiv[i][em][d] - ymod)**2 / sigma_sqrd
                                 + np.log(2.0 * np.pi * sigma_sqrd) )
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

def get_pars(transf_type, getnames=False):
    """
    Public method for getting the parameters and the parameter names
    from the mod-hyp module. (Called by mcmc_mixed-effects.py)
    """
    if ( (transf_type == 'print')
         | ( (plot_with_print_transformation) & (transf_type == 'plot') ) ):
        thetavec = mm.give_vec_for(mm.transf_print, justnames=False)
        if getnames:
            return thetavec, mm.give_vec_for(mm.transf_print, justnames=True)
        else:
            return thetavec
    elif (transf_type == 'plot'):
        thetavec = mm.give_vec_for(mm.transf_plot, justnames=False)        
        if getnames:
            return thetavec, mm.give_vec_for(mm.transf_plot, justnames=True)
        else:
            return thetavec

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

