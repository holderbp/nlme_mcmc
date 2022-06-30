import sys
import functools
import numpy as np
import pandas as pd

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#            Import the mixed-effects module
#
#   Contains helper-functions to be used in this script
#
#==============================================================
import module_mixed_effects_model as mm
#==============================================================
#//////////////////////////////////////////////////////////////

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#            Import the chosen model hypothesis
#
#   This must match the input "sample" data file below so that
#   the parameters in the sample are correctly identified.
#                                                              
#==============================================================
#
#--- virus decay
#
#import module_hivtcell_modhyp_virusdecay_IPOP_none_INOPOP_A_B_COM_thalf_sigma as mod
#import module_hivtcell_modhyp_virusdecay_IPOP_thalf_INOPOP_A_B_COM_sigma as mod
#
#--- uninfected timeseries
#
#import module_hivtcell_modhyp_ecp_uninf_timeseries_IPOP_none_INOPOP_tauT_nT_tD_t2_COM_sigma as mod
import module_hivtcell_modhyp_ecp_uninf_timeseries_IPOP_none_INOPOP_N_tauT_nT_tD_COM_sigma_CONST_s as mod
#//////////////////////////////////////////////////////////////

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                Figure/Plotting options
#==============================================================
# put the max-likelihood/posterior on cornerplot as "truths"
#     (otherwise no "truths")
mm.cornerplot_plot_maxvals = True
# in addition to an "important parameters" plot, make one for all
mm.cornerplot_make_allplot = True
#==============================================================
#//////////////////////////////////////////////////////////////

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#         Maximum a posteriori/likelihood options
#       (MAP runs before MCMC to provide initial bundle)
#==============================================================
mm.do_maximum_likelihood = False # otherwise, maximum a *posteriori*
mm.minimize_tol = 1e-12
# spo.minimize methods
#
#   default:             'BFGS'
#   same result as def:  'L-BFGS-B', 'Powell', 'CG', 'SLSQP'
#   different result:    'Nelder-Mead', 'TNC', 'COBYLA'
#
# and my own mash of them:
#
#   'avg' = get avg of:  'BFGS', 'Nelder-Mead', 'TNC', 'COBYLA'
#
mm.minimize_method = 'BFGS'
#==============================================================
#//////////////////////////////////////////////////////////////

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#             Marginal likelihood calculation
#==============================================================
# pars for performing a multivariate normal fit to the
# current posterior to implement Gelfand and Dey's correction
# to the harmonic mean of the likelihood estimator
mm.ml_fac_to_make_smooth = 10   # more samples than posterior
mm.ml_fac_to_reduce_sigma = 2    # skinnier than posterior
#==============================================================
#//////////////////////////////////////////////////////////////

def get_modhyp_from_filename(filename):
    # file should be:
    #
    #  output/YYYY-mm-dd_HHMM_samples_project_subproject_the_mod-hyp.dat
    #
    list_parts = filename.split('_')
    lp = np.array(list_parts)[5:]
    modhyp = ""
    for i in range(len(lp)):
        modhyp += lp[i] + '_'
    modhyp = modhyp.split('.')[0]
    return modhyp

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                      File Handling
#==============================================================
#
# Flush the print output by default
print = functools.partial(print, flush=True)
#
#--- samples file:
#
#     * give samples filename as commandline argument or load here
#
#     * samples file must have been produced by same mod-hyp
#       module as is loaded above.
#
#     * the blobs should also be available (loglike, logprior) in
#       the standard file names output by the mcmc script.
#
if (len(sys.argv) > 1):
    file_sample = sys.argv[1]
else:
    print("***Error: must give the samples data filename as argument")
#
#--- Check that the model hypothesis of the samples matches
#    that of the imported mod-hyp module
modhyp = get_modhyp_from_filename(file_sample)
print("\nModel Hypothesis [mod-hyp module] = " + mod.model_hyp)
print("Model Hypothesis [samples]        = " + modhyp)
if (modhyp != mod.model_hyp):
    print("***Error: model hypothesis of samples and module must match.")
    exit(0)
else:
    print("\t --> Sample file and module match.  Proceeding...\n")
#
#--- Get the blobs file names
#
file_loglike, file_logprior = mm.get_blob_filenames(file_sample)

#
#--- Create output plot file with same date-time tag as sample file
#
parts = file_sample.split('/')[1].split('_')
sample_tag = parts[0] + "_" + parts[1]  #date-time of the sample file
plotdir = "plots/"
file_cornerplot_all = plotdir + sample_tag + "_" + "cornerplot_all" + ".pdf"
file_cornerplot_imp = plotdir + sample_tag + "_" + "cornerplot_imp" + ".pdf"
#//////////////////////////////////////////////////////////////


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                      MAIN CODE
#==============================================================
#--- Print info to user/logfile
dividerstr = 65*"="
#
#--- Get the initial guess for the maximum a posteriori point, while:
#     Initializing model and data modules, and print their parameters to user/logfile
print(dividerstr)
initial_theta = mod.initialize(indentstr="  ", dividerstr=dividerstr)
print(dividerstr)
#
#--- Get the parameter names and, specifically, the indices,
#    names, and number of the "important" (non-nuisance) parameters
#
mm.par_names_print = mod.get_pars('print', justnames=True)
mm.par_names_plot = mod.get_pars('plot', justnames=True)
mm.ind_important_pars, junk = mod.get_index_important_and_discrete_pars()
mm.imp_par_names_print = mm.par_names_print[mm.ind_important_pars]
mm.imp_par_names_plot = mm.par_names_plot[mm.ind_important_pars]
mm.N_imp_pars = len(mm.ind_important_pars)    

#
#--- Find the maximum of the posterior function (or likelihood)
#
print(dividerstr)
# Have the option to do just maximum likelihood (not recommended)
if mm.do_maximum_likelihood:
    print("Seeking maximum of the likelihood function...")
    # function for the negative log likelihood
    neglogprob = lambda *args: -1.0*mod.log_like_and_prior(*args)[0]
else:
    print("Seeking maximum of the posterior density...")        
    # function for negative log *posterior*
    neglogprob = lambda *args: -1.0*np.sum( mod.log_like_and_prior(*args) )
# Find the minimum of the negative log-posterior/likelihood
pars_MAP = mm.get_maximum_a_posteriori_estimate(neglogprob, initial_theta)
# Transform the parameter values from mcmc-transformed to printable
mod.set_pars(pars_MAP)
theta_transformed = mod.get_pars('print')
# Print MAP info to user
mm.print_maximum_a_posteriori_estimate(theta_transformed)

#
#--- Transform the samples (for printing and plotting)
#
print(dividerstr)
print("Loading the samples from file...")
# load the samples and blobs
flat_samples = np.loadtxt(file_sample)
print(f"Found {len(flat_samples):d} samples")
flat_samples_loglike = np.loadtxt(file_loglike)
flat_samples_logprior = np.loadtxt(file_logprior)
# transform the samples
print("Transforming the samples for printing and plotting...")
flat_samples_transformed_print = []
flat_samples_transformed_plot = []
for i in range(len(flat_samples)):
    mod.set_pars(flat_samples[i])    
    flat_samples_transformed_print.append(mod.get_pars('print'))
    flat_samples_transformed_plot.append(mod.get_pars('plot'))
flat_samples_transformed_print = np.array(flat_samples_transformed_print)
flat_samples_transformed_plot = np.array(flat_samples_transformed_plot)

#
#--- Output median + 68% percentiles
#
print(dividerstr)
mm.print_percentiles(flat_samples_transformed_print, [16, 50, 84])

#
#--- Make corner plots (of "important" and "all" parameters
#
print(dividerstr)
mod.set_pars(pars_MAP)
pars_MAP_transformed = mod.get_pars('plot')
print("Creating cornerplots of \"important\" and all parameters...")
figCI, figCA = mm.get_corner_plots(flat_samples_transformed_plot,
                                   pars_MAP_transformed)

#
#--- Get a multivariate normal fit to the posterior (MVN)
#
#     (to be overplotted on the cornerplots, and
#      used for the G&D marginal likelihood)
#
print(dividerstr)
print("Getting the best-fit multivariate normal to the posterior")
mvn, mvn_samples = \
    mm.get_bestfit_mvn(flat_samples)

#
#--- Transform the MVN samples for plotting on cornerplot
#
print(dividerstr)
print("Transforming the MVN samples for plotting the G&D function")
mvn_samples_transformed = []
for i in range(len(mvn_samples)):
    mod.set_pars(mvn_samples[i])
    mvn_samples_transformed.append(mod.get_pars('plot'))
mvn_samples_transformed = np.array(mvn_samples_transformed)

#
#--- Overplot the MVN on the cornerplt and save figures
#
ndim = len(flat_samples[0])
mm.overplot_mvn_on_cornerplots(figCI, figCA, mvn_samples_transformed, ndim)
print("Saving cornerplot figures to\n\t" + file_cornerplot_imp
      + "\n\t" + file_cornerplot_all)
figCI.savefig(file_cornerplot_imp)
figCA.savefig(file_cornerplot_all)
#
#--- Calculate marginal likelihood and print to user
#
print(dividerstr)
a, b = mm.calc_marg_like(flat_samples, flat_samples_loglike,
                         flat_samples_logprior, mvn)
