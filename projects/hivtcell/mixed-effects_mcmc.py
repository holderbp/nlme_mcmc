import numpy as np
import pandas as pd
import os
import functools
import datetime as dt
import emcee

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
#  Model hypothesis modules are associated to a "data analysis
#  subproject" of a "data project", and should be named
#  accordingly, e.g.,
#
#     module_<project>_modhyp_<subproject-name>_<model-hyp-name>
#
#==============================================================
#
#=== Virus decay
#
#import module_hivtcell_modhyp_virusdecay_IPOP_none_INOPOP_A_B_thalf_COM_sigma as mod
#import module_hivtcell_modhyp_virusdecay_IPOP_none_INOPOP_A_B_COM_thalf_sigma as mod
#import module_hivtcell_modhyp_virusdecay_IPOP_thalf_INOPOP_A_B_COM_sigma as mod
#import module_hivtcell_modhyp_virusdecay_IPOP_thalf_sigma_INOPOP_A_B_COM_none as mod
#import module_hivtcell_modhyp_virusdecay_IPOP_sigma_INOPOP_A_B_COM_thalf as mod
#
#=== uninfected timeseries
#
#import module_hivtcell_modhyp_ecp_uninf_timeseries_IPOP_none_INOPOP_N_tauT_tD_COM_none_CONST_nT as mod
#import module_hivtcell_modhyp_ecp_uninf_timeseries_IPOP_none_INOPOP_N_tauT_tD_COM_sigma_CONST_nT as mod
import module_hivtcell_modhyp_ecp_uninf_timeseries_IPOP_none_INOPOP_N_tauT_nT_tD_COM_sigma_CONST_s as mod
#==============================================================
#//////////////////////////////////////////////////////////////


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                   Initial Set-up
#==============================================================
# flush the print output by default (for running remotely)
print = functools.partial(print, flush=True)
#--- get a random seed and print it out to user
theseed = np.random.randint(low=0, high=1000000)
# or set the value of the seed (comment out if not)
theseed = 412611
np.random.seed(theseed)
# get starttime
mm.start_time = dt.datetime.now()
# Don't allow, e.g., numpy to parallelize
#
#   although Kelvin says maybe not to trust this advice:
#        https://github.com/bmcguir2/molsim/issues/15
#
os.environ["OMP_NUM_THREADS"] = "1"
#==============================================================
#//////////////////////////////////////////////////////////////

            
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                     Runmode options                          
#==============================================================
runmode = 'newrun' # ['newrun', 'analysis', 'restart', 'MAPonly']
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
#                MCMC run and output parameters
#==============================================================
#
#--- save to a backend or not  (add a lot of time)
#
mm.mcmc_do_backend = False
# backend file for restart/analysis
#   (within the chains directory, defined below)
mm.mcmc_existing_chain_file = ""
#
#--- change the moves (default (None) is stretch w/ a=2.0)
#
#      'default' - stretch w/ a=2.0
#      'stretch_5.0' - stretch w/ a=5.0
#
#
mcmc_moves = None  # default (stretch, a=2)
#mcmc_moves = emcee.moves.StretchMove(a=1.2)
#mcmc_moves = [
#        (emcee.moves.DEMove(), 0.8),
#        (emcee.moves.DESnookerMove(), 0.2),
#    ]
#
#--- do parallel processing
#
mm.mcmc_do_multiprocessing = True
mm.mcmc_ncpus_to_use = None # (if None, uses max available)
# show the progress bar
#    (not rec for non-interactive runs, writing to log file)
mm.mcmc_show_progress = False
# sample to convergence (otherwise: run for mcmc_N_steps_max)
mm.mcmc_run_to_convergence = True
#
#--- Sample until convergence parameters
#
#  Definition (adapted from emcee tutorial):
#
#     ( np.all(tau * N_autocorrs < sampler.iteration) )
#        [evolve for many (>50) autocorrelation times]
#
#       &
#
#     ( np.all(tau[i] - tau[i-1])/tau[i] < tau_frac )
#        [estimate of autocorrelation time not changing]
#
# Warning: The check takes longer and longer (20s at 50000 steps?),
#          because of the time to find the integrated autocorr times,
#          so make the check_steps is pretty large to avoid wasting
#          a lot of time.  Although if the likelihood fcn is
#          expensive this probably doesn't matter much... adjust as
#          needed.
#
# Caveat:  Most of the info to user is provided every check_steps,
#          so you do want to see that fairly often (in human time).
#          (Marginal likelihood is checked more often though.)
#
# basic pars for convergence check
mm.mcmc_run_to_convergence_check_steps = 5000 # check convergence every __ steps (mult of 10)
mm.mcmc_run_to_convergence_N_autocorrs = 60  # should be >= 50
mm.mcmc_run_to_convergence_delta_tau_frac = 0.05 # should be <~ 0.05
mm.mcmc_run_to_convergence_verbose = True  # print autocorr and taufrac
mm.mcmc_run_to_convergence_plot_autocorr = True # make a plot of autocorr vs iterations each check
# marginal likelihood check
mm.mcmc_run_to_convergence_get_marginal_likelihood = True # calc ML 10x every conv check
mm.mcmc_run_to_convergence_marg_like_checks_per_conv = 10 # check_steps / checks per conv = integer
# output of (last few autocorr times of) raw MCMC chains
mm.mcmc_run_to_convergence_plot_chains = True # plot raw timeseries for important pars (last few taus)
mm.mcmc_run_to_convergence_save_chains = False # save raw timeseries for important pars (last few taus)
mm.mcmc_run_to_convergence_chains_max_taus = 2 # the last __ autocorr times of chain will be plotted/saved
# output of (current approximate) independent samples
mm.mcmc_run_to_convergence_save_samples = True # plot and save *independent sample* of important pars
mm.mcmc_save_samples_max = 10000 # number of samples to save to file
#
# Ignore non-independence in the initial bundle
#
#    (I originally set this to True, I think because I had a zero value
#     for one of the parameter guesses that was therefore not able to be
#     spread out into a bundle.  Since then, I haven't had a problem...
#     best to leave this as False unless the problem appears again.)
#
mm.mcmc_skip_initial_state_check = False
#
#--- MCMC walkers/chain parameters
#
#  * Number of walkers should be >= 2*Nparameters
#      (and possibly a multiple of the number of cpus for parallelization)
#
#  * Max number of steps per walkers (will stop earlier at convergence)
#
#  * Bundle size for initial start
#
#  * Burn-in and Thin parameters for getting independent samples from chain
#
# number of walkers
#
#    if None, then set below after mod-hyp module is initialized with:
#          mm.mcmc_N_walkers = int( ncpu * ceil( 2 * npars / ncpu) )
#
mm.mcmc_N_walkers = None
mm.mcmc_N_steps_max = 1000000
mm.mcmc_N_steps_restart = 20000 # for restarting a chain (mcmc_run_to_convergence = False)
mm.mcmc_initial_bundle_fractional_width = 1e-1
mm.mcmc_burnin_drop = 2 # in number of autocorrelation times
mm.mcmc_thin_by = 0.5 # keep data at only every XX autocorrelation times
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
mm.ml_plot_only_GD = True # don't plot the harmonic mean
#==============================================================
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
#                File and output handling
#==============================================================
#--- Start files with date-time string for unique names
now = dt.datetime.now()
timestring = now.strftime("%Y-%m-%d_%H%M")
#--- File suffix based on model hyp and prior choices
file_suffix = mod.project_name + '_' \
    + mod.data_analysis_subproject + '_' + mod.model_hyp 
#
#--- Backend chain files (saving progress and restarting) 
#
#  Warning (2022-05-15): there is an error in the backend.reset(...)
#  which does clear out the chains (if a 'newrun' overwrites a file
#  and then you pause that new run and restart, only the new run
#  iterations are seen.... BUT the actual file on disk just keeps
#  growing in size!!!  I'll need to copy/delete these files manually.
# 
chaindir = "chains/"     # should cull this directory pretty often
if (runmode in ['analysis', 'restart']):
    # must provide a backend filepath
    input_backend_file = chaindir + mm.mcmc_existing_chain_file
chainfile = 'chain' + '_' + file_suffix + '.h5'
mm.file_chain = chaindir + timestring + '_' + chainfile
if (runmode in ['analysis', 'restart']):
    # if doing analysis or restart, use the provided backend
    mm.file_chain = input_backend_file
#
#--- Output files
#
plotdir = "plots/"
datadir = "output/"
mm.autocorr_fig_file = plotdir + timestring + "_" \
    + "autocorr" + "_" + file_suffix + ".pdf"
mm.autocorr_data_file = datadir + timestring + "_" \
    + "autocorr" + "_" + file_suffix + ".dat"  # txt file
mm.marglike_fig_file = plotdir + timestring + "_" \
    + "marglike" + "_" + file_suffix + ".pdf"
mm.marglike_data_file = datadir + timestring + "_" \
    + "marglike" + "_" + file_suffix + ".dat"
mm.chains_fig_file = plotdir + timestring + "_" \
    + "chains" + "_" + file_suffix + ".pdf"
mm.chains_data_file = datadir + timestring + "_" \
    + "chains" + "_" + file_suffix + ".npy"  # numpy file
mm.samples_fig_file = plotdir + timestring + "_" \
    + "samples" + "_" + file_suffix + ".pdf"
mm.samples_hist_file = plotdir + timestring + "_" \
    + "hist-samples" + "_" + file_suffix + ".pdf"
mm.samples_data_file = datadir + timestring + "_" \
    + "samples" + "_" + file_suffix + ".dat"
mm.samples_logprob_data_file = datadir + timestring + "_" \
    + "samples" + "_" + "logprob" + "_" + file_suffix + ".dat"
mm.samples_loglike_data_file = datadir + timestring + "_" \
    + "samples" + "_" + "loglike" + "_" +  file_suffix + ".dat"
mm.samples_logprior_data_file = datadir + timestring + "_" \
    + "samples" + "_" + "logprior" + "_" + file_suffix + ".dat"
mm.cornerplotall_fig_file = plotdir + timestring + "_" \
    + "cornerplotall" + "_" + file_suffix +  ".pdf"
mm.cornerplotimp_fig_file = plotdir + timestring + "_" \
    + "cornerplotimp" + "_" + file_suffix +  ".pdf"
#==============================================================
#//////////////////////////////////////////////////////////////



#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                      MAIN CODE
#==============================================================
#
#--- Print info to user/logfile
#
dividerstr = 65*"="
print(dividerstr)
print(f"Random seed = {theseed:d}")
print("Start time = " + str(mm.start_time))
print("Run mode = " + runmode)
print("Backend = " + str(mm.mcmc_do_backend))
print("If restart/analysis, existing chain file is:")
if (mm.mcmc_existing_chain_file == ""):
    print("\t<none>")
else:
    print("\t" + mm.mcmc_existing_chain_file)
print("Moves = " + str(mcmc_moves))
print("Multiprocessing = " + str(mm.mcmc_do_multiprocessing))
print("\tmcmc_ncpus_to_use = " + str(mm.mcmc_ncpus_to_use))
print("mcmc_run_to_convergence = " + str(mm.mcmc_run_to_convergence))
print("mcmc_run_to_convergence_N_autocorrs = " + str(mm.mcmc_run_to_convergence_N_autocorrs))
print("mcmc_run_to_convergence_delta_tau_frac = " + str(mm.mcmc_run_to_convergence_delta_tau_frac))
print("mcmc_run_to_convergence_check_steps = " + str(mm.mcmc_run_to_convergence_check_steps))
print("mcmc_run_to_convergence_get_marginal_likelihood = " + str(mm.mcmc_run_to_convergence_get_marginal_likelihood))
print("mcmc_run_to_convergence_marg_like_checks_per_conv = " + str(mm.mcmc_run_to_convergence_marg_like_checks_per_conv))
print("mcmc_skip_initial_state_check = " + str(mm.mcmc_skip_initial_state_check))
print("mcmc_N_steps_max = " + str(mm.mcmc_N_steps_max))
print("mcmc_N_steps_restart = " + str(mm.mcmc_N_steps_restart))
print("mcmc_initial_bundle_fractional_width = " + str(mm.mcmc_initial_bundle_fractional_width))
print("mcmc_burnin_drop = " + str(mm.mcmc_burnin_drop))
print("mcmc_thin_by = " + str(mm.mcmc_thin_by))

#
#--- Get the initial guess for the maximum a posteriori point, while:
#     Initializing model and data modules, and print their parameters to user/logfile
print(dividerstr)
initial_theta = mod.initialize(indentstr="  ", dividerstr=dividerstr)
print(dividerstr)
#
#--- Set the number of walkers based on the number of parameters (and cpus)
#
if (mm.mcmc_N_walkers == None):
    mm.mcmc_N_walkers = int( mm.ncpu * np.ceil( 2 * len(initial_theta) / mm.ncpu) )
print("mcmc_N_walkers = " + str(mm.mcmc_N_walkers))
#
#--- Get the parameter names and, specifically, the indices,
#    names, and number of the "important" (non-nuisance) parameters
#
mm.par_names_print = mod.get_pars('print', justnames=True)
mm.par_names_plot = mod.get_pars('plot', justnames=True)
mm.par_names_mcmc = mod.get_pars('mcmc', justnames=True)
mm.par_names_print_short = mod.get_pars('print',
                                        justnames=True, shortnames=True)
mm.par_names_plot_short = mod.get_pars('plot',
                                        justnames=True, shortnames=True)
mm.par_names_mcmc_short = mod.get_pars('mcmc',
                                        justnames=True, shortnames=True)
mm.ind_important_pars, mm.ind_discrete_pars = \
    mod.get_index_important_and_discrete_pars()
mm.imp_par_names_print = mm.par_names_print[mm.ind_important_pars]
if len(mm.ind_discrete_pars > 0):
    mm.discrete_par_names = mm.par_names_print[mm.ind_discrete_pars]
mm.imp_par_names_plot = mm.par_names_plot[mm.ind_important_pars]
mm.imp_par_names_mcmc = mm.par_names_mcmc[mm.ind_important_pars]
mm.N_imp_pars = len(mm.ind_important_pars)    
print("Important parameters to track are:\n", mm.imp_par_names_print)
print("Discrete parameters are:\n", mm.discrete_par_names)

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
# Exit if only doing MAP
if (runmode == "MAPonly"):
    exit(0)
                                      
#==============================================================
#           Run MCMC (new run or restart) or do analysis                 
#==============================================================
print(dividerstr)
# Set the log-likelihood-and-prior function as that
# provided by the model-hypothesis module
mm.mod_log_like_and_prior = mod.log_like_and_prior
if (runmode == 'newrun'):
    # Run emcee's MCMC to generate sample of the posterior
    # create a backend for saving the chain (or not)
    if mm.mcmc_do_backend:
        backend = mm.create_backend()
    else:
        backend = None
    sampler, n_dim = mm.run_mcmc_sampler(pars_MAP, mcmc_moves, backend)
elif (runmode == 'restart'):
    # print info to user
    print("Loading backend from " + mm.file_chain + " with:")
    # load previous MCMC run from file for analysis
    new_backend = emcee.backends.HDFBackend(mm.file_chain)
    (Nsteps, Nchains, Npars) = sampler.get_chain().shape
    print(f"\tNsteps, Nchains, Npars = {Nsteps:d}, {Nchains:d}, {Npars:d}")
    sampler, n_dim = mm.run_mcmc_sampler(pars_MAP, mcmc_moves, new_backend,
                                         restart=True)    
elif (runmode == 'analysis'):
    # print info to user
    print("Loading backend from " + mm.file_chain + " with:")
    # load previous MCMC run from file for analysis
    sampler = emcee.backends.HDFBackend(mm.file_chain)
    (Nsteps, Nchains, Npars) = sampler.get_chain().shape
    n_dim = Npars
    print(f"\tNsteps, Nchains, Npars = {Nsteps:d}, {Nchains:d}, {Npars:d}")

#==============================================================
#          Post-processing (or 'analysis' processing)
#==============================================================
#
#--- Output autocorrelation times for each parameter
#
print(dividerstr)
print("Getting autocorrelation times from sampler.")
if (runmode == 'analysis'):
    taus = mm.get_autocorrelation_times(sampler, ignorewarning=True)
else:
    taus = mm.get_autocorrelation_times(sampler)
print("Sampler autocorrelation times are:")
mm.print_autocorrelation_times(taus)
#
#--- Flatten the samples into single list of samples from posterior, and:
#
#   - drop the burn-in (2x the autocorrelation time of initial steps)
#   - "thin" the results (Use only data only every 1/2 autocorrelation time)
#   - get samples of posterior and the associated "blob" values (prob, prior, like)
#
print(dividerstr)
print("Preparing a set of independent samples.")
[flat_samples, flat_samples_logprob, flat_samples_loglike,
 flat_samples_logprior, burnin, thin] = \
     mm.get_flattened_samples(taus, sampler)
print(f"\tDiscarded a burn-in of {burnin:d} and thinned by {thin:d}")
#
#--- Find the largest posterior values and compare to
#    the max-posterior value found earlier
#
print(dividerstr)
mm.get_top_posterior_values_and_compare(flat_samples_logprob,
                                        flat_samples, pars_MAP)
#
#--- Transform the samples (for printing and plotting)
#
print(dividerstr)
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
#--- Overplot the MVN on the cornerplt
#
ndim = len(flat_samples[0])
mm.overplot_mvn_on_cornerplots(figCI, figCA, mvn_samples_transformed, ndim)
#
#--- Calculate marginal likelihood and print to user
#
print(dividerstr)
a, b = mm.calc_marg_like(flat_samples, flat_samples_loglike,
                         flat_samples_logprior, mvn)
#
#--- Save figures
#
print(dividerstr)
print("Saving cornerplot figures to\n\t" + mm.cornerplotimp_fig_file
      + "\n\t" + mm.cornerplotall_fig_file)
figCI.savefig(mm.cornerplotimp_fig_file)
figCA.savefig(mm.cornerplotall_fig_file)
#==============================================================
#//////////////////////////////////////////////////////////////
