import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.optimize as spo
import scipy.stats as sps
import scipy.special as spsp
import scipy.spatial as spspat
import emcee
import corner
import multiprocessing as mp
import os
import functools
import shutil
import datetime as dt

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#            Import the chosen model hypothesis
#
#  A model hypothesis module contains the specific method of
#  calculating the log-posterior, which is the sum of:
#
#     * conditional log-prior of the I_pop pars with respect
#       to the current population distribution
#     * log-prior for the Pop (pop dist) parameters
#     * log-prior for I_nopop parameters
#     * log-prior of any Other parameters
#
#  and
#
#     * the log-likelihood of the data given these parameters
#
#  It gets the log-likelihood calculation from the data module,
#  which depends on everything but the Pop parameters.
#
#  Model hypothesis modules are associated to a "data analysis
#  subproject" of a "data project", and should be named
#  accordingly, e.g.,
#
#     module_<project>_modhyp_<subproject-name>_<model-hyp-name>
#
#==============================================================
#import module_sleepstudy_modhyp_REmb_nocov_COMsig as mod
import module_sleepstudy_modhyp_REbsigma_COMm as mod
#import module_sleepstudy_modhyp_REb_COMmsig as mod
#import module_sleepstudy_modhyp_REm_COMbsig as mod
#import module_sleepstudy_modhyp_REnone_COMsig as mod
#import module_sleepstudy_modhyp_REnone_INDIVmbsig as mod
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
start_time = dt.datetime.now()
# Don't allow, e.g., numpy to parallelize
#
#   although Kelvin says maybe not to trust this advice:
#        https://github.com/bmcguir2/molsim/issues/15
#
os.environ["OMP_NUM_THREADS"] = "1"
ncpu = mp.cpu_count()
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
do_maximum_likelihood = False # otherwise, maximum a *posteriori*
minimize_tol = 1e-12
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
minimize_method = 'BFGS'
#==============================================================
#//////////////////////////////////////////////////////////////


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                MCMC run and output parameters
#==============================================================
#
#--- save to a backend or not  (add a lot of time)
#
mcmc_do_backend = False
# backend file for restart/analysis
#   (within the chains directory, defined below)
mcmc_existing_chain_file = ""
#
#--- do parallel processing
#
mcmc_do_multiprocessing = True
mcmc_ncpus_to_use = None # (if None, uses max available)
# show the progress bar
#    (not rec for non-interactive runs, writing to log file)
mcmc_show_progress = False
# sample to convergence (otherwise: run for mcmc_N_steps_max)
mcmc_run_to_convergence = True
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
mcmc_run_to_convergence_check_steps = 5000 # check convergence every __ steps (mult of 10)
mcmc_run_to_convergence_N_autocorrs = 60  # should be >= 50
mcmc_run_to_convergence_delta_tau_frac = 0.05 # should be <~ 0.05
mcmc_run_to_convergence_verbose = True  # print autocorr and taufrac
mcmc_run_to_convergence_plot_autocorr = True # make a plot of autocorr vs iterations each check
# marginal likelihood check
mcmc_run_to_convergence_get_marginal_likelihood = True # calc ML 10x every conv check
mcmc_run_to_convergence_marg_like_checks_per_conv = 10 # check_steps / checks per conv = integer
# output of (last few autocorr times of) raw MCMC chains
mcmc_run_to_convergence_plot_chains = True # plot raw timeseries for important pars (last few taus)
mcmc_run_to_convergence_save_chains = False # save raw timeseries for important pars (last few taus)
mcmc_run_to_convergence_chains_max_taus = 2 # the last __ autocorr times of chain will be plotted/saved
# output of (current approximate) independent samples
mcmc_run_to_convergence_save_samples = True # plot and save *independent sample* of important pars
mcmc_save_samples_max = 10000 # number of samples to save to file
#
# Ignore non-independence in the initial bundle
#
#    (I originally set this to True, I think because I had a zero value
#     for one of the parameter guesses that was therefore not able to be
#     spread out into a bundle.  Since then, I haven't had a problem...
#     best to leave this as False unless the problem appears again.)
#
mcmc_skip_initial_state_check = False
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
# number of walkers is set below after mod-hyp module is initialized
mcmc_N_walkers = None
mcmc_N_steps_max = 500000
mcmc_N_steps_restart = 20000 # for restarting a chain (mcmc_run_to_convergence = False)
mcmc_initial_bundle_fractional_width = 1e-1
mcmc_burnin_drop = 2 # in number of autocorrelation times
mcmc_thin_by = 0.5 # keep data at only every XX autocorrelation times
#==============================================================
#//////////////////////////////////////////////////////////////

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#             Marginal likelihood calculation
#==============================================================
# pars for performing a multivariate normal fit to the
# current posterior to implement Gelfand and Dey's correction
# to the harmonic mean of the likelihood estimator
ml_fac_to_make_smooth = 100   # more samples than posterior
ml_fac_to_reduce_sigma = 2    # skinnier than posterior
#==============================================================
#//////////////////////////////////////////////////////////////


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#        Info on important (non-nuisance) parameters
#==============================================================
# all set below after mod-hyp module initializes
ind_imp_pars = None
imp_par_names = None
N_imp_pars = None

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                Figure/Plotting options
#==============================================================
# put the max-likelihood/posterior on cornerplot as "truths"
#     (otherwise no "truths")
cornerplot_plot_maxvals = True 
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
    input_backend_file = chaindir + mcmc_existing_chain_file
chainfile = 'chain' + '_' + file_suffix + '.h5'
file_chain = chaindir + timestring + '_' + chainfile
if (runmode in ['analysis', 'restart']):
    # if doing analysis or restart, use the provided backend
    file_chain = input_backend_file
#
#--- Output files
#
plotdir = "plots/"
datadir = "output/"
autocorr_fig_file = plotdir + timestring + "_" \
    + "autocorr" + "_" + file_suffix + ".pdf"
autocorr_data_file = datadir + timestring + "_" \
    + "autocorr" + "_" + file_suffix + ".dat"  # txt file
marglike_fig_file = plotdir + timestring + "_" \
    + "marglike" + "_" + file_suffix + ".pdf"
marglike_data_file = datadir + timestring + "_" \
    + "marglike" + "_" + file_suffix + ".dat"
chains_fig_file = plotdir + timestring + "_" \
    + "chains" + "_" + file_suffix + ".pdf"
chains_data_file = datadir + timestring + "_" \
    + "chains" + "_" + file_suffix + ".npy"  # numpy file
samples_fig_file = plotdir + timestring + "_" \
    + "samples" + "_" + file_suffix + ".pdf"
samples_data_file = datadir + timestring + "_" \
    + "samples" + "_" + file_suffix + ".dat"
cornerplot_fig_file = plotdir + timestring + "_" \
    + "cornerplot" + "_" + file_suffix +  ".pdf"
#==============================================================
#//////////////////////////////////////////////////////////////


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                       Subroutines
#==============================================================
#//////////////////////////////////////////////////////////////
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

def get_maximum_a_posteriori_estimate(initial_theta):
    # option to do just maximum likelihood (not recommended)
    if do_maximum_likelihood:
        print("Seeking maximum of the likelihood function...")
        # function for the negative log likelihood
        negloglik = lambda *args: -1.0*mod.log_like_and_prior(*args)[0]
    else:
        print("Seeking maximum of the posterior density...")        
        # function for negative log *posterior*
        negloglik = lambda *args: -1.0*np.sum( mod.log_like_and_prior(*args) )
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
    # Print MAP info to user
    if do_maximum_likelihood:
        print("Maximum likelihood estimate (using minimize_method = " + minimize_method + "): ")
    else:
        print("Maximum posterior estimates (using minimize_method = " + minimize_method + "): ")
    # Set the parameters in the model module
    mod.set_pars(theta_ml)
    # Get back the associated names and transformed values
    theta_transformed, par_names = mod.get_pars('print', getnames=True)
    # Print names and values to user/logfile
    for i in range(len(par_names)):
        print('\t' + par_names[i] + f" = {theta_transformed[i]:0.4e}")
    # Return the untransformed numpy vector for use in initializing MCMC
    return theta_ml

def log_posterior_probability(theta):
    """ 
    Get the posterior density from the model hypothesis module
       (up to the marginal likelihood normalizing constant)
    """
    # evaluate likelihood and prior
    ll, lp = mod.log_like_and_prior(theta)
    # if theta not in prior, return posterior probability of zero
    if not np.isfinite(lp):
        return -np.inf
    # otherwise return the (log of the) product of prior and likelihood
    # update (2022-05-26): add "blobs" for likelihood and prior individually
    return lp + ll, ll, lp

def print_percentiles(flat_samples_transformed, pctile_values):
    # get names of the transformed variables
    junk, par_names = mod.get_pars('print', getnames=True)    
    print("The " + str(pctile_values) + " percentiles are:")
    for i in range(len(par_names)):
        pctiles = np.percentile(flat_samples_transformed[:,i], pctile_values)
        print('\t' + par_names[i] + \
              f" = {pctiles[1]:0.4e} [{pctiles[0]:0.4e}:{pctiles[2]:0.4e}]")

def get_mcmc_N_steps(restart):
    if restart:
        return mcmc_N_steps_restart
    else:
        return mcmc_N_steps_max
    
def get_init_pos(pars_map):
    n_dim = len(pars_map)
    pos = ( pars_map + np.abs(pars_map)*mcmc_initial_bundle_fractional_width
            * np.random.randn(mcmc_N_walkers, n_dim))
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
                
def conv_check_get_marginal_likelihood(flat_samples):
    mvn = sps.multivariate_normal(np.mean(flat_samples, axis=0),
                                  np.cov(flat_samples, rowvar=0)/
                                  ml_fac_to_reduce_sigma**2)
    hm, gd = calc_marg_like(flat_samples, mvn, convergence_check=True)
    return hm, gd

def plot_and_save_chains(sampler, tau_max):
    # Plot the chains of each "important" parameter
    # Get the last few autocorr times of the chain
    Npoints = int(mcmc_run_to_convergence_chains_max_taus * tau_max)
    # get only the last bit of the chain for only those parameters
    samples = sampler.get_chain()[-Npoints:]
    # make a figure
    plt.close('all')
    figCh, axes = plt.subplots(N_imp_pars, figsize=(10, 7), sharex=True)
    if (N_imp_pars > 1):
        for i in range(N_imp_pars):
            ax = axes[i]
            ind = ind_imp_pars[i]
            ax.plot(samples[:, :, ind], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(imp_par_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel(f"step number (of last {Npoints:d})")
    else:
        ind = ind_imp_pars[0]        
        axes.plot(samples[:, :, ind], "k", alpha=0.3)
        axes.set_xlim(0, len(samples))
        axes.set_ylabel(imp_par_names[0])
        axes.yaxis.set_label_coords(-0.1, 0.5)
        axes.set_xlabel(f"step number (of last {Npoints:d})")
    figCh.savefig(chains_fig_file)
    print(f"  Plotted the last {Npoints:d} of the chains.")        
    # Save up last bit of the chain
    if mcmc_run_to_convergence_save_chains:
        print(f"  Saved the last {Npoints:d} of the chain to file.")    
        np.save(chains_data_file, samples)

def plot_and_save_samples(sampler, tau_max):
    # get independent samples
    burnin, thin = get_burn_thin(tau_max)
    flat_samples = sampler.get_chain(discard=burnin,
                                     thin=thin, flat=True)
    nsamps = len(flat_samples)
    # but keep only the last few of them (~1000)
    if ( nsamps > mcmc_save_samples_max ):
        Npoints = mcmc_save_samples_max
    else:
        Npoints = nsamps
    # make a figure showing the important parameters
    plt.close('all')
    figSa, axes = plt.subplots(N_imp_pars, figsize=(10, 7), sharex=True)
    if (N_imp_pars > 1):
        for i in range(N_imp_pars):
            ax = axes[i]
            ind = ind_imp_pars[i]
            ax.plot(flat_samples[-1*Npoints:,ind], "k", alpha=0.3)
            ax.set_xlim(0, Npoints)
            ax.set_ylabel(imp_par_names[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel(f"steps / thin (last {Npoints:d} samples)")
    else:
        ind = ind_imp_pars[0]
        axes.plot(flat_samples[-1*Npoints:,ind], "k", alpha=0.3)
        axes.set_xlim(0, Npoints)
        axes.set_ylabel(imp_par_names[0])
        axes.yaxis.set_label_coords(-0.1, 0.5)
        axes.set_xlabel(f"steps / thin (last {Npoints:d} samples)")
    figSa.savefig(samples_fig_file)
    # Save up to mcmc_save_samples_max samples (all parameters)
    print(f"  Plotted and saved the last {Npoints:d} samples.")
    np.savetxt(samples_data_file, flat_samples[-1*Npoints:,:])
        
def plot_and_save_marginal_likelihood(n_steps, lml_hm, lml_gd):
    # Plot log(marg_like) vs steps
    maxval = np.concatenate((lml_hm, lml_gd)).max()
    minval = np.concatenate((lml_hm, lml_gd)).min()           
    plt.close('all')
    figml, axml = plt.subplots(figsize=(10,8))    
    axml.plot(n_steps, lml_hm, '-ro')
    axml.plot(n_steps, lml_gd, '-bo')
    axml.set_xlim(0, n_steps[-1])
    axml.set_ylim(minval - 0.1 * (maxval - minval), maxval + 0.1 * (maxval - minval))
    axml.yaxis.get_ticklocs(minor=True)
    axml.minorticks_on()
    axml.grid(True, which='both')
    axml.set_xlabel("number of steps")
    axml.set_ylabel("logML_hm (r), logML_gd (b)")
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
            if ( ( (Nsteps % margsteps) == 0 )
                 & mcmc_run_to_convergence_get_marginal_likelihood
                 & first_tau_checked ):
                #--- 10 times every convergence check: do marginal likelihood check
                burnin, thin = get_burn_thin(np.max(old_taus))
                if (Nsteps > 2*burnin):
                    flat_samples = sampler.get_chain(discard=burnin,
                                                     thin=thin, flat=True)
                    # calculate and print ML to user
                    print(f"  ({Nsteps:d})", end="")
                    hm, gd = conv_check_get_marginal_likelihood(flat_samples)
                    log_marg_like_hm.append(hm)
                    log_marg_like_gd.append(gd)
                    n_steps_ml.append(Nsteps)
                    # plot marginal likelihood vs steps; save plot and data
                    plot_and_save_marginal_likelihood(n_steps_ml, log_marg_like_hm, log_marg_like_gd)
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
                # calculate and print ML to user
                hm, gd = conv_check_get_marginal_likelihood(flat_samples)
                log_marg_like_hm.append(hm)
                log_marg_like_gd.append(gd)
                n_steps_ml.append(sampler.iteration)
                # plot marginal likelihood vs steps; save plot and data
                plot_and_save_marginal_likelihood(n_steps_ml, log_marg_like_hm, log_marg_like_gd)
        index += 1
        # if reached convergence then done
        if converged:
            break

def do_sampling(init_pos, n_dim, backend = None, pool = None, restart=False):
    print("Creating EnsembleSampler...")
    # blobs for log_likelihood and log_prior in the log_posterior_prob return
    dtype = [("log_like", float), ("log_prior", float)]
    # define the sampler
    sampler = emcee.EnsembleSampler(
        mcmc_N_walkers, n_dim, log_posterior_probability,
        pool=pool, backend=backend, blobs_dtype=dtype
    )
    # checking convergence on the fly or just running for max steps
    if mcmc_run_to_convergence:
        print("Running sampler to convergence ", end='')
        print(f"(N_autocorr = {mcmc_run_to_convergence_N_autocorrs:d}; ", end='')
        print(f"Delta_autocorr_frac = {mcmc_run_to_convergence_delta_tau_frac:.3f}")
        print(f"with max steps {mcmc_N_steps_max:d}, ", end='')
        print(f"checking convergence every {mcmc_run_to_convergence_check_steps:d} ...\n")
        sample_to_convergence(sampler, init_pos, restart=restart)
    else:
        mcmc_N_steps = get_mcmc_N_steps(restart)
        print(f"Running sampler for {mcmc_N_steps:d} steps ... \n")
        sampler.run_mcmc(init_pos, mcmc_N_steps, progress=mcmc_show_progress,
                         skip_initial_state_check = mcmc_skip_initial_state_check)
    return sampler

def run_mcmc_sampler(pars_map, backend, restart=False):
    # Initialize walkers in small bundle about MAP position
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
            sampler = do_sampling(initial_pos, n_dim,
                                  backend = backend, pool = pool, restart=restart)
    else:
        # no pool
        print(f"--> Single-thread processing")
        sampler = do_sampling(initial_pos, n_dim, backend = backend, restart=restart)
    return sampler, n_dim

def calc_marg_like(flat_samples, mvn, convergence_check=False):
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
    ls = []
    ps = []
    hs = []
    for i in range(Nsamp):
        ll, lp = mod.log_like_and_prior(flat_samples[i])        
        lh = np.log(hdist(flat_samples[i]))
        ls.append(ll)
        ps.append(lp)
        hs.append(lh)
    ls = np.array(ls)
    ps = np.array(ps)
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
#==============================================================
#//////////////////////////////////////////////////////////////


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                      MAIN CODE
#==============================================================
#--- Print info to user/logfile
dividerstr = "==============================================================="
print(dividerstr)
print(f"Random seed = {theseed:d}")
print("Start time = " + str(start_time))
print("Run mode = " + runmode)
print("Backend = " + str(mcmc_do_backend))
print("If restart/analysis, existing chain file is:")
if (mcmc_existing_chain_file == ""):
    print("\t<none>")
else:
    print("\t" + mcmc_existing_chain_file)
print("Multiprocessing = " + str(mcmc_do_multiprocessing))
print("\tmcmc_ncpus_to_use = " + str(mcmc_ncpus_to_use))
print("mcmc_run_to_convergence = " + str(mcmc_run_to_convergence))
print("mcmc_run_to_convergence_N_autocorrs = " + str(mcmc_run_to_convergence_N_autocorrs))
print("mcmc_run_to_convergence_delta_tau_frac = " + str(mcmc_run_to_convergence_delta_tau_frac))
print("mcmc_run_to_convergence_check_steps = " + str(mcmc_run_to_convergence_check_steps))
print("mcmc_run_to_convergence_get_marginal_likelihood = " + str(mcmc_run_to_convergence_get_marginal_likelihood))
print("mcmc_run_to_convergence_marg_like_checks_per_conv = " + str(mcmc_run_to_convergence_marg_like_checks_per_conv))
print("mcmc_skip_initial_state_check = " + str(mcmc_skip_initial_state_check))
print("mcmc_N_steps_max = " + str(mcmc_N_steps_max))
print("mcmc_N_steps_restart = " + str(mcmc_N_steps_restart))
print("mcmc_initial_bundle_fractional_width = " + str(mcmc_initial_bundle_fractional_width))
print("mcmc_burnin_drop = " + str(mcmc_burnin_drop))
print("mcmc_thin_by = " + str(mcmc_thin_by))

#
#--- Get the initial guess for the maximum a posteriori point, while:
#     Initializing model and data modules, and print their parameters to user/logfile
print(dividerstr)
initial_theta = mod.initialize(indentstr="  ", dividerstr=dividerstr)
print(dividerstr)
#
#--- Get a bit more information from the mod-hyp module
#
#      * set the number of walkers based on the number of parameters
#        (and cpus)
#
#      * get info on the "important" (non-nuisance) parameters
# 
mcmc_N_walkers = int( ncpu * np.ceil( 2 * len(initial_theta) / ncpu) )
#--- Maximum number of steps per walker
print("mcmc_N_walkers = " + str(mcmc_N_walkers))
# get the indices, names, and number of the non-nuisance parameters
ind_imp_pars = mod.get_ind_imp_pars()
junk, par_names = mod.get_pars('plot', getnames=True)
imp_par_names = par_names[ind_imp_pars]
N_imp_pars = len(ind_imp_pars)    
print("Important parameters to track are:\n", imp_par_names)

#
#--- Find the maximum of the posterior function (or likelihood)
#
print(dividerstr)
pars_MAP = get_maximum_a_posteriori_estimate(initial_theta)
if (runmode == "MAPonly"):
    exit(0)
                                      
#==============================================================
#           Run MCMC (new run or restart) or do analysis                 
#==============================================================
print(dividerstr)
if (runmode == 'newrun'):
    # Run emcee's MCMC to generate sample of the posterior
    # create a backend for saving the chain (or not)
    if mcmc_do_backend:
        backend = create_backend()
    else:
        backend = None
    sampler, n_dim = run_mcmc_sampler(pars_MAP, backend)
elif (runmode == 'restart'):
    # print info to user
    print("Loading backend from " + file_chain + " with:")
    # load previous MCMC run from file for analysis
    new_backend = emcee.backends.HDFBackend(file_chain)
    (Nsteps, Nchains, Npars) = sampler.get_chain().shape
    print(f"\tNsteps, Nchains, Npars = {Nsteps:d}, {Nchains:d}, {Npars:d}")
    sampler, n_dim = run_mcmc_sampler(pars_MAP, new_backend, restart=True)    
elif (runmode == 'analysis'):
    # print info to user
    print("Loading backend from " + file_chain + " with:")
    # load previous MCMC run from file for analysis
    sampler = emcee.backends.HDFBackend(file_chain)
    (Nsteps, Nchains, Npars) = sampler.get_chain().shape
    n_dim = Npars
    print(f"\tNsteps, Nchains, Npars = {Nsteps:d}, {Nchains:d}, {Npars:d}")

#==============================================================
#          Post-processing (or 'analysis' processing)
#==============================================================

#--- Output autocorrelation times for each parameter
junk, par_names = mod.get_pars('print', getnames=True)
print(dividerstr)
print("Getting autocorrelation times...")
if (runmode == 'analysis'):
    taus = sampler.get_autocorr_time(tol=0)
else:
    taus = sampler.get_autocorr_time()
print("Sampler autocorrelation times are:")
for i in range(len(taus)):
    outstr = "\ttau_" + par_names[i]
    outstr += f" = {taus[i]:0.2f}"
    print(outstr)

#--- Flatten the samples into single list of samples from posterior, and:
#
#   - drop the burn-in (2x the autocorrelation time of initial steps)
#   - "thin" the results (Use only data only every 1/2 autocorrelation time)
#
tau_max = np.max(taus)
tau_min = np.min(taus)
burnin, thin = get_burn_thin(tau_max)
print(dividerstr)
print(f"Preparing independent samples: discarding burn-in of {burnin:d} and thinning by {thin:d}")
flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
flat_logprob = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
flat_blobs = sampler.get_blobs(discard=burnin, thin=thin, flat=True)
flat_log_like = flat_blobs["log_like"]
flat_log_prior = flat_blobs["log_prior"]
print("Shape of flattened samples is", flat_samples.shape)
maxinds = get_index_of_top_N_vals(flat_logprob, 10)
MAPs = flat_logprob[maxinds]
print("The top 10 largest posterior values among the samples are:")
for i in range(10):
    print(f"{MAPs[i]:.5e} with random-effect parameters:\n\t", end='')
    print(flat_samples[maxinds[i],ind_imp_pars])
tree = spspat.KDTree(flat_samples)
dist, parind = tree.query(pars_MAP)
print(f"The closest sample to the maximum likelihood theta has:")
print(flat_samples[parind,ind_imp_pars])
print(f"With log(posterior) = {flat_logprob[parind]:.5e}")

#--- Transform the samples (for printing and plotting)
print(dividerstr)
print("Transforming the samples for printing and plotting...")
flat_samples_transformed_print = []
flat_samples_transformed_plot = []
for i in range(len(flat_samples)):
    mod.set_pars(flat_samples[i])    
    flat_samples_transformed_print.append(mod.get_pars('print'))
    flat_samples_transformed_plot.append(mod.get_pars('print'))
flat_samples_transformed_print = np.array(flat_samples_transformed_print)
flat_samples_transformed_plot = np.array(flat_samples_transformed_plot)

#--- Output median + 68% percentiles
print_percentiles(flat_samples_transformed_print, [16, 50, 84])

#--- Make a corner plot of the "random effect" parameters
flat_samples_re = flat_samples_transformed_plot[:,ind_imp_pars]
if cornerplot_plot_maxvals:
    mod.set_pars(pars_MAP)    
    pars_MAP_transformed = mod.get_pars('plot')
    fig2 = corner.corner(flat_samples_re, labels=imp_par_names,
                         truths=pars_MAP_transformed[ind_imp_pars])
else:
    fig2 = corner.corner(flat_samples_re, labels=imp_par_names)

#--- Get a multivariate normal fit to the posterior
#     (overplotted on cornerplot, used for G&D Marginal likelihood)
#
# find bestfit multivariate normal to full distribution (untransformed)
mvn = sps.multivariate_normal(np.mean(flat_samples, axis=0),
                              np.cov(flat_samples, rowvar=0)/
                              ml_fac_to_reduce_sigma**2)
# generate samples from mvn for plotting
mvn_samples = mvn.rvs(size=ml_fac_to_make_smooth*flat_samples.shape[0])
# transform the mvn samples for plotting
print("Transforming the MVN samples for plotting the G&D function")
mvn_samples_transformed = []
for i in range(len(mvn_samples)):
    mod.set_pars(mvn_samples[i])
    mvn_samples_transformed.append(mod.get_pars('plot'))
mvn_samples_transformed = np.array(mvn_samples_transformed)
# overplot just the important parameters on cornerplot
if (len(ind_imp_pars) == 1):
    iip = np.concatenate( (ind_imp_pars, [ind_imp_pars[0]-1]) )
    corner.corner(mvn_samples_transformed[:,iip],
                  fig=fig2, color='red',
                  plot_datapoints=False, plot_density=False,
                  hist_kwargs = {'alpha' : 0.0})
    # reset the yrange on the 1-D histograms
    ndim = len(flat_samples[0])
    axes = np.array(fig2.axes).reshape((2,2))
    for i in range(2):
        ax = axes[i, i]
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax/ml_fac_to_make_smooth)
else:
    corner.corner(mvn_samples_transformed[:,ind_imp_pars],
                  fig=fig2, color='red',
                  plot_datapoints=False, plot_density=False,
                  hist_kwargs = {'alpha' : 0.0})
    # reset the yrange on the 1-D histograms
    ndim = len(flat_samples[0])
    axes = np.array(fig2.axes).reshape((N_imp_pars, N_imp_pars))
    for i in range(N_imp_pars):
        ax = axes[i, i]
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax/ml_fac_to_make_smooth)

#--- Calculate marginal likelihood and print to user
print(dividerstr)
a, b = calc_marg_like(flat_samples, mvn)
#--- Save figures
print(dividerstr)
print("Saving cornerplot figure to\n\t" + cornerplot_fig_file)
fig2.savefig(cornerplot_fig_file)
#==============================================================
#//////////////////////////////////////////////////////////////
