import sys
import functools
import numpy as np
import pandas as pd
import scipy.optimize as spo
import scipy.stats as sps
import scipy.special as spsp
import corner
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#            Import the chosen model hypothesis
#
#   This must match the input "sample" data file below so that
#   the parameters in the sample are correctly identified.
#                                                              
#==============================================================
import module_hivtcell_modhyp_virusdecay_IPOP_none_INOPOP_A_B_COM_thalf_sigma as mod
#//////////////////////////////////////////////////////////////

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                Figure/Plotting options
#==============================================================
# put the max-likelihood/posterior on cornerplot as "truths"
#     (otherwise no "truths")
cornerplot_plot_maxvals = True
# in addition to an "important parameters" plot, make one for all
cornerplot_make_allplot = True
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
#             Marginal likelihood calculation
#==============================================================
# pars for performing a multivariate normal fit to the
# current posterior to implement Gelfand and Dey's correction
# to the harmonic mean of the likelihood estimator
ml_fac_to_make_smooth = 10   # more samples than posterior
ml_fac_to_reduce_sigma = 2    # skinnier than posterior
#==============================================================
#//////////////////////////////////////////////////////////////

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
if (len(sys.argv) > 1):
    file_sample = sys.argv[1]
else:
    file_sample = "output/2022-06-11_2251_samples_sleepstudy_response-time-vs-days-deprivation_no-RE-indiv-mbsigma.dat"
#
#--- Check that the model hypothesis of the samples matches
#    that of the imported mod-hyp module
modhyp = file_sample.split('_')[-1].split('.')[0]
print("\nModel Hypothesis [mod-hyp module] = " + mod.model_hyp)
print("Model Hypothesis [samples]        = " + modhyp)
if (modhyp != mod.model_hyp):
    print("***Error: model hypothesis of samples and module must match.")
    exit(0)
else:
    print("\t --> Sample file and module match.  Proceeding...\n")
#
#--- Create output plot file with same date-time tag as sample file
#
parts = file_sample.split('/')[1].split('_')
sample_tag = parts[0] + "_" + parts[1]  #date-time of the sample file
plotdir = "plots/"
file_cornerplot_all = plotdir + sample_tag + "_" + "cornerplot_all" + ".pdf"
file_cornerplot_imp = plotdir + sample_tag + "_" + "cornerplot_imp" + ".pdf"
#//////////////////////////////////////////////////////////////


def print_percentiles(flat_samples_transformed, pctile_values):
    # get names of the transformed variables
    junk, par_names = mod.get_pars('print', getnames=True)    
    print("The " + str(pctile_values) + " percentiles are:")
    for i in range(len(par_names)):
        pctiles = np.percentile(flat_samples_transformed[:,i], pctile_values)
        print('\t' + par_names[i] + \
              f" = {pctiles[1]:0.4e} [{pctiles[0]:0.4e}:{pctiles[2]:0.4e}]")

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

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                      MAIN CODE
#==============================================================
#--- Print info to user/logfile
dividerstr = "==============================================================="
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

#--- Transform the samples (for printing and plotting)
print(dividerstr)
# load the samples
flat_samples = np.loadtxt(file_sample)
print("Transforming the samples for printing and plotting...")
flat_samples_transformed_print = []
flat_samples_transformed_plot = []
for i in range(len(flat_samples)):
    mod.set_pars(flat_samples[i])    
    flat_samples_transformed_print.append(mod.get_pars('print'))
    flat_samples_transformed_plot.append(mod.get_pars('plot'))
flat_samples_transformed_print = np.array(flat_samples_transformed_print)
flat_samples_transformed_plot = np.array(flat_samples_transformed_plot)

#--- Output median + 68% percentiles
print_percentiles(flat_samples_transformed_print, [16, 50, 84])

#--- Make a corner plot of the "important" parameters
print(dividerstr)
print("Making corner plot of the posterior")
flat_samples_re = flat_samples_transformed_plot[:,ind_imp_pars]
if cornerplot_plot_maxvals:
    mod.set_pars(pars_MAP)    
    pars_MAP_transformed = mod.get_pars('plot')
    figCI = corner.corner(flat_samples_re, labels=imp_par_names,
                         truths=pars_MAP_transformed[ind_imp_pars])
    if cornerplot_make_allplot:
        figCA = corner.corner(flat_samples_transformed_plot, labels=par_names,
                              truths=pars_MAP_transformed)
else:
    figCI = corner.corner(flat_samples_re, labels=imp_par_names)
    if cornerplot_make_allplot:
        figCA = corner.corner(flat_samples_transformed_plot, labels=par_names)
print(dividerstr)
#--- Get a multivariate normal fit to the posterior
#     (overplotted on cornerplot, used for G&D Marginal likelihood)
#
# find bestfit multivariate normal to full distribution (untransformed)
print("Fitting a multivariate normal to samples to be used as the G&D function")
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
print("Overplotting GD function onto corner plots...")
# overplot just the important parameters on cornerplot
if (len(ind_imp_pars) == 1):
    iip = np.concatenate( (ind_imp_pars, [ind_imp_pars[0]-1]) )
    corner.corner(mvn_samples_transformed[:,iip],
                  fig=figCI, color='red',
                  plot_datapoints=False, plot_density=False,
                  hist_kwargs = {'alpha' : 0.0})
    # reset the yrange on the 1-D histograms
    ndim = len(flat_samples[0])
    axes = np.array(figCI.axes).reshape((2,2))
    for i in range(2):
        ax = axes[i, i]
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax/ml_fac_to_make_smooth)
else:
    corner.corner(mvn_samples_transformed[:,ind_imp_pars],
                  fig=figCI, color='red',
                  plot_datapoints=False, plot_density=False,
                  hist_kwargs = {'alpha' : 0.0})
    # reset the yrange on the 1-D histograms
    ndim = len(flat_samples[0])
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

#--- Calculate marginal likelihood and print to user
print(dividerstr)
print("Calculating marginal likelihood...")
a, b = calc_marg_like(flat_samples, mvn)
#--- Save figures
print(dividerstr)
print("Saving cornerplot figures to\n\t" + file_cornerplot_imp
      + "\n\t" + file_cornerplot_all)
figCA.savefig(file_cornerplot_all)
figCI.savefig(file_cornerplot_imp)
#==============================================================
#//////////////////////////////////////////////////////////////
