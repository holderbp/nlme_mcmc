# Nonlinear Mixed Effects MCMC 
## A package for performing Bayesian inference with a Nonlinear Mixed Effects (NLME) model (aka, Hierarchical Bayesian model) with MCMC

This code wraps the [emcee](https://emcee.readthedocs.io/en/stable/) python module, providing scripts for: building a Bayesian Hierarchical model, running MCMC to convergence (and giving informative running diagnostic plots and information), plotting the posterior density, and calculating the marginal likelihood associated to the chosen model hypothesis.

The following modules are static and provide the structure of the model and the running of the MCMC and other related tasks:

* `mixed-effects_mcmc.py` --- Imports a user-provided "model hypothesis" module and then runs the MCMC to convergence, providing regular output (diagnostic information to user/logfile; most recent samples; plots of: the autocorrelation time, the marginal likelihood, recent chain dynamics).

* `module_mixed-effects_model.py` --- Module providing the structure and manipulation of the mixed effects (aka, Bayesian Hierarchical) model, including evaluation of the prior.

* `mixed-effects_plot-data.py` --- Helper script to be run on sample file output by `mixed-effects_mcmc.py` which plots the posterior sample overtop the data set from which it was generated (optionally: 68% and 95% CI regions/lines, median value)

And the following three template scripts must be edited by the user to provide the data, the statistical model hypothesis (i.e., choices of: which parameters to be estimated, which parameters arise from a population distribution, which parameters have common values over all individuals, what is the error model within the likelihood function), and the "evolve model" for dynamical simulation of a theoretical model to be matched to data.

* `module_projectname_data.py` --- 

## Background on Bayesian Hierarchical / NLME

## Creating a new project

