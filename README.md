# Nonlinear Mixed Effects MCMC 
## A package for performing Bayesian inference on a Nonlinear Mixed Effects (NLME) model (aka, Hierarchical Bayesian model) using MCMC

This code wraps the [emcee](https://emcee.readthedocs.io/en/stable/) python module, providing scripts for: building a Bayesian Hierarchical model, running MCMC to convergence (and giving informative running diagnostic plots and information), plotting the posterior density, and calculating the marginal likelihood associated to the chosen model hypothesis.

Sections below:

1. [Overview of files](#overview-of-files)

2. [Theoretical background on Bayesian Hierarchical Models](#theoretical-background-on-bayesian-hierarchical-models)

3. [How to create a new project](#how-to-create-a-new-project)

## Overview of files

The following three modules are static and provide the structure of the model, code for running the MCMC, and other related tasks:

* `mixed-effects_mcmc.py` --- Imports a user-provided "model hypothesis" module and then runs the MCMC to convergence, providing regular output (diagnostic information to user/logfile; most recent samples; plots of: the autocorrelation time, the marginal likelihood, recent chain dynamics).

* `module_mixed-effects_model.py` --- Module providing the structure and manipulation of the mixed effects (aka, Bayesian Hierarchical) model, including evaluation of the prior.

* `mixed-effects_plot-data.py` --- Helper script to be run on a posterior sample produced and output by the main MCMC script.  It plots a representation of that posterior distribution overtop the data set from which it was generated (optionally: 68% and 95% CI regions/lines, median value). Must import the same "model hypothesis" module that was used in the MCMC run producing the samples.

The following three template modules must be edited by the user when creating a new project, to provide the data, the statistical model hypothesis (including, e.g., choices of: which parameters are to be estimated; which parameters arise from a population distribution; the distribution-type and parameters of the the population distribution(s); which parameters have common values over all individuals; what is the error model within the likelihood function), and the "evolve model" for dynamical simulation of a theoretical model to be compared to the data via the likelihood function.

* `module_projectname_data.py` --- Loads the full project data set at the request of the modhyp module, (optionally) restricts to a sub-project for analysis, identifies the distinct individuals providing data for that sub-project, and specifies their respective data sets and associated evolve-models. For each sub-project, must also specify the format of figure output for the `mixed-ffects_plot-data.py` script.

* `module_projectname_modhyp_subprojectname_modhypname.py` --- Imported by the `mixed-effects_mcmc.py` main script and provides a public method for getting the current value of the log-posterior density. Thus, this module organizes the parameters, specifies the likelihood function and prior density, and the handles the data sets provided by the data module for a chosen "sub-project".  Specifically, the module:
    * Initializes the data module, identifying the "data analysis sub-project" to be analyzed.
    * Specifies the parameters to be tracked under Bayesian inference, separating them into three categories: INDIV_POP (individual-specific parameters assumed arise from a distribution over the population), INDIV_NOPOP (individual-specific parameters not controlled by a population distribution), POP (parameters specifying the population distribution for each parameter in INDIV_POP), OTHER (e.g., parameters associated to the error model, common parameters for all individuals).
    * Specifies the prior density for each parameter in INDIV_NOPOP, POP, and OTHER (the priors for each INDIV_POP parameter is a conditional prior with respect to the POP distribution and is calculated automatically --- this is the essence of a BAyesian Hierarchical model).
    * Specifies the likelihood function, in particular the error model to be used (which is potentially different for different data sets within the data analysis sub-project).
    
  For any data analysis subproject, there will likely be multiple "modhyp" modules, each specifying a different statistical model hypothesis, which can be compared/ranked for parsimony by their marginal likelihood.

* Another item

## Theoretical background on Bayesian Hierarchical Models

## How to create a new project

