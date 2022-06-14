# Nonlinear Mixed Effects MCMC 
## A package for performing Bayesian inference on a Nonlinear Mixed Effects (NLME) model (aka, Bayesian Hierarchical Model) using MCMC

This code wraps the [emcee](https://emcee.readthedocs.io/en/stable/) python module, providing scripts for: building a Bayesian Hierarchical model, running the MCMC chains to convergence (and outputting frequent diagnostic information, plots, and data), plotting the samples from the posterior density, and calculating the marginal likelihood associated to a particular statistical "model hypothesis".

Sections below:

1. [Overview of files](#overview-of-files)

2. [Theoretical background on Bayesian Hierarchical Models](#theoretical-background-on-bayesian-hierarchical-models)

3. [How to create a new project](#how-to-create-a-new-project)

## Overview of files

The following three modules are static and provide the structure of the model, code for running the MCMC, and other related tasks:

* `mixed-effects_mcmc.py` --- Imports a user-provided "model hypothesis" module and then runs the MCMC to convergence, providing regular output (diagnostic information to user/logfile; most recent samples; plots of: the autocorrelation time, the marginal likelihood, recent chain dynamics).

* `module_mixed-effects_model.py` --- Module providing the structure and manipulation of the mixed effects (aka, Bayesian Hierarchical) model, including evaluation of the prior.

* `mixed-effects_plot-data.py` --- Helper script to be run on a posterior sample produced and output by the main MCMC script.  It plots a representation of that posterior distribution overtop the data set from which it was generated (optionally: 68% and 95% CI regions/lines, median value). Must import the same "model hypothesis" module that was used in the MCMC run producing the samples.

The following three template modules must be edited by the user when creating a new project to provide: the **data**, the **statistical model hypothesis** (including, e.g., choices of: which parameters are to be estimated; which parameters arise from a population distribution; the distribution-type and parameters of the the population distribution(s); which parameters have common values over all individuals; what is the error model within the likelihood function), and the **"evolve model"** for dynamical simulation of a theoretical model to be compared to the data via the likelihood function.

* `module_<projectname>_data.py` --- Loads the full project data set at the request of the modhyp module, (optionally) restricts to a sub-project for analysis, identifies the distinct individuals providing data for that sub-project, and specifies their respective data sets and associated evolve-models. For each sub-project, must also specify the format of figure output for the `mixed-ffects_plot-data.py` script.

* `module_<projectname>_modhyp_<subprojectname>_<modhypname>.py` --- Imported by the `mixed-effects_mcmc.py` main script and provides a public method for getting the current value of the log-posterior density. Thus, this module organizes the parameters, specifies the likelihood function and prior density, and the handles the data sets provided by the data module for a chosen "sub-project".  Specifically, the module:
    * Initializes the data module, identifying the "data analysis sub-project" to be analyzed.
    * Specifies the parameters to be tracked under Bayesian inference, separating them into three categories: INDIV_POP (individual-specific parameters assumed arise from a distribution over the population), INDIV_NOPOP (individual-specific parameters not controlled by a population distribution), POP (parameters specifying the population distribution for each parameter in INDIV_POP), OTHER (e.g., parameters associated to the error model, common parameters for all individuals).
    * Specifies the prior density for each parameter in INDIV_NOPOP, POP, and OTHER (the priors for each INDIV_POP parameter is a conditional prior with respect to the POP distribution and is calculated automatically --- this is the essence of a Bayesian Hierarchical model).
    * Specifies the likelihood function, in particular the error model to be used (which is potentially different for different data sets within the data analysis sub-project).
    
  For any data analysis subproject, a user will likely create multiple "modhyp" modules, each specifying a different statistical model hypothesis, which can be compared/ranked for parsimony by their marginal likelihood.

* `module_<projectname>_evolvemodel.py` --- Contains all dynamical models necessary for generating simulations to be compared to all data sets in the project (comparison occurs within the likelihood function for a particular sub-project + model-hypothesis choice). A single evolve-model might produce multiple "y-types" of output that can be compared to multiple different data sets. The data module provides the structure for --- and the model-hyp module executes --- evolving the necessary evolve-models for the data sets within a chosen data analysis sub-project.

## Theoretical background on Bayesian Hierarchical Models

## How to create a new project

1. Create a new directory called \<*projectname*\> within the `projects` directory.

2. Copy the static module files (first three listed in the previous section) into this new directory.

3. Copy the `data`, `modhyp`, and `evolvemodel` template modules into the new directory, replacing \<*projectname*\>, \<*subprojectname*\>, and \<*modhypname*\> with descriptive names for the project (the larger dataset), the data analysis subproject (the particular subset of the larger project dataset to be considered now), and the statistical model hypothesis (what choices are being made about the parameters to be inferred).  Later, different model hypotheses (and different data analysis sub-projects) can be considered by creating new versions of the `modhyp` module.

4. Create subdirectories within the new project directory: `plots`, `output`, `logfiles`, and `data`.  Place the raw project data file within `data`.

4. Edit the `data` module:
    * Specify the filename for import
    * 

5. Edit the `evolvemodel` module:
    * For every evolvemodel specified within the data module, create a method that takes independent values `X` and returns a dictionary of numpy arrays with keys the y_types.

