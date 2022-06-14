# Nonlinear Mixed Effects MCMC 
## A package for performing Bayesian inference on a Nonlinear Mixed Effects (NLME) model (aka, Hierarchical Bayesian model) using MCMC

This code wraps the [emcee](https://emcee.readthedocs.io/en/stable/) python module, providing scripts for: building a Bayesian Hierarchical model, running MCMC to convergence (and giving informative running diagnostic plots and information), plotting the posterior density, and calculating the marginal likelihood associated to the chosen model hypothesis.

The following modules are static and provide the structure of the model, the running of the MCMC, and other related tasks:

* `mixed-effects_mcmc.py` --- Imports a user-provided "model hypothesis" module and then runs the MCMC to convergence, providing regular output (diagnostic information to user/logfile; most recent samples; plots of: the autocorrelation time, the marginal likelihood, recent chain dynamics).

* `module_mixed-effects_model.py` --- Module providing the structure and manipulation of the mixed effects (aka, Bayesian Hierarchical) model, including evaluation of the prior.

* `mixed-effects_plot-data.py` --- Helper script (to be run on sample file output by `mixed-effects_mcmc.py`) that plots a representation of the posterior distribution overtop the data set from which it was generated (optionally: 68% and 95% CI regions/lines, median value)

And the following three template scripts must be edited by the user to provide the data, the statistical model hypothesis (including, e.g., choices of: which parameters are to be estimated, which parameters arise from a population distribution, distribution choice and parameters of the population distribution, which parameters have common values over all individuals, what is the error model within the likelihood function), and the "evolve model" for dynamical simulation of a theoretical model to be compared to the data via the likelihood function.

* `module_projectname_data.py` --- Loads the full project data set, (optionally) restricts to a sub-project for analysis, identifies the distinct individuals providing data for that sub-project and specifies their respective data sets and associated evolve-models.

* `module_projectname_modhyp_specification.py` --- Provides the `mixed-effects_mcmc.py` script with the current value of the log-posterior density by keeping track of the parameters, the likelihood function and prior density, and the data sets within the data module.  Specifically, it:
    * Initializes the data module, identifying the "data analysis sub-project" to be analyzed.
    * Specifies the parameters to be tracked under Bayesian inference, separating them into three categories: INDIV_POP (individual-specific parameters assumed arise from a distribution over the population), INDIV_NOPOP (individual-specific parameters not controlled by a population distribution), POP (parameters specifying the population distribution for each parameter in INDIV_POP), OTHER (e.g., parameters associated to the error model, common parameters for all individuals).
    * Specifies the prior density for each parameter in INDIV_NOPOP, POP, and OTHER (the priors for each INDIV_POP parameter is a conditional prior with respect to the POP distribution and is calculated automatically --- this is the essence of a BAyesian Hierarchical model).
    * Specifies the likelihood function, in particular the error model to be used (which is potentially different for different data sets within the data analysis sub-project).
  For any data analysis subproject, there will likely be multiple "modhyp" modules, each specifying a different statistical model hypothesis, which can be compared/ranked for parsimony by their marginal likelihood.

* Another item

## Background on Bayesian Hierarchical / NLME

## Creating a new project

