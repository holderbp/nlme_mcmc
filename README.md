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

In a mixed-effects model (or, in Bayesian language, a Bayesian Hierarchical Model), we assume four types of parameters:

* INDIV_POP: individual-specific, and assumed to be drawn from a distribution over the population of individuals. These are sometimes called "random effects" but we will call them "individual-population" parameters.
* INDIV_NOPOP: individual-specific, but without any assumption of being drawn from a distribution.
* POP: parameters associated to specifying the population distribution itself.
* OTHER: for instance parameters that are assumed to have a common value for all individuals, or those specifying the error-model for the calculation of log-likelihood.

Bayes theorem for densities (for a particular data set *d*, a chosen model hypothesis, *H*, and associated parameters, *theta*) is:
![f_{\vec{\Theta}} \left(\vec{\theta} \; \middle| \;  \left\{ \vec{D} = \vec{d} \right\} \cap \mathcal{H}_k \right)  = \frac{f_{\vec{D}} \left( \vec{d} \; \middle| \; \left\{ \vec{\Theta} = \vec{\theta} \right\} \cap \mathcal{H}_k \right) \, \cdot \,f_{\vec{\Theta}} \left( \vec{\theta} \; \middle| \; \mathcal{H}_k \right)}{f_{\vec{D}} \left( \vec{d} \; \middle| \; \mathcal{H}_k \right)}](/images/eqn_bayes-thm-densities.png)

   where the lhs is the posterior density of the parameters,
   the numerator on the rhs is the "likelihood" of the parameters
   multiplied by the prior density of the parameters, and the
   denominator on the rhs is called the "marginal likelihood"
   (of the model hypothesis).

   For a Bayesian hierarchical model, we assume that the
   joint prior distribution of the individual-pop and the
   population-distribution parameters should be written in terms
   of a conditional density

      f(I_pop, Pop) = f(I_pop | Pop) * f(Pop)

   so we must specify:

      * a distribution for the conditional density, e.g.,
        I_pop is normally-distributed over the population.

      * a prior distribution for each of the Pop pars
        specifying that distribution

   Generally we consider the priors of the "Other" and the
   "I_nopop" parameters are independent, such that the full
   joint prior is written:

    f(Other, I_nopop, I_pop, Pop) =

               f(Other) * f(I_nopop) * f(I_pop, Pop)

   where the priors for Other and I_nopop pars must also be
   specified.

   This code, "mcmc_mixed-effects.py", is a wrapper for the
   emcee module, and runs MCMC chains until convergence is
   achieved.  Along the way, it provides the user with relevant
   convergence statistics, and it outputs regularly the current
   best samples, posterior estimates, and marginal likelihood
   estimates.

   This main code assumes the existence of a data "project"
   module, e.g., "data_hiv-tcell.py", which can provide data
   sets for a multiple possible "data analysis" sub-projects
   (e.g., for data_hiv-tcell.py, we can look at just virus-decay,
   just timeseries, timeseries+virus-dilution, all-exper, ...).
   The data module is loaded by the model hypothesis module
   (see below) and provides the log-likelihood function.

   For each "data analysis" sub-project, there can be multiple
   "model hypotheses", one of which is imported here as a module.
   The model hypothesis module provides the log-posterior for a
   given data-project/data-analysis/model-hypothesis choice. It
   also provides the individual log-prior and log-likelihood values
   that go into that sum, which are saved as blobs by emcee.

   The comparison of different model hypotheses for a given data
   analysis sub-project is done by ranking the marginal
   likelihoods (ML) of each (or finding the Bayes factor comparing
   two hypotheses).  Marginal likelihood, as estimated by MCMC
   chains, is a very noisy quantity (even when using the Gelfand
   & Dey method for "correcting" the "harmonic mean of the
   likelihood" estimator) and should be estimated using (many
   steps in) the tail of a long converged MCMC run (the data
   and a plot for the ML are output regularly).


  A model hypothesis module contains the specific method of
  calculating the log-posterior, which is the sum of:

     * conditional log-prior of the I_pop pars with respect
       to the current population distribution
     * log-prior for the Pop (pop dist) parameters
     * log-prior for I_nopop parameters
     * log-prior of any Other parameters

  and

     * the log-likelihood of the data given these parameters

  It gets the log-likelihood calculation from the data module,
  which depends on everything but the Pop parameters.

  Model hypothesis modules are associated to a "data analysis
  subproject" of a "data project", and should be named
  accordingly, e.g.,

     modhyp_<project>_<subproject>_<model-hypothesis>


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

