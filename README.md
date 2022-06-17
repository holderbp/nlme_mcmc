# Nonlinear Mixed Effects MCMC 
## A package for performing Bayesian inference on a Nonlinear Mixed Effects (NLME) model (aka, Bayesian Hierarchical Model) using MCMC

This code wraps the [emcee](https://emcee.readthedocs.io/en/stable/) python module, providing scripts for: building a Bayesian Hierarchical model, running the MCMC chains to convergence (and outputting frequent diagnostic information, plots, and data), plotting the samples from the posterior density, and calculating the marginal likelihood associated to a particular statistical "model hypothesis".

Sections below:

1. [Theoretical background and implementation of Mixed-Effect Models](#theoretical-background-and-implementation-of-mixed-effect-models)

2. [Overview of files](#overview-of-files)

3. [How to create a new project](#how-to-create-a-new-project)

4. [Example projects](#example-projects)

## Theoretical background and implementation of Mixed-Effect Models

A mixed-effects statistical model (or, in Bayesian language, a Bayesian Hierarchical Model) is applied to datasets where the responses of multiple "individuals" have been measured.  Instead of assuming that all explanatory parameters for the responses of different individuals are independent, one assumes that some are random variables drawn from a (as yet unknown) distribution over the population of all individuals.  

Bayesian inference for a mixed-effects model follows the standard inference protocol (i.e., applying Bayes Theorem for probability densities), but two implementation assumptions are made:
* the prior density for a parameter that is assumed to be drawn from a population distribution is expressed in terms of its conditional prior with respect to that distribution (see below), and
* the likelihood function is assumed to be independent of the parameters that specify those population distributions.

To implement this, we allow for four types of parameters, stored in the following variables:

* `indiv_pop`: Individual-specific parameters that are assumed to be drawn from a distribution over the population of individuals. In a standard mixed-effects model, these are the "random effects" (although often only the deviation from the population mean is called a random effect). [Implemented as a pandas dataframe indexed by individuals, with columns labeled by parameter name.]
* `indiv_nopop`: Individual-specific parameters, but without any assumption of being drawn from a distribution. [Implemented as a pandas dataframe indexed by individuals, with columns labeled by parameter name.]
* `pop`: Parameters associated to specifying the population distribution(s). [Implemented as a dictionary, with parameter names as keys.]
* `other`: Other parameters, for example those that are assumed to have a common value for all individuals, or those specifying the error-model for the calculation of log-likelihood. [Implemented as a dictionary, with parameter names as keys.]

Bayes theorem for densities (for a particular data set, *d*, a particular chosen model hypothesis, *Hk*, and associated parameters, *theta*) is:

![f_{\vec{\Theta}} \left(\vec{\theta} \; \middle| \;  \left\{ \vec{D} = \vec{d} \right\} \cap \mathcal{H}_k \right)  = \frac{f_{\vec{D}} \left( \vec{d} \; \middle| \; \left\{ \vec{\Theta} = \vec{\theta} \right\} \cap \mathcal{H}_k \right) \, \cdot \,f_{\vec{\Theta}} \left( \vec{\theta} \; \middle| \; \mathcal{H}_k \right)}{f_{\vec{D}} \left( \vec{d} \; \middle| \; \mathcal{H}_k \right)}](/images/eqn_bayes-thm-densities.png)

which we can write in shorthand as:

![f\left(\vec{\theta} \; \middle| \;  \vec{d}, \mathcal{H} \right)  = \frac{f \left( \vec{d} \; \middle| \;  \vec{\theta},  \mathcal{H} \right) \, \cdot \,f \left( \vec{\theta} \; \middle| \; \mathcal{H} \right)}{f \left( \vec{d} \; \middle| \; \mathcal{H} \right)}](/images/eqn_bayes-thm-densities_simplified.png)

where the left-hand side is the *posterior density* of the parameters, the numerator on the right-hand side is the *likelihood function* of the parameters multiplied by the (joint) *prior density* of the parameters, and the denominator on the right-hand side is the *marginal likelihood* (of the model hypothesis).

For a Bayesian hierarchical model, we assume that the joint prior distribution of the `indiv_pop` and the `pop` parameters should be written in terms of a conditional density:

![ f\left({\rm indiv\_pop},  {\rm pop} \right) =  f\left({\rm indiv\_pop} \; \middle| \;  {\rm pop} \right) \, \cdot \,  f\left({\rm pop} \right)](/images/eqn_joint-prior.png)

in other words, our assumption that the `indiv_pop` parameters are drawn from a distribution over the population is encoded in the prior.  So, we must specify:

* a functional form for the conditional prior density, i.e., the population distribution's type (e.g., `indiv_pop`(s) are normally-distributed about some population mean), and

* a prior density for each of the `pop` parameters specifying that distribution.

Currently, the implentation considers the priors of the `other` and `indiv_nopop` parameters to be independent such that the full joint prior is written:

![f\left(\theta \; \middle| \; H \right) &= f\left({\rm indiv\_pop},  {\rm pop}, {\rm indiv\_nopop}, {\rm other} \right) \\
& =  f\left({\rm indiv\_pop},  {\rm pop}\right) \, \cdot \, f\left({\rm indiv\_nopop}, {\rm other} \right)](/images/eqn_joint-prior-full.png)

where the priors for the `other` and `indiv_nopop` parameters must be specified.

## Overview of files

The code is written in python3 and depends on the modules within the conda environment file stats.yml.  After installing `miniconda`, run:
```
> conda env create -f stats.yml
> conda activate stats
```
or, if any changes are made to the environment file, re-install with
```
> conda deactivate
> conda env update -f stats.yml --prune
> conda activate stats
```

The following three modules are static and provide the structure of the model, code for running the MCMC, and other related tasks:

* `mixed-effects_mcmc.py` --- Imports a user-provided "model hypothesis" module and then runs the MCMC to convergence, providing regular output (diagnostic information to user/logfile; most recent samples; plots of: the autocorrelation time, the marginal likelihood, recent chain dynamics).

* `module_mixed-effects_model.py` --- Module providing the structure and manipulation of the mixed effects (aka, Bayesian Hierarchical) model, including evaluation of the prior.

* `mixed-effects_plot-data.py` --- Helper script to be run on a posterior sample produced and output by the main MCMC script.  It plots a representation of that posterior distribution overtop the data set from which it was generated (optionally: 68% and 95% CI regions/lines, median value). Must import the same "model hypothesis" module that was used in the MCMC run producing the samples.

The following three template modules must be edited by the user when creating a new project to provide: the **data**, the **statistical model hypothesis** (including, e.g., choices of: which parameters are to be estimated; which parameters arise from a population distribution; the distribution-type and the parameters of the population distribution(s); which parameters have common values over all individuals; what is the error model within the likelihood function), and the **"evolve model"** for dynamical simulation of a theoretical model to be compared to the data via the likelihood function.

* `module_<projectname>_data.py` --- Loads the full project data set at the request of the modhyp module, (optionally) restricts to a sub-project for analysis, identifies the distinct individuals providing data for that sub-project, and specifies their respective data sets and associated evolve-models. For each sub-project, one must also specify the format of figure output for the `mixed-ffects_plot-data.py` script.

* `module_<projectname>_modhyp_<subprojectname>_<modhypname>.py` --- Imported by the `mixed-effects_mcmc.py` main script and provides a public method for getting the current value of the log-posterior density. Thus, this module organizes the parameters, specifies the likelihood function and prior density, and the handles the data sets provided by the data module for a chosen "sub-project".  Specifically, the module:
    * Initializes the data module, identifying the "data analysis sub-project" to be analyzed.
    * Specifies the parameters to be tracked under Bayesian inference, separating them into three categories: `indiv_pop` (individual-specific parameters assumed arise from a distribution over the population), `indiv_nopop` (individual-specific parameters not controlled by a population distribution), `pop` (parameters specifying the population distribution for each parameter in `indiv_pop`), `other` (e.g., parameters associated to the error model, common parameters for all individuals).
    * Specifies the prior density for each of the `indiv_nopop`, `pop`, and `other` parameters (the prior for each `indiv_pop` parameter is a conditional prior with respect to the `pop` distribution and is calculated automatically --- this is the essence of a Bayesian Hierarchical model; see below in the Theoretical/Implementation section).
    * Specifies the likelihood function, in particular the error model to be used (which is potentially different for different data sets within the data analysis sub-project).
    
  For any data analysis subproject, a user will likely create multiple "modhyp" modules, each specifying a different statistical model hypothesis, which can be compared/ranked for parsimony by their marginal likelihood.

* `module_<projectname>_evolvemodel.py` --- Contains all dynamical models necessary for generating simulations to be compared to all data sets in the project (comparison occurs within the likelihood function for a particular sub-project + model-hypothesis choice). A single evolve-model might produce multiple "y-types" of output that can be compared to multiple different data sets. The data module provides the structure for --- and the model-hyp module executes --- evolving the necessary evolve-models for the data sets within a chosen data analysis sub-project.

## How to create a new project

1. Create a new directory called \<*projectname*\> within the `projects` directory.

2. Copy the static module files (the first three files listed in the previous section) into this new directory.

3. Copy the `data`, `modhyp`, and `evolvemodel` template modules into the new directory, replacing \<*projectname*\>, \<*subprojectname*\>, and \<*modhypname*\> with descriptive names for the project (the larger dataset), the data analysis subproject (the particular subset of the larger project dataset to be considered now), and the statistical model hypothesis (what choices are being made about the parameters to be inferred).  Later, different model hypotheses (and different data analysis sub-projects) can be considered by creating new versions of the `modhyp` module.

4. Create subdirectories within the new project directory: `plots`, `output`, `logfiles`, and `data`.  Place the raw project data file within `data`.

5. Edit the data module, `module_<projectname>_data.py`:

    * Within `Parameters`, set the `project_name` to \<*projectname*\> and specify the filename for import in `data_file`.
    
    * Within `Data Handling`, specify:

        - `data_sets`: the list of data-sets (names to be specified in the next step) that are simulated by each evolve-model (names to be specified in the evolve-model module), and
        - `evolve_model`: the inverse mapping to `data_sets`, i.e., the evolve-model whose simulation is to be compared to a particular data-set.

      These mappings allow for the data to be compared to the model within the likelihood function, for a very general collection of data sets and models.
      
    * Within the `prep_data()` method, manage the import of the raw data file into a dataframe of standard form, and define multiple `data_analysis_subproject`(s):

        - Adjust the import of the raw data file and create a dataframe with these columns: `['indiv', 'data_set', 'x_data_type', 'X', 'y_data_type', 'Y']`.

        - Assign descriptive unique names to each individual in the `indiv` column.

        - Assign descriptive unique names for each distinct data set in the `'data_set'` column.

        - Perform any necessary cleaning of the dataframe.

        - Ensure that there is a single data point per row (e.g., if multiple replicates are given per row in the raw file, flatten them into their own rows).

        - Specify which data sets are to be selected for each `data_analysis_subproject`.

        - Sort the dataframe such that: the names of the individuals have a known order (the ordered array created by `np.unique(df['indiv'])` will be used in the model hypothesis module for setting initial guess values in the `indiv_pop` and `indiv_nopop` dataframes), and the `'X'` values for each data set are ordered.

    * Within the `plot_data()` method, establish a multi-page figure structure for each `data_analysis_subproject` choice.

        - Decide the logic of plotting (how many subplots on each page, which individual's data is plotted where) and create separate figures for each page (`fig1`, `fig2`, etc).

        - Specify the plotting of the data sets into the subplots of these figures.

        - Fill in the dictionary `theaxis` to specify which `[individual][evolve-model][data-set]` is assigned to which axis.  This will allow the associated evolve-model outputs to be automatically plotted into the correct subplots.

6. Edit the model-hypothesis module, `module_<projectname>_modhyp_<subprojectname>_<modhypname>.py`:

    * Specify the data module for import (`module_<projectname>_data`).

    * Specify the `data_analysis_subproject`, which must match one created within the `data` module.
    
    * Create a short descriptive string for this statistical model hypothesis in `model_hyp`.  This will be used for naming output files.

    * Within the `INDIV_POP` and `INDIV_NOPOP` sections:

        - Specify an ordered list of parameters in `mm.list_indiv_x_pars` (`x` = `pop` or `nopop`), which will be the column names for the dataframe.

        - Give a descriptive name for each parameter as an entry in the `mm.descrip_name` dictionary.

        - Set the initial guess values of the parameters in the `indiv_x` dataframe, using the ordered list of individual names (specified in the data module) as an index.

        - If there are no parameters of this type, set `indiv_x` to an empty dataframe.

    * Within the `POP` section:

        - For each parameter `<par>` in `mm.list_indiv_pop_pars`:

            * Set the distribution type in the `mm.pop_dist` dictionary.

            * Set initial guess values for the distribution parameters in the `pop` dictionary, using the keys `mm.pop['<par>_mean']` and `mm.pop['<par>_stdev']` (normal/lognormal distribution).

            * Specify the prior information for both `'<par>_mean'` and `'<par>_stdev'`, using the dictionaries: `mm.prior_dist`; `mm.prior_mean` and `mm.prior_stdev` (normal priors); `mm.prior_min` and `mm.prior_max` (uniform prior).

            * Specify the "transformations" of both `'<par>_mean'` and `'<par>_stdev'`, using the dictionaries: `mm.tranf_mcmc` (transformation of value while conducting mcmc, e.g., `'log'` if the parameter should be strictly positive); `mm.transf_plot` (when making cornerplots of the posterior distribution); `mm.transf_print` (when printing the values for user/logfile).

    * Within the `OTHER` section:

        - Specify an ordered list of parameters in `mm.list_other_pars`.

        - Give a descriptive name for each parameter as an entry in the `mm.descrip_name` dictionary.

        - Set the initial guess value as an entry in the `mm.other` dictionary.

        - Specify the prior and transformation information (as in `POP`, above) for each parameter.

    * Within the `IMPORTANT PARAMETERS` section, set `important_pars` to the list of names of non-nuisance and non-individual parameters to be tracked (usually the `POP` distribution parameters and the `OTHER` parameters).  These are the parameters that will be shown, e.g., in cornerplots.

    * Within the `get_log_likelihood()` method, it should only be necessary to alter the error-model specification (e.g., whether the data variance is individual-specific, and thus one of the `INDIV_POP` or `INDIV_NOPOP` parameters, or whether it is common to all data sets, or, more generally, which non-normal distribution should be used).

7. Edit the evolution-model module, `module_<projectname>_evolvemodel.py`:

    * Within the `run_evolve_model(...)` method, assign an evolve-model run for each evolve-model name that has been identified in the `data` module (see the `Data Handling` section of Step 5). Create a method that takes independent values `X` and returns a dictionary of numpy arrays with keys the y_types.
    
    * Add additional methods, or module imports (e.g., a module that brings in a system of ODEs), as needed to achieve these model evolutions.

8. Import the model-hypothesis module within `mixed-effects_mcmc.py` and then run it with
```
> conda activate stats           # if not already done
stats > python mixed-effects_mcmc.py
```
or unattended
```
> conda activate stats           # if not already done
stats > at now
./runit <descriptive-string-about-run>
[Ctrl-d]
```

9. After analysis of a particular `data_analysis_subproject` with a particular `model_hyp`, create new model-hypothesis modules to test other model hypotheses, or move on to other subprojects.

## Example projects

1. Sleepstudy

2. HIV-Tcell