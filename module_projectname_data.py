import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.backends.backend_pdf as mplbp
import matplotlib.pyplot as plt
import seaborn as sns
# make default style for seaborn
sns.set()

#=======================================================
#                   Parameters
#=======================================================
#--- Name of this data project
project_name = "sleepstudy"
#--- Model hypothesis pars (to be set by mod-hyp module)
data_analysis_subproject = None
model_hyp = None
#--- File containing all data for project
data_file = "data/sleepstudy.csv"
#--- Data parameters
#    (to be set when choosing data_analysis_subproject)
Nindiv = None
indiv_names = None
#--- Evolve-model and Error-model pars
#    (imported from model_hyp module at each log-like calc)
indiv_pop = None     # dataframe (rows are individuals)
indiv_nopop = None   # dataframe (rows are individuals)
other = None         # dictionary

#=======================================================
#                 Data Handling
#=======================================================
#--- map: evolve-model to (list of) data-set(s)
data_sets = {
    'linear' : ['reaction-ms_vs_deprivation-d']
}
#--- map: data-set to evolve-model
evolve_model = {
    'reaction-ms_vs_deprivation-d' : 'linear'
}
# 
#--- The data for each individual
#
#  emfi = { indiv1: [em1, em2, ...]
#           inviv2: [em1, ...]
#         }
#
#  dsfi = { indiv1: {em1: [dsA, dsB, ...],
#                    em2: [dsAA, dsC], ...}, ...}
#           indiv2: {...}
#         }
#
#  xvfi = { indiv1: { em1: { dsA: [xA1, xA2, ...],
#                            dsB: [xB1, xB2, ...], ...
#                          },
#                     em2: { dsAA: [xAA1, xAA2, ...],
#                            dsC: [xC1, xC2, ...], ...},
#                          ....
#                   },
#           indiv2: {...}
#            ... }
#
#  yvfi = { indiv1: { em1: { dsA: [yA1, yA2, ...],
#                            dsB: [yB1, yB2, ...], ...
#                          },
#                     em2: { dsAA: [yAA1, yAA2, ...],
#                            dsC: [yC1, yC2, ...], ...},
#                          ....
#                   },
#           indiv2: {...}
#            ... }
#
#  yvtfi = { indiv1: { em1: {dsA: 'ytypeA',
#                            dsB: 'ytypeB', ...
#                           },
#                      em2: { dsAA: 'ytypeAA',
#                             dsC:  'ytypeC', ... },
#                       ...
#                    }
#           indiv2: {...}
#            ... }
#
evolve_models_for_indiv = None
datasets_for_indiv = None
xvals_for_indiv = None
yvals_for_indiv = None        
yvals_type_for_indiv = None
# The set of unique x values needed for each evolve-model run
evolve_models_xvals_for_indiv = None
# And, conversely, the indicies of the above xval vectors for each dataset
xindices_for_indiv = None    

def prep_data():
    """
    Put the project data file into the common form for a 
    Bayesian hierarchical / NLME analysis

    This function is specific to the particular project, so that
    the other functions can be independent of project.
    """
    #
    #=== sleepstudy data set has
    #
    #  columns = [time_ms, deprivation_d, subject]
    #
    df = pd.read_csv(data_file)
    #--- Ensure that individual label is a string
    df['subject'] = df['subject'].apply(str)
    #--- Flatten any replicates into single 'Y' column
    #     (sleepstudy individuals have only one replicate)
    #
    #--- Rename columns to standard names
    df.columns = ['Y', 'X', 'indiv']
    #--- Construct labels for all distinct data sets
    #      (sleepstudy is all same data-set)
    df['data_set'] = 'reaction-ms_vs_deprivation-d'
    #--- Add info and re-order to get these columns:
    #
    #    [indiv, data_set, x_data_type, X, y_data_type, Y]
    #
    df['x_data_type'] = 'deprivation_days'
    df['y_data_type'] = 'reaction-time_ms'
    df = df[['indiv', 'data_set', 'x_data_type', 'X',
             'y_data_type', 'Y']]
    #--- Clean up the data
    #
    #      for sleepstudy:
    #       * Get rid of days 0-1 (they are just baseline values)
    #       * Re-number.  Remaining are 0-7 days of sleep deprivation
    #
    df = df[df['X'] > 1]
    df['X'] = df['X'] - 2
    #--- Select out only the data sets for the particular data_analysis_name
    if (data_analysis_subproject == 'response-time-vs-days-deprivation'):
        # only one data set for the sleepstudy project
        df = df[(df['data_set'] == 'reaction-ms_vs_deprivation-d')]
    #--- Sort dataframe by:
    #
    #      [indiv, data_set, 'X']
    #
    df = df.sort_values(['indiv', 'data_set', 'X'])
    #--- Get the distinct individuals for these data sets
    indiv_names = np.unique(df['indiv'].to_numpy())
    Nindiv = len(indiv_names)
    #--- return info on the individuals and the dataframe
    return Nindiv, indiv_names, df

def load_data(mhyp, da_subproj):
    """
    Public method used by model hypothesis module to load the
    requested data sets, arrange the data, return individual information
    """
    global model_hyp, data_analysis_subproject, Nindiv, indiv_names, \
        evolve_models_for_indiv, datasets_for_indiv, \
        xvals_for_indiv, yvals_for_indiv, yvals_type_for_indiv, \
        evolve_models_xvals_for_indiv, xindices_for_indiv
    #
    #--- Set the model hypothesis and data analysis type
    #
    model_hyp = mhyp
    data_analysis_subproject = da_subproj
    #
    #--- Load data set, extract only that data for this
    #    data_analysis_subproject, and put it in the standard format
    #    with:
    #
    #    columns = [indiv, data_set, x_data_type, X, y_data_type, Y]
    #
    #    where all Y-replicates have been flattenend (so there can be
    #    multiple X values for same data_set), and it is sorted by
    #
    #              [indiv, data_set, X]
    #
    Nindiv, indiv_names, df = prep_data()
    #
    #--- Place the data for each dictionaries, organized by:
    #
    #        -> individual
    #           -> evolution-model
    #              -> data-set
    #
    #    So that each evolution-model "child" of an individual will be
    #    run and then the model-values associated to each data-set will
    #    be extracted from the soln array of that evolution-model.
    #
    evolve_models_for_indiv = {}       # evolve_models list for individuals in dict [indiv]
    datasets_for_indiv = {}            # list of datasets in dict [indiv][ev-model]
    xvals_for_indiv = {}               # xval-arr in dict [indiv][ev-model][dataset]
    yvals_for_indiv = {}               # yval-arr in dict [indiv][ev-model][dataset]
    yvals_type_for_indiv = {}          # yval-type-arr in dict [indiv][ev-model][dataset]
    evolve_models_xvals_for_indiv = {} # necessary distinct xval-arr in dict [indiv][ev-model]
    xindices_for_indiv = {}            # indices of xvals in dict [indiv][ev-model][dataset]
    for i in indiv_names:
        datasets_for_indiv[i] = {}
        xvals_for_indiv[i] = {}
        yvals_for_indiv[i] = {}
        yvals_type_for_indiv[i] = {}
        ddf = df[(df['indiv'] == i)]
        # get list of data sets for one individual
        datasets = np.unique(ddf['data_set'].to_numpy())
        # get associated necessary evolve-models for those data sets
        evolve_models_for_indiv[i] = \
            np.unique([evolve_model[d] for d in datasets])
        # arrange data sets (x, y, and y-type) into dictionaries 
        for em in evolve_models_for_indiv[i]:
            datasets_for_indiv[i][em] = []
            xvals_for_indiv[i][em] = {}
            yvals_for_indiv[i][em] = {}
            yvals_type_for_indiv[i][em] = {}
        for d in datasets:
            em = evolve_model[d]
            datasets_for_indiv[i][em].append(d)
            dddf = ddf[(ddf['data_set'] == d)]
            xvals = dddf['X'].to_numpy()
            yvals = dddf['Y'].to_numpy()
            yval_type = dddf['y_data_type'].to_numpy()[0]
            xvals_for_indiv[i][em][d] = xvals
            yvals_for_indiv[i][em][d] = yvals
            yvals_type_for_indiv[i][em][d] = yval_type
        # get the set of distinct x-values needed for each evolve-model run
        evolve_models_xvals_for_indiv[i] = {}
        dx = {}
        for em in evolve_models_for_indiv[i]:
            xvals = []
            for d in datasets_for_indiv[i][em]:
                xvals = np.concatenate( (xvals, xvals_for_indiv[i][em][d]) )
            unique_xvals = np.sort(np.unique(xvals))
            evolve_models_xvals_for_indiv[i][em] = unique_xvals
            # make an index dictionary for the index vectors below
            x_length = len(evolve_models_xvals_for_indiv[i][em])
            dx[em] = dict(zip(evolve_models_xvals_for_indiv[i][em],
                              np.arange(x_length)))
        # then get the indices of the unique_xvals array that are
        # necessary for each dataset
        xindices_for_indiv[i] = {}
        for em in evolve_models_for_indiv[i]:
            xindices_for_indiv[i][em] = {}
            for d in datasets_for_indiv[i][em]:
                idx = [ dx[em][x] for x in xvals_for_indiv[i][em][d] ]
                xindices_for_indiv[i][em][d] = np.array(idx)
    #
    #--- Return info on individuals to the model-hyp module
    #
    return [Nindiv, indiv_names, yvals_for_indiv, yvals_type_for_indiv,
            evolve_models_for_indiv, evolve_models_xvals_for_indiv,
            datasets_for_indiv, xindices_for_indiv]

def plot_data(ppars):
    """
    Public method called by the model hypothesis module to plot
    data sets for a particular data analysis subproject.

    The data are plotte and then a representation of the posterior
    distribution overplotted (spaghetti; lines to indicate confidence
    intervals; or, probably best: shaded regions of 68 and 
    95% confidence).

    For each data_analysis_subproject, this script should be to create
    the relevant plots
    """
    #================================================================    
    #    Establish different plot for each data_analysis_subproject
    #
    #    Let's create a multi-page pdf. To do so, we'll make three
    #    figures with 3 rows and 2 columns each, so six individuals
    #    on each page.
    #
    #    First, plot the data into these axes.
    #
    #    Then take the evolve-model functions for 68-95% CI,
    #    passed from the model-hypothesis module in ppars,  to
    #    overplot the representation of the posterior samples.
    #
    #    Finally use PdfPages to save the multiple figures to a
    #    multi-page pdf
    #
    #    At the time of setting up data module the user can
    #    decide how to display the data sets used for each
    #    data_analysis_subproject.
    #
    #================================================================    
    if (data_analysis_subproject == 'response-time-vs-days-deprivation'):
        #
        #=======================================================
        #            Plot the data values first
        #=======================================================        
        #
        # First create figures for each page
        #
        # Page 1  (label each figure object with a "num")
        #
        nrows, ncols = 3, 2
        fig1, axes1 = plt.subplots(nrows, ncols, figsize=(8, 10.5), num='pg1')
        # add a hidden axis and give it the common x and y labels
        ax1 = fig1.add_subplot(111, frameon=False)
        ax1.grid(False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Sleep Deprivation (d)")
        plt.ylabel("Reaction Time (ms)")
        # a title on page 1
        fig1.suptitle("Sleepstudy data with mod_hyp = " + model_hyp)
        #
        # Page 2
        #
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(8, 10.5), num='pg2')
        # add a hidden axis and give it the common x and y labels
        ax2 = fig2.add_subplot(111, frameon=False)
        ax2.grid(False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Sleep Deprivation (d)")
        plt.ylabel("Reaction Time (ms)")
        #
        # Page 3
        #
        fig3, axes3 = plt.subplots(nrows, ncols, figsize=(8, 10.5), num='pg3')
        # add a hidden axis and give it the common x and y labels
        ax3 = fig3.add_subplot(111, frameon=False)
        ax3.grid(False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Sleep Deprivation (d)")
        plt.ylabel("Reaction Time (ms)")
        #
        #--- Plot each individual's data in their respective subplot axes
        #
        # There's only one evolve-model and dataset here, and it's the
        # same for all individuals.
        #
        em = evolve_models_for_indiv[indiv_names[0]][0]
        ds = datasets_for_indiv[indiv_names[0]][em][0]
        # activate first page figure and loop over first six individuals
        plt.figure('pg1')
        for i in np.arange(0,6):
            x = xvals_for_indiv[indiv_names[i]][em][ds]
            y = yvals_for_indiv[indiv_names[i]][em][ds]
            r = i // ncols
            c = i % ncols
            axes1[r,c].set_ylim(200,500)
            sns.scatterplot(ax=axes1[r,c], zorder=10, x=x, y=y, color='black')
        # activate second page figure 
        plt.figure('pg2')
        for i in np.arange(6,12):
            ii = i % 6
            x = xvals_for_indiv[indiv_names[i]][em][ds]
            y = yvals_for_indiv[indiv_names[i]][em][ds]
            r = ii // ncols
            c = ii % ncols
            axes2[r,c].set_ylim(200,500)
            sns.scatterplot(ax=axes2[r,c], zorder=10, x=x, y=y, color='black')
        # activate third page figure             
        plt.figure('pg3')
        for i in np.arange(12,18):
            ii = i % 6
            x = xvals_for_indiv[indiv_names[i]][em][ds]
            y = yvals_for_indiv[indiv_names[i]][em][ds]
            r = ii // ncols
            c = ii % ncols
            axes3[r,c].set_ylim(200,500)
            sns.scatterplot(ax=axes3[r,c], zorder=10, x=x, y=y, color='black')
        # exit if just plotting data
        if (ppars['plot_type'] == 'only_data'):
            # just save the plot
            pass
        else:
            #================================================================
            #               Assign axes to each [i][em][ds]
            #================================================================
            # for sleepmodel there is only one evolve-model
            # and one associated dataset... same for each indiv
            em = evolve_models_for_indiv[indiv_names[0]][0]
            ds = datasets_for_indiv[indiv_names[0]][em][0]
            theaxis = {}
            for i in range(len(indiv_names)):
                n = indiv_names[i]
                theaxis[n] = {}
                theaxis[n][em] = {}
                ii = i % 6
                r = ii // ncols
                c = ii % ncols
                if (i < 6):
                    theaxis[n][em][ds] = axes1[r,c]
                elif (i < 12):
                    theaxis[n][em][ds] = axes2[r,c]
                else:
                    theaxis[n][em][ds] = axes3[r,c]                    
            #================================================================
            #   Plot the posterior distribution of the model over the data
            #================================================================
            if (ppars['plot_type'] == 'median'):
                # plot just the median of the samples
                for i in indiv_names:
                    for em in evolve_models_for_indiv[i]:
                        for ds in datasets_for_indiv[i][em]:
                            xx = ppars['xmodels'][i][em]
                            y = ppars['CIbounds'][i][em][ds][2,:]
                            sns.lineplot(ax=theaxis[i][em][ds], zorder=1, x = xx,
                                         y = y, color='black', linestyle='--')
            elif (ppars['plot_type'] == 'spaghetti'):
                # for each dataset, plot all model evaluations
                for i in indiv_names:
                    for em in evolve_models_for_indiv[i]:
                        for ds in datasets_for_indiv[i][em]:
                            xx = ppars['xmodels'][i][em]
                            for s in range(len(ppars['ymodels'][i][em][ds])):
                                yy = ppars['ymodels'][i][em][ds][s]
                                sns.lineplot(ax=theaxis[i][em][ds], zorder=1, x = xx,
                                             y = yy, alpha = 0.1, color='red')
                            if (ppars['plot_median']):
                                sns.lineplot(ax=theaxis[i][em][ds], zorder=1, x = xx,
                                             y = ppars['CIbounds'][i][em][ds][2,:],
                                             color='black', linestyle='--')                                             
            elif (ppars['plot_type'] == '68CI_line'):
                # plot just the bounds of the 68% confidence interval
                for i in indiv_names:
                    for em in evolve_models_for_indiv[i]:
                        for ds in datasets_for_indiv[i][em]:
                            xx = ppars['xmodels'][i][em]
                            ylow = ppars['CIbounds'][i][em][ds][1,:]
                            yhigh = ppars['CIbounds'][i][em][ds][3,:]                            
                            sns.lineplot(ax=theaxis[i][em][ds], zorder=1,
                                         x = xx , y = ylow, color=ppars['color_dark'])
                            sns.lineplot(ax=theaxis[i][em][ds], zorder=1,
                                         x = xx, y = yhigh, color=ppars['color_dark'])
                            if (ppars['plot_median']):
                                sns.lineplot(ax=theaxis[i][em][ds], zorder=1, x = xx,
                                             y = ppars['CIbounds'][i][em][ds][2,:],
                                             color='black', linestyle='--')                                             
            elif (ppars['plot_type'] == '95CI_line'):
                # plot just the bounds of the 95% confidence interval
                for i in indiv_names:
                    for em in evolve_models_for_indiv[i]:
                        for ds in datasets_for_indiv[i][em]:
                            xx = ppars['xmodels'][i][em]
                            ylow = ppars['CIbounds'][i][em][ds][0,:]
                            yhigh = ppars['CIbounds'][i][em][ds][4,:]                            
                            sns.lineplot(ax=theaxis[i][em][ds], zorder=1,
                                         x = xx , y = ylow, color=ppars['color_dark'])
                            sns.lineplot(ax=theaxis[i][em][ds], zorder=1,
                                         x = xx, y = yhigh, color=ppars['color_dark'])
                            if (ppars['plot_median']):
                                sns.lineplot(ax=theaxis[i][em][ds], zorder=1, x = xx,
                                             y = ppars['CIbounds'][i][em][ds][2,:],
                                             color='black', linestyle='--')                                             
            elif (ppars['plot_type'] == '68-95CI_line'):
                # plot just the bounds of the 68% and 95% confidence intervals
                for i in indiv_names:
                    for em in evolve_models_for_indiv[i]:
                        for ds in datasets_for_indiv[i][em]:
                            xx = ppars['xmodels'][i][em]
                            ylowest = ppars['CIbounds'][i][em][ds][0,:]
                            ylow = ppars['CIbounds'][i][em][ds][1,:]
                            yhigh = ppars['CIbounds'][i][em][ds][3,:]                            
                            yhighest = ppars['CIbounds'][i][em][ds][4,:]                            
                            sns.lineplot(ax=theaxis[i][em][ds], zorder=1,
                                         x = xx , y = ylowest, color=ppars['color_light'])
                            sns.lineplot(ax=theaxis[i][em][ds], zorder=1,
                                         x = xx , y = ylow, color=ppars['color_dark'])
                            sns.lineplot(ax=theaxis[i][em][ds], zorder=1,
                                         x = xx , y = yhigh, color=ppars['color_dark'])
                            sns.lineplot(ax=theaxis[i][em][ds], zorder=1,
                                         x = xx, y = yhighest, color=ppars['color_light'])
                            if (ppars['plot_median']):
                                sns.lineplot(ax=theaxis[i][em][ds], zorder=1, x = xx,
                                             y = ppars['CIbounds'][i][em][ds][2,:],
                                             color='black', linestyle='--')
            elif (ppars['plot_type'] == '68-95CI_region'):
                # plot the 68% and 95% confidence regions
                for i in indiv_names:
                    for em in evolve_models_for_indiv[i]:
                        for ds in datasets_for_indiv[i][em]:
                            xx = ppars['xmodels'][i][em]
                            ylowest = ppars['CIbounds'][i][em][ds][0,:]
                            ylow = ppars['CIbounds'][i][em][ds][1,:]
                            yhigh = ppars['CIbounds'][i][em][ds][3,:]                            
                            yhighest = ppars['CIbounds'][i][em][ds][4,:]
                            theaxis[i][em][ds].fill_between(xx, ylowest, yhighest, zorder=1,
                                                            color=ppars['color_light'], alpha = 1)
                            theaxis[i][em][ds].fill_between(xx, ylow, yhigh, zorder=5,
                                                            color=ppars['color_dark'], alpha = 1)
                            if (ppars['plot_median']):
                                sns.lineplot(ax=theaxis[i][em][ds], zorder=6, x = xx,
                                             y = ppars['CIbounds'][i][em][ds][2,:],
                                             color='black', linestyle='--')                                             
            elif (ppars['plot_type'] == '68CI_region'):
                # plot the 68% confidence region
                for i in indiv_names:
                    for em in evolve_models_for_indiv[i]:
                        for ds in datasets_for_indiv[i][em]:
                            xx = ppars['xmodels'][i][em]
                            ylow = ppars['CIbounds'][i][em][ds][1,:]
                            yhigh = ppars['CIbounds'][i][em][ds][3,:]                            
                            theaxis[i][em][ds].fill_between(xx, ylow, yhigh, zorder=5,
                                                            color=ppars['color_dark'], alpha = 1)
                            if (ppars['plot_median']):
                                sns.lineplot(ax=theaxis[i][em][ds], zorder=6, x = xx,
                                             y = ppars['CIbounds'][i][em][ds][2,:],
                                             color='black', linestyle='--')                                             
            elif (ppars['plot_type'] == '95CI_region'):
                # plot the 95% confidence interval region
                for i in indiv_names:
                    for em in evolve_models_for_indiv[i]:
                        for ds in datasets_for_indiv[i][em]:
                            xx = ppars['xmodels'][i][em]
                            ylowest = ppars['CIbounds'][i][em][ds][0,:]
                            yhighest = ppars['CIbounds'][i][em][ds][4,:]
                            theaxis[i][em][ds].fill_between(xx, ylowest, yhighest, zorder=1,
                                                            color=ppars['color_light'], alpha = 1)
                            if (ppars['plot_median']):
                                sns.lineplot(ax=theaxis[i][em][ds], zorder=6, x = xx,
                                             y = ppars['CIbounds'][i][em][ds][2,:],
                                             color='black', linestyle='--')                                             
    # adjust all figures
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    # put into multi-page pdf
    with mplbp.PdfPages(ppars['plot_file']) as pdf:
        plt.figure('pg1')
        pdf.savefig()
        plt.close()
        plt.figure('pg2')
        pdf.savefig()
        plt.close()
        plt.figure('pg3')
        pdf.savefig()
        plt.close()
    print("\n\nFigure saved to:")
    print("\t" + ppars['plot_file'])
