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
project_name = "hivtcell"
#--- Model hypothesis pars (to be set by mod-hyp module)
data_analysis_subproject = None
model_hyp = None
#--- File containing all data for project
data_file = "data/hivtcell_alldata.csv"
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
    'virus-decay-moi' : ['virus_decay_count_inf',
                         ],
}
#--- map: data-set to evolve-model
evolve_model = {
    'virus_decay_count_inf' : 'virus-decay-moi',
}
# 
#--- The data for each individual
#
# dictionaries for holding evolve-model and data-set information
#
evolve_models_for_indiv = None # dict[indiv] of lists
datasets_for_indiv = None      # dict[indiv][em] of lists
#
# dictionaries for holding the data
#
xvals_for_indiv = None         # dict[indiv][em][ds] of np arrays
yvals_for_indiv = None         # dict[indiv][em][ds] of np arrays
yvals_type_for_indiv = None    # dict[indiv][em][ds] of strings
#
# set of unique x values needed for each evolve-model run
#
evolve_models_xvals_for_indiv = None   # dict[indiv][em] of np arrays
#
# conversely, the indicies of the above xval vectors for each dataset
#
xindices_for_indiv = None              # dict[indiv][em][ds] of np arrays

#======================================================================#
#                                                                      #
#   When creating a new data project (or adding a new data analysis    #
#   subproject within a larger project), edit the prep_data() method   #
#   to import the data and select out subsets for a subproject, and    #
#   edit the plot_data() method to specify how to build the            #
#   associated multi-page figure associated to each subproject.        #
#                                                                      #
#======================================================================#

def prep_data():
    """
    Private method called by load_data() to put the project data file
    into the standard form for a Bayesian hierarchical / NLME analysis
    
    This method is specific to the particular project, so that all
    other methods (except plot_data()) can be independent of project.
    """
    #
    #=== Import the raw data file
    #
    #    The HIV-Tcell raw data file has columns:
    #
    #     [cell_type, exp_type, exp_trial, cell_donor, virus_date,
    #      virus_vol_ul, x_data_type, X, y_data_type,
    #      rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8]
    #
    df = pd.read_csv(data_file)
    #
    #=== Create unique labels of individuals
    #
    #    Although the cell donors whose blood was used for
    #    a given trial are the actual individuals, their
    #    Tcells were prepared in one of four different
    #    'cell_type'(s): [act, ecm, ecp, rest].  So
    #    donor+celltype should identify the "individuals" for
    #    the Bayesian hierarchical analysis.
    #
    #    For the purposes of specifying initial guesses for
    #    parameters, it would be easier to identify the
    #    individuals by celltype-exper-trial.  The only
    #    repeated use of donor cells was on the uninfected
    #    and infected time-series, so we will merge those
    #    two identifiers and then use this more informative
    #    description.
    #
    # write inf/uninf timeseries as "timeseries" in a dummy col
    df['exp_type_dummy'] = df['exp_type']
    df.at[( (df['exp_type'] == 'timeseries_uninf')
            | (df['exp_type'] == 'timeseries_inf') ), 'exp_type_dummy'] = \
                'timeseries'
    #
    # set the unique informative individual names, which are,
    # for this project:
    #
    # [act_drugstopping_efav_1, act_drugstopping_efav_2, act_drugstopping_efav_3,
    #  act_drugstopping_ralt_1, act_drugstopping_ralt_2,
    #  act_timeseries_1, act_timeseries_2, act_timeseries_3,
    #  act_virus_decay_1, act_virus_decay_2,
    #  act_virus_dilution_1, act_virus_dilution_2,
    #  ecm_drugstopping_efav_1, ecm_drugstopping_efav_2, ecm_drugstopping_efav_3,
    #  ecm_drugstopping_ralt_1, ecm_drugstopping_ralt_2,
    #  ecm_timeseries_1, ecm_timeseries_2, ecm_timeseries_3,
    #  ecm_virus_decay_1, ecm_virus_decay_2,
    #  ecm_virus_dilution_1, ecm_virus_dilution_2,
    #  ecp_drugstopping_efav_1, ecp_drugstopping_efav_2, ecp_drugstopping_efav_3,
    #  ecp_drugstopping_ralt_1, ecp_drugstopping_ralt_2,
    #  ecp_timeseries_1, ecp_timeseries_2, ecp_timeseries_3,
    #  ecp_virus_decay_1, ecp_virus_decay_2,
    #  ecp_virus_dilution_1, ecp_virus_dilution_2,
    #  rest_drugstopping_efav_1, rest_drugstopping_efav_2, rest_drugstopping_efav_3,
    #  rest_drugstopping_ralt_1, rest_drugstopping_ralt_2,
    #  rest_timeseries_1, rest_timeseries_2, rest_timeseries_3,
    #  rest_virus_decay_1, rest_virus_decay_2,
    #  rest_virus_dilution_1, rest_virus_dilution_2, rest_virus_dilution_3]
    #
    df['indiv'] = df['cell_type'] + '_' + df['exp_type_dummy'] \
        + '_' + df['exp_trial'].astype(str)
    #
    #=== Clean up the dataframe
    #
    #--- Remove rows of infected cell data for uninfected trials
    df = df[ ~( (df['exp_type'] == 'timeseries_uninf')
               & ( (df['y_data_type'] == 'frac_inf')
                   | (df['y_data_type'] == 'dead_frac_of_inf') ) )]
    #
    #--- Get total number of cells infected for the virus decay experiments
    #
    vdf = df[ (df['exp_type'] == 'virus_decay')
              & (df['y_data_type'] == 'count_tot')].copy()
    vdf.reset_index(drop=True, inplace=True)
    #
    #--- Loop through the frac_inf data, and multiply
    #    onto the associated count_tot data
    #
    repstrings = ['rep1', 'rep2', 'rep3', 'rep4', 'rep5', 'rep6', 'rep7', 'rep8']
    curindex = 0
    for index, row in df.iterrows():
        if ( (row.exp_type == 'virus_decay')
             & (row.y_data_type == 'frac_inf') ):
            for rep in repstrings:
                if not pd.isnull(row[rep]):
                    vdf.at[curindex, rep] = row[rep] * vdf.at[curindex, rep]
                    # Set any zero-valued points to Nan (will be fitting log(val))
                    if (vdf.at[curindex, rep] == 0):
                        vdf.loc[curindex, rep] = np.nan
            curindex +=1
    vdf['y_data_type'] = 'count_inf'
    #
    #--- Append the virus_decay-count_inf data to the full dataframe
    #   (this is not independent, but is what we want to use for virus_decay)
    df = df.append(vdf)
    #
    #=== Create unique data-set names
    #
    #    We'll specify datasets by: exp_type + y_data_type
    #
    #    (these names do not include cell type, since each cell type
    #     has same data-set(s), for which the associated evolve-model
    #     will just be run with different parameters)/
    #
    df['data_set'] = df['exp_type'] + '_' + df['y_data_type']
    #
    #
    #=== Flatten any replicate data, putting each dependent
    #    data point into its own row
    #
    # Set 'rep1' column to be 'Y'
    df['Y'] = df['rep1']
    #
    # move the repN (N>1) into their own rows
    newrows = []
    for index, row in df.iterrows():
        r = row.copy()
        for rep in repstrings[1:]:
            if not pd.isnull(row[rep]):
                r['Y'] = r[rep]
                newrows.append(r)
    df = df.append(pd.DataFrame(newrows), ignore_index=True)
    # delete any nan-valued points
    df = df[~( df['Y'].isna() )]
    df.to_csv("junk1.csv", index=False)    
    #
    #=== Select out only the data sets for the chosen subproject
    #
    #--- First check if only a single cell type should be
    #    used and restrict to that data.  In that case the
    #    data_analysis_subproject would be of the form, e.g.,
    #
    #         rest_infected-timeseries
    #
    dasp_celltype = data_analysis_subproject.split('_')[0]
    if (dasp_celltype in ['act', 'rest', 'ecp', 'ecm']):
        df = df[ (df['cell_type'] == dasp_celltype) ]
    else:
        dasp_celltype = 'all'
    #--- Set data to be used for each data analysis subproject
    if (data_analysis_subproject == 'virus-decay'):
        # VIRUS DECAY: only using the number infected data
        df = df[ (df['data_set'] == 'virus_decay_count_inf') ]
    else:
        print("***Error: Data Analysis Subproject" 
              + data_analysis_subproject + " not recognized.")
        exit(0)
    #
    #=== Sort dataframe such that the order of individuals
    #    is clear (for specifying initial values in the
    #    model-hypothesis module), and such that the
    #    dependent values X are ordered.
    #
    #    Here, since our individual names are:
    #
    #               cell_type + exp_type + exp_trial
    #
    #    we sort by:
    #
    #       [cell_type, exp_type, exp_trial, y_data_type, X]
    #
    df = df.sort_values(['cell_type', 'exp_type', 'exp_trial', 'y_data_type', 'X'])
    #
    #=== Retain only the standard columns:
    #
    #       [indiv, data_set, x_data_type, X, y_data_type, Y]
    #
    df = df[['indiv', 'data_set', 'x_data_type', 'X', 'y_data_type', 'Y']]
    df.to_csv("junk2.csv", index=False)
    #
    #--- Return the ordered list of distinct individual names for
    #    this data analysis subproject, along with the dataframe.
    #
    inames = df['indiv'].to_numpy()
    _ , idx = np.unique(inames, return_index=True)
    inames = inames[np.sort(idx)]
    return inames, df

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

#======================================================================#
#                                                                      #
#                No adjustments should be needed below                 #
#                                                                      #
#======================================================================#    

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
    indiv_names, df = prep_data()
    Nindiv = len(indiv_names)
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

