import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.backends.backend_pdf as mplbp
import matplotlib.pyplot as plt
import seaborn as sns
# make default style for seaborn
sns.set()

#=======================================================
#                   Parameters
#=======================================================
#-------------------------------------------------------
#
#          set the name of this data project
#
#-------------------------------------------------------
project_name = "hivtcell"
#--- Model hypothesis pars (to be set by mod-hyp module)
data_analysis_subproject = None
model_hyp = None
#-------------------------------------------------------
#
#  Specify the filepath containing all data for project
#
#-------------------------------------------------------
data_file = "data/hivtcell_alldata.csv"
#
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
#-------------------------------------------------------
#
# Set the list of data-sets produced by each evolvemodel
#
#-------------------------------------------------------
data_sets = {
    'virus-decay-moi' : ['virus_decay_count_inf',
                         ],
}
#-------------------------------------------------------
#
#   Set the evolve model that produces each data-set
#
#-------------------------------------------------------
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
    #-------------------------------------------------------
    #
    #                Import the data file
    #
    #-------------------------------------------------------
    #
    #    The HIV-Tcell raw data file has columns:
    #
    #     [cell_type, exp_type, exp_trial, cell_donor, virus_date,
    #      virus_vol_ul, x_data_type, X, y_data_type,
    #      rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8]
    #
    df = pd.read_csv(data_file)
    #-------------------------------------------------------
    #
    #    Identify and set names for the "individuals"
    #
    #-------------------------------------------------------
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
    # This specification of individuals leaves these names:
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
    #-------------------------------------------------------
    #
    # Perform any necessary clean-up/pre-analysis of the data
    #
    #-------------------------------------------------------    
    #
    #--- Remove rows of infected cell data for uninfected trials
    #
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
    df.reset_index(inplace=True)
    #-------------------------------------------------------
    #
    #        Identify and give names to each data set
    #
    #-------------------------------------------------------        
    #
    #  We'll specify datasets by: exp_type + y_data_type
    #
    #  (these names do not include cell type, since each
    #   cell-type uses same data-set(s), for which the
    #   associated evolve-model just run with different
    #   parameters)
    #
    df['data_set'] = df['exp_type'] + '_' + df['y_data_type']
    #-------------------------------------------------------
    #
    #    Flatten any replicate data to one point per row
    #
    #-------------------------------------------------------    
    #
    # Set 'rep1' column to be 'Y'
    df['Y'] = df['rep1']
    #
    # move the repN (N>1) into their own rows
    newrows = []
    for index, row in df.iterrows():
        r = df.iloc[index]
        for rep in repstrings[1:]:
            if not pd.isnull(row[rep]):
                r['Y'] = r[rep]
                # must append a copy or you'll get the same value
                newrows.append(r.copy())
    df = df.append(pd.DataFrame(newrows), ignore_index=True)
    # delete any nan-valued points
    df = df[~( df['Y'].isna() )]
    df.to_csv("junk1.csv", index=False)
    #-------------------------------------------------------
    #
    #     Select only the data sets for the DA-subproject
    #
    #-------------------------------------------------------        
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
    #
    #--- Then select data to be used for each data
    #    analysis subproject
    #
    if (data_analysis_subproject == 'virus-decay'):
        # VIRUS DECAY: only using the number infected data
        df = df[ (df['data_set'] == 'virus_decay_count_inf') ]
    else:
        # NONE RECOGNIZED
        print("***Error: Data Analysis Subproject" 
              + data_analysis_subproject + " not recognized.")
        exit(0)
    #-------------------------------------------------------
    #
    # Sort the dataframe, to know that order of individuals
    #
    #-------------------------------------------------------            
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
    #-------------------------------------------------------
    #
    #   Retain only these standard columns:
    #
    #     [indiv, data_set, x_data_type, X, y_data_type, Y]
    #
    #-------------------------------------------------------
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
    #-------------------------------------------------------    
    #
    #     Establish a different set of figures for each
    #     data_analysis_subproject
    #
    #-------------------------------------------------------
    #
    if (data_analysis_subproject == 'virus-decay'):
        #-------------------------------------------------------
        #
        #       Create the figure objects for each page
        #
        #-------------------------------------------------------        
        #
        #=== common parameters 
        #
        #--- same shape size of each page
        nrows, ncols = 2,2
        figx, figy = 8, 10.5
        #
        #--- plotting pars
        #
        # logscale on y
        #   (if you want logscale, you have to do it *after* seaborn plots)
        plotlogy = True
        #
        # common y/x --- if to be set automatically, leave as None
        #
        xlims = [-0.2,4.2]
        ylims = [0.8, 2e5]
        #
        #
        # data point color
        datacolor = 'black'
        #
        # Figures for each page (label each figure object with a "num")
        #
        figure_nums = ['pg1', 'pg2']
        #
        #=== Page 1  
        #
        fig1, axes1 = plt.subplots(nrows, ncols, figsize=(figx, figy), num='pg1')
        #
        # to make large common x- and y- labels, create a hidden axis
        #
        ax1 = make_hidden_axis(fig1, "Time of incubation (d)",
                               "Number infected cells")
        #
        #=== Make a super-title on page 1, declaring mod-hyp
        #
        fig1.suptitle("Virus decay w/ model hypothesis = " + model_hyp)
        #
        #=== Page 2
        #
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(figx, figy), num='pg2')
        # add a hidden axis to make common x/y-labels
        ax2 = make_hidden_axis(fig2, "Time of incubation (d)",
                               "Number infected cells")
        #
        #-------------------------------------------------------
        #
        #         Plot each individual's data in their axis
        #
        #-------------------------------------------------------        
        #
        # set individual names
        #
        act1 = 'act_virus_decay_1'; act2 = 'act_virus_decay_2'
        rest1 = 'rest_virus_decay_1'; rest2 = 'rest_virus_decay_2'
        ecm1 = 'ecm_virus_decay_1'; ecm2 = 'ecm_virus_decay_2'
        ecp1 = 'ecp_virus_decay_1'; ecp2 = 'ecp_virus_decay_2'
        #
        # get evolvemodel and dataset names (same for all)
        #
        em = evolve_models_for_indiv[indiv_names[0]][0]
        ds = datasets_for_indiv[indiv_names[0]][em][0]
        #
        # activate first page figure, plot ACT and REST
        #
        plt.figure('pg1')
        ind = 0
        for i in [act1, act2, rest1, rest2]:
            x = xvals_for_indiv[i][em][ds]
            y = yvals_for_indiv[i][em][ds]
            r = ind // ncols
            c = ind % ncols
            axes1[r,c].set_title(i)
            # use zorder to keep data points on "top"
            sns.scatterplot(ax=axes1[r,c], zorder=10, x=x, y=y, color=datacolor)
            ind +=1
        # activate second page figure, plot ECM and ECP
        plt.figure('pg2')
        ind = 0
        for i in [ecm1, ecm2, ecp1, ecp2]:
            x = xvals_for_indiv[i][em][ds]
            y = yvals_for_indiv[i][em][ds]
            r = ind // ncols
            c = ind % ncols
            axes2[r,c].set_title(i)            
            sns.scatterplot(ax=axes2[r,c], zorder=10, x=x, y=y, color=datacolor)
            ind += 1
        #
        # if just plotting data, exit  (skip to end to set output)
        #
        if (ppars['plot_type'] == 'only_data'):
            # just save the plot
            pass
        else:
            #-------------------------------------------------------
            #
            #           Assign axes to each [i][em][ds]
            #
            #-------------------------------------------------------
            #
            # for virus-decay there is only one evolve-model
            # and one associated dataset... same for each indiv
            #
            em = evolve_models_for_indiv[indiv_names[0]][0]
            ds = datasets_for_indiv[indiv_names[0]][em][0]
            theaxis = {}
            theaxis[act1] = {}
            theaxis[act1][em] = {}
            theaxis[act1][em][ds] = axes1[0,0]            
            theaxis[act2] = {}
            theaxis[act2][em] = {}
            theaxis[act2][em][ds] = axes1[0,1]            
            theaxis[rest1] = {}
            theaxis[rest1][em] = {}
            theaxis[rest1][em][ds] = axes1[1,0]            
            theaxis[rest2] = {}
            theaxis[rest2][em] = {}
            theaxis[rest2][em][ds] = axes1[1,1]            
            theaxis[ecm1] = {}
            theaxis[ecm1][em] = {}
            theaxis[ecm1][em][ds] = axes2[0,0]            
            theaxis[ecm2] = {}
            theaxis[ecm2][em] = {}
            theaxis[ecm2][em][ds] = axes2[0,1]            
            theaxis[ecp1] = {}
            theaxis[ecp1][em] = {}
            theaxis[ecp1][em][ds] = axes2[1,0]            
            theaxis[ecp2] = {}
            theaxis[ecp2][em] = {}
            theaxis[ecp2][em][ds] = axes2[1,1]            
            #=== do the plotting of models over data
            plot_posterior_over_data(ppars, theaxis)
    else:
        #
        # put other data_analysis_subproject possibilities above in elif blocks
        #
        print("***Error: data analysis subproject \"" + data_analysis_subproject
              + "\" not recognized.")
        exit(0)
    #-------------------------------------------------------
    #
    #         No further adjustment should be needed
    #
    #-------------------------------------------------------
    #
    #=== set logscale
    #
    #  (this must be done *after* seaborn makes its plots)
    #
    for r in range(nrows):
        for c in range(ncols):
            if plotlogy:
                axes1[r,c].set(yscale='log')
                axes2[r,c].set(yscale='log')
            if xlims is not None:
                axes1[r,c].set_xlim(xlims)
                axes2[r,c].set_xlim(xlims)                
            if ylims is not None:
                axes1[r,c].set_ylim(ylims)
                axes2[r,c].set_ylim(ylims)                
    #
    #=== put into multi-page pdf
    #
    with mplbp.PdfPages(ppars['plot_file']) as pdf:
        for p in figure_nums:
            plt.figure(p)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    print("\n\nFigure saved to:")
    print("\t" + ppars['plot_file'])

#======================================================================#
#                                                                      #
#                No adjustments should be needed below                 #
#                                                                      #
#======================================================================#    

def make_hidden_axis(fig, xlabel, ylabel):
    ax = fig.add_subplot(111, frameon=False)
    ax.grid(False)
    plt.tick_params(labelcolor='none', which='both',
                    top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel, fontsize=20, labelpad=10)
    plt.ylabel(ylabel, fontsize=20, labelpad=10)
    return ax

def plot_posterior_over_data(ppars, theaxis):
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
                    theaxis[i][em][ds].fill_between(xx, ylowest, yhighest,
                                                    zorder=1,
                                                    color=ppars['color_light'],
                                                    alpha = 1)
                    theaxis[i][em][ds].fill_between(xx, ylow, yhigh, zorder=5,
                                                    color=ppars['color_dark'],
                                                    alpha = 1)
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
                                                    color=ppars['color_dark'],
                                                    alpha = 1)
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
                    theaxis[i][em][ds].fill_between(xx, ylowest, yhighest,
                                                    zorder=1,
                                                    color=ppars['color_light'],
                                                    alpha = 1)
                    if (ppars['plot_median']):
                        sns.lineplot(ax=theaxis[i][em][ds], zorder=6, x = xx,
                                     y = ppars['CIbounds'][i][em][ds][2,:],
                                     color='black', linestyle='--')

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

