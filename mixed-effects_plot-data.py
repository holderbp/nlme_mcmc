import sys
import functools
import numpy as np
import pandas as pd
import seaborn as sns

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#            Import the chosen model hypothesis
#
#   This must match the input "sample" data file below so that
#   the parameters in the sample are correctly identified.
#                                                              
#==============================================================
#import module_sleepstudy_modhyp_REmb_nocov_COMsig as mod
#import module_sleepstudy_modhyp_REb_COMmsig as mod
#import module_sleepstudy_modhyp_REm_COMbsig as mod
#import module_sleepstudy_modhyp_REnone_COMsig as mod
#import module_sleepstudy_modhyp_REnone_INDIVmbsig as mod
import module_sleepstudy_modhyp_REbsigma_COMm as mod
#//////////////////////////////////////////////////////////////

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#==============================================================
#                    Plotting parameters
#==============================================================
#
# get some seaborn colors from a palette
#
#  can explore in "jupyter notebook" session
#
purple = sns.color_palette('gist_stern')[0]
blue = sns.color_palette('gist_stern')[2]
grey = sns.color_palette('gist_stern')[3]
dark_grey = sns.color_palette('bone')[2]
light_grey = sns.color_palette('bone')[5]        
dark_blue = sns.color_palette('PuBu')[3]
light_blue = sns.color_palette('PuBu')[2]        
dark_red = sns.color_palette('Reds')[3]
light_red = sns.color_palette('Reds')[1]        
dark_red = sns.color_palette('RdBu')[1]
light_red = sns.color_palette('RdBu')[2]
#
#  plot_type:  ['only_data', 'median', 'spaghetti',
#               '68CI_line', '95CI_line', '68-95CI_line',
#               '68CI_region', '95CI_region', '68-95CI_region']
#
#  Npoints: <number of points in each fit-model dataset>
#
plotpars = {
    'plot_type' : '68-95CI_region',
    'Npoints' : 1000,
    'Nsamp' : None,  # set to None for all (might use less for spaghetti)
    'plot_median' : True,
    'color_dark' : dark_blue,
    'color_light': light_blue,
}
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
    plotpars['sample_file'] = sys.argv[1]
else:
    #plotpars['sample_file'] = "output/2022-06-08_1646_samples_sleepstudy_response-time-vs-days-deprivation_RE-single-group-mb-nocov-common-sigma.dat"
    #plotpars['sample_file'] = "output/2022-06-11_2251_samples_sleepstudy_response-time-vs-days-deprivation_no-RE-common-sigma.dat"
    #plotpars['sample_file'] = "output/2022-06-11_1859_samples_sleepstudy_response-time-vs-days-deprivation_RE-single-group-m-common-b-sigma.dat"
    #plotpars['sample_file'] = "output/2022-06-10_1243_samples_sleepstudy_response-time-vs-days-deprivation_RE-single-group-b-common-m-sigma.dat"
    plotpars['sample_file'] = "output/2022-06-11_2251_samples_sleepstudy_response-time-vs-days-deprivation_no-RE-indiv-mbsigma.dat"
#
#--- Check that the model hypothesis of the samples matches
#    that of the imported mod-hyp module
modhyp = plotpars['sample_file'].split('_')[-1].split('.')[0]
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
parts = plotpars['sample_file'].split('/')[1].split('_')
sample_tag = parts[0] + "_" + parts[1]  #date-time of the sample file
plotdir = "plots/"
plotpars['plot_file'] = plotdir + sample_tag \
    + "_" + plotpars['plot_type'] + ".pdf"
#//////////////////////////////////////////////////////////////

# run the data/evolve-model sample plotting from the
# model hypothesis module (which calls the data module)
mod.plot_data(plotpars)
