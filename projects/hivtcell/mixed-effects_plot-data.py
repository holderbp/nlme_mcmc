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
#
#=== virus-decay 
#
#import module_hivtcell_modhyp_virusdecay_IPOP_none_INOPOP_A_B_thalf_COM_sigma as mod
#import module_hivtcell_modhyp_virusdecay_IPOP_none_INOPOP_A_B_COM_thalf_sigma as mod
#import module_hivtcell_modhyp_virusdecay_IPOP_thalf_INOPOP_A_B_COM_sigma as mod
#import module_hivtcell_modhyp_virusdecay_IPOP_thalf_sigma_INOPOP_A_B_COM_none as mod
import module_hivtcell_modhyp_virusdecay_IPOP_sigma_INOPOP_A_B_COM_thalf as mod
#
#=== uninfected timeseries 
#

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
    'Nsamp' : 1000,  # set to None for all (might use less for spaghetti)
    'plot_median' : True,
    'color_dark' : dark_blue,
    'color_light': light_blue,
}

def get_modhyp_from_filename(filename):
    # file should be:
    #
    #  output/YYYY-mm-dd_HHMM_samples_project_subproject_the_mod-hyp.dat
    #
    list_parts = filename.split('_')
    lp = np.array(list_parts)[5:]
    modhyp = ""
    for i in range(len(lp)):
        modhyp += lp[i] + '_'
    modhyp = modhyp.split('.')[0]
    return modhyp

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
modhyp = get_modhyp_from_filename(plotpars['sample_file'])
plotpars['sample_file'].split('_')[-1].split('.')[0]
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
