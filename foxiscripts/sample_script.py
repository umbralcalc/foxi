import sys
sys.path.append('INSERT PATH TO foxisource/')
from foxi import foxi

model_list = ['###.txt','###.dat','ADD MODELS HERE'] # List the model file names to compute utilities
chains_column_numbers = [0,1] # Which columns are the relevant parameters in the chains? Order matters 
prior_column_numbers = [[0,1],[0,1]] # Which columns are the relevant parameters in the prior? Order matters 
column_types = ['flat','flat'] # Set the format of each column in the chains (if needed) - choose from 'flat', 'log' and 'log10'
prior_column_types = ['flat','flat'] # Set the format of each column in the prior (if needed) - choose from 'flat', 'log' and 'log10'
number_of_points = 100 # How many points to read in from the forecast data chains
number_of_prior_points = 100 # How many points to read in from the prior samples
error_vector = [0.02,0.02,0.02] # Set some future predictions for the measurements on each parameter corresponding to the columns in e.g. 'chains_column_numbers'

new_foxi_instance = foxi('INSERT PATH TO foxi/') # Fire up a new instance of foxi
new_foxi_instance.set_chains('INSERT FILENAME') # Set the name of the chains file 
foxi_instance.set_column_types(column_types) # Comment/uncomment this feature where needed/not needed
foxi_instance.set_prior_column_types(prior_column_types) # Comment/uncomment this feature where needed/not needed

# Running to output foxiplot data file first...
new_foxi_instance.set_model_name_list(model_list)
new_foxi_instance.run_foxi(chains_column_numbers,prior_column_numbers,number_of_points,number_of_prior_points,error_vector)


foxiplot_file = 'foxiplots_data.txt' # Name of the foxiplot data file to analyse using 'rerun_foxi'
number_of_foxiplot_samples = 100 # How many points to read in from the foxiplot data file 
prior_types = ['flat','flat','flat'] # Set the type of predictive prior weight in each column of the fiducial points - choose from 'flat' or 'log'
TeX_output = True # Output a fancy LaTeX table with the relevant data for each model pair

rerun_foxi_instance = foxi('/home/.../foxi') # Fire up a new instance of foxi for a rerun to obtain secondary quantities
rerun_foxi_instance.rerun_foxi(foxiplot_file,number_of_foxiplot_samples,prior_types,TeX_output=TeX_output)
