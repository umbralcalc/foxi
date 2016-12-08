import sys
sys.path.insert(0, '' + '/foxi/foxisource/foxi.py')
from foxi import foxi

model_list = ['hi','mhi'] # List the model names to compute utilities
chains_column_numbers = [1,2] # Which columns are the relevant parameters in the chains? Order matters 
prior_column_numbers = [1,2] # Which columns are the relevant parameters in the prior? Order matters 
number_of_points = 10 # How many points to read in from the forecast data chains
number_of_prior_points = 10 # How many points to read in from the prior samples
error_vector = [0.1,0.1] # Set some future predictions for the measurements on each parameter corresponding to the columns in e.g. 'chains_column_numbers'

new_foxi_instance = foxi('INSERT YOUR PATH TO FOXI HERE') # Fire up a new instance of foxi
new_foxi_instance.set_chains('INSERT NAME OF CHAINS') # Set the name of the chains file 

# Running as normal from here...
new_foxi_instance.set_model_name_list(model_list)
new_foxi_instance.run_foxi(chains_column_numbers,prior_column_numbers,number_of_points,number_of_prior_points,error_vector)
