import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text',usetex=True)
import pylab as pl

class foxi:
# Initialize the 'foxi' method class
    

    def __init__(self,path_to_foxi_directory):
        self.path_to_foxi_directory = path_to_foxi_directory # Set this upon declaration
        self.current_data_chains = ''
        self.model_name_list = [] 
        self.set_chains
        self.set_model_name_list
        self.decisivity_lnB_utility_functions
        self.jensen_shannon_div_utility_function
        self.information_radius_utility_function
        self.foxi_mode = 'null'
        self.run_foxi
        self.chains_directory = 'foxichains/'
        self.priors_directory = 'foxipriors/'
        self.output_directory = 'foxioutput/'
        self.plots_directory = 'foxiplots/'
        self.source_directory = 'foxisource/'   
        self.gaussian_forecast
        self.flashy_foxi
        self.set_column_types
        self.column_types = []
        self.column_functions 
        self.column_types_are_set = False
        self.evaluate_Fab
        #self.evaluate_Sabc
        #self.evaluate_Qabcd


    def set_chains(self,name_of_chains): 
    # Set the file name of the chains of the current data set
        self.current_data_chains = name_of_chains


    def set_model_name_list(self,model_name_list): 
    # Set the name list of models which can then be referred to by number for simplicity
        self.model_name_list = model_name_list


    def set_column_types(self,column_types):
    # Input a list of strings to set the type of column transformation in both prior and data chains i.e. flat, log, log10, ...
        self.column_types = column_types
        self.column_types_are_set = True # All columns are as input unless this is True


    def column_functions(self,i,input_value):
    # Choose to transform whichever column
        if self.column_types[i] == 'flat': return input_value
        if self.column_types[i] == 'log': return np.log(input_value)
        if self.column_types[i] == 'log10': return np.log10(input_value)

    
    def evaluate_Fab(self,chains_column_numbers,current_data_central_point_vector,stepsizes,number_of_points):
    # Compute the Fisher matrix (or 1st approximation) from the likelihood chains which can be used to efficiently evaluate the Kullback-Leibler divergence utility
        '''
        Inputs to evaluate_Fab:       


        chains_column_numbers              =  A list of the numbers of the columns in the chains that correspond to
                                              the parameters of interest, starting with 0. These should be in the same
                                              order as all other structures.
     

        current_data_central_point_vector  = 

        
        stepsizes                          = 


        number_of_points                   =  The number of points specified to be read off from the current data chains.        


        '''

        differences = np.zeros(len(chains_column_numbers)) # Initialise the array of zeros for the differences summation
        first_deriv_differences = []
        for i in range(0,len(chains_column_numbers)): first_deriv_differences.append(np.zeros(len(chains_column_numbers))) # Initialise zeros for derivatives
        
        running_total = 0 # Initialize a running total of points read in from the chains

        with open(self.path_to_foxi_directory + '/' + self.chains_directory + self.current_data_chains) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the chains
            for line in file:
                columns = line.split()
                fiducial_point_vector = [] 
                for j in range(0,len(chains_column_numbers)):
                    if self.column_types_are_set == True: fiducial_point_vector.append(self.column_functions(j,float(columns[chains_column_numbers[j]])))
                    if self.column_types_are_set == False: fiducial_point_vector.append(float(columns[chains_column_numbers[j]])) # All columns are flat priors unless this is True
                differences += np.asarray(fiducial_point_vector)-np.asarray(current_data_central_point_vector)
                for i in range(0,len(chains_column_numbers)): first_deriv_differences[i] += np.asarray(fiducial_point_vector)-np.asarray(current_data_central_point_vector) - np.asarray(stepsizes) # Compute first dervative differences 
                running_total+=1 # Also add to the running total
                if running_total >= number_of_points: break # Finish once reached specified number of data points 

        covmat = np.array([[differences[i]*differences[j]/float(number_of_points) for j in range(0,len(chains_column_numbers))] for i in range(0,len(chains_column_numbers))])
        covmat_change = [np.array([[first_deriv_differences[i][k]*first_deriv_differences[j][k]/float(number_of_points) for j in range(0,len(chains_column_numbers))] for i in range(0,len(chains_column_numbers))]) for k in range(0,len(chains_column_numbers))]
        first_deriv_covmat = [covmat_change[i] - covmat for i in range(0,len(chains_column_numbers))]
        # Central and first derivative covariance matrices

        inverse_covmat = np.linalg.inv(covmat)
        Fab = [[0.5*np.sum(inverse_covmat*first_deriv_covmat[i]*inverse_covmat*first_deriv_covmat[i]) for i in range(0,len(chains_column_numbers))] for j in range(0,len(chains_column_numbers))]  # Fisher matrix calculation  

        output_data_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiout_likelihood_Fab.txt",'w') 
        for i in range(0,len(Fab[0])): 
            for j in range(0,len(Fab[0])):     
                output_data_file.write(str(Fab[i][j]) + "\t")
            output_data_file.write("\n")
        output_data_file.close()
        # Output to a newly created data file


    def decisivity_lnB_utility_functions(self,fiducial_point_vector,forecast_data_function,prior_column_numbers,number_of_prior_points,error_vector):
    # The decisivity and lnB utility functions defined in arxiv:1639.3933, but first used in arxiv:1012.3195
        '''
        Inputs to decisivity_utility_function:       


        fiducial_point_vector            =  Forecast distribution fiducial central values


        forecast_data_function           =  A function like gaussian_forecast


        prior_column_numbers             =  A 2D list [i][j] of the numbers of the columns j in the prior for model i that 
                                            correspond to the parameters of interest, starting with 0. These should 
                                            be in the same order as all other structures.


        number_of_prior_points           =  The number of points specified to be read off from the prior values.


        error_vector                     =  Fixed predicted future measurement errors corresponding to the fiducial_point_vector 


        '''
        
        decisivity = np.zeros(len(self.model_name_list))
        # Initialize an array of values of the decisivity for the number of models (reference model is element 0)
        E = np.zeros(len(self.model_name_list))
        # Initialize an array of values of the Bayesian evidence for the number of models (reference model is element 0)
        lnB = np.zeros(len(self.model_name_list))
        # Initialize an array of values of the log Bayes factor for the number of models (reference model is element 0)

        running_total = 0 # Initialize a running total of points read in from the prior

        for i in range(0,len(self.model_name_list)):
            with open(self.path_to_foxi_directory + '/' + self.priors_directory + 'bayesinf_' + self.model_name_list[i] + '.txt') as file:
            # Compute quantities in loop dynamically as with open(..) reads off the prior values
                for line in file:
                    columns = line.split()
                    prior_point_vector = [] 
                    for j in range(0,len(prior_column_numbers[i])): # Take a prior point 
                        if self.column_types_are_set == True: prior_point_vector.append(self.column_functions(j,float(columns[prior_column_numbers[i][j]])))
                        if self.column_types_are_set == False: prior_point_vector.append(float(columns[prior_column_numbers[i][j]])) # All columns are as input unless this is True
                    E[i] += forecast_data_function(prior_point_vector,fiducial_point_vector,error_vector) 
                    # Calculate the forecast probability and therefore the contribution to the Bayesian evidence for each model
                    running_total+=1 # Also add to the running total                         
                    if running_total >= number_of_prior_points: break # Finish once reached specified number of prior points

        for j in range(0,len(self.model_name_list)):  
            lnB[j] = np.log(E[j]) - np.log(E[0]) # Compute log Bayes factor utility
            if np.log(E[j]) - np.log(E[0]) <= -5.0:
                if j > 0: decisivity[j] = 1.0 # Compute decisivity utility (for each new forecast distribution this is either 1 or 0) 
 
        return [lnB,decisivity] # Output both utilities        


    def jensen_shannon_div_utility_function(self): 
    # A utility based on the total Jensen-Shannon divergence introduced in arxiv:1606.06758
        return 1.0

    
    def information_radius_utility_function(self,fiducial_point_vector,error_vector,current_data_central_point_vector,current_data_error_vector):
        # The information radius utility function defined in arxiv:1639.3933
        '''
        Inputs to decisivity_utility_function:       


        fiducial_point_vector              =  Forecast distribution fiducial central values


        forecast_data_function             =  A function like gaussian_forecast


        current_data_central_point_vector  =

         
        current_data_error_vector          =
  

        error_vector                       =  Fixed predicted future measurement errors corresponding to the fiducial_point_vector 


        '''

        fisher_metric = np.zeros(len(error_vector),len(error_vector))
        delta_full_coordinates = np.zeros(2*len(error_vector))

        for i in range(0,2*len(error_vector)):
            if i < len(error_vector): 
                fisher_metric[i,i] = 1.0/(current_data_error_vector**2.0)
                delta_full_coordinates = (fiducial_point_vector[i]-current_data_central_point_vector[i])
            if i >= len(error_vector): 
                fisher_metric[i,i] = 1.0/(2.0*(current_data_error_vector**4.0))
                delta_full_coordinates = (error_vector[i]-current_data_error_vector[i])

        information_radius = (1.0/8.0)*(sum((delta_full_coordinates.T)*fisher_metric*(delta_full_coordinates)))
 
        return information_radius # Output the IRad

   
    def gaussian_forecast(self,prior_point_vector,fiducial_point_vector,error_vector):
    # Posterior probability at some point - prior_point_vector - in arbirary dimensions using a Gaussian 
    # forecast distribution given a fiducial point vector and error vector
        prior_point_vector = np.asarray(prior_point_vector) 
        fiducial_point_vector = np.asarray(fiducial_point_vector)      
        error_vector = np.asarray(error_vector)
        return (1.0/np.sqrt(2.0*np.pi))*np.exp(-(0.5*sum(((prior_point_vector-fiducial_point_vector)/error_vector)**2))-sum(np.log(error_vector)))


    def run_foxi(self,chains_column_numbers,prior_column_numbers,number_of_points,number_of_prior_points,error_vector,foxi_mode='deci_ExlnB'): 
    # This is the main algorithm to compute expected utilites and compute other quantities
        ''' 
        Quick usage and settings:
                 

        chains_column_numbers      =  A list of the numbers of the columns in the chains that correspond to
                                      the parameters of interest, starting with 0. These should be in the same
                                      order as all other structures.


        prior_column_numbers       =  A 2D list [i][j] of the numbers of the columns j in the prior for model i that 
                                      correspond to the parameters of interest, starting with 0. These should 
                                      be in the same order as all other structures.       


        number_of_points           =  The number of points specified to be read off from the current data chains.


        number_of_prior_points     =  The number of points specified to be read off from the prior.


        error_vector               =  Fixed predicted future measurement errors corresponding to the fiducial_point_vector.

        
        foxi_mode                   = 'Exjsd'             - Sets up foxi to compute just the total expected
                                                            Jensen-Shannon divergence between all models in 
                                                            model_name_list.
                                      'deci_ExlnB'        - Sets up foxi to compute both the decisivity and 
                                                            expected log Bayes factor utilities where we
                                                            set model_name_list[0] as the reference model.
                                      'deci_ExlnB_Exjsd'  - Sets up foxi to compute 'deci_ExlnB' and 'Exjsd'.

                                      < Default >         - This is set to 'deci_ExlnB'. 


        '''
        self.flashy_foxi() # Display propaganda      

        forecast_data_function = self.gaussian_forecast # Change the forecast data function here (if desired)

        running_total = 0 # Initialize a running total of points read in from the chains

        if foxi_mode == 'deci_ExlnB':
            deci = np.zeros(len(self.model_name_list))
            # Initialize an array of values of the decisivity utility function for the number of models (reference model is element 0)
            lnB = np.zeros(len(self.model_name_list))
            # Initialize an array of values of the log Bayes factor for the number of models (reference model is element 0)
            decisivity = np.zeros(len(self.model_name_list))
            # Initialize an array of values of the full decisivity for the number of models (reference model is element 0)
            expected_lnB = np.zeros(len(self.model_name_list))
            # Initialize an array of values of the expected log Bayes factor for the number of models (reference model is element 0)

        with open(self.path_to_foxi_directory + '/' + self.chains_directory + self.current_data_chains) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the chains
            for line in file:
                columns = line.split()
                fiducial_point_vector = [] 
                for j in range(0,len(chains_column_numbers)):
                    if self.column_types_are_set == True: fiducial_point_vector.append(self.column_functions(j,float(columns[chains_column_numbers[j]])))
                    if self.column_types_are_set == False: fiducial_point_vector.append(float(columns[chains_column_numbers[j]])) # All columns are flat priors unless this is True
                if foxi_mode == 'deci_ExlnB': # Decide on which quantity(ies) to calculate
                    [lnB,deci] = self.decisivity_lnB_utility_functions(fiducial_point_vector,forecast_data_function,prior_column_numbers,number_of_prior_points,error_vector)
                    decisivity = decisivity + deci 
                    expected_lnB = expected_lnB + lnB # Do quick vector additions to update both expected utilities
                running_total+=1 # Also add to the running total
                if running_total >= number_of_points: break # Finish once reached specified number of data points 
        
        if foxi_mode == 'deci_ExlnB':
            expected_lnB /= float(number_of_points)
            decisivity /= float(number_of_points) # Normalise outputs
            output_data_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiout_deci_ExlnB.txt",'w') 
            output_data_file.write(str(self.model_name_list) + ' [<lnB>_1, <lnB>_2, ...] = ' + str(expected_lnB) + "\n")
            output_data_file.write(str(self.model_name_list) + ' [DECI_1, DECI_2, ...] = ' + str(decisivity))
            output_data_file.close()


    def flashy_foxi(self): # Just some front page propaganda...
        print('                                          ')
        print('>>>>>>>>>>>                               ')
        print('>>>       >>                              ')
        print('>>                                        ')
        print('>>                                  >>    ')
        print('>>>>>>>>>    >>>>>>>    >>>    >>>        ') 
        print('>>          >>     >>     >>  >>    >>    ')
        print('>>         >>       >>     >>>>     >>    ')
        print('>>         >>       >>    >>>>      >>    ')
        print('>>          >>     >>    >>  >>     >>    ')
        print('>>           >>>>>>>   >>>    >>>   >>>   ')
        print('                                          ')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('       Author: Robert J. Hardwick         ')
        print('      DISTRIBUTED UNDER MIT LICENSE       ')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('                                          ')


