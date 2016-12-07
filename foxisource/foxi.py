import imports

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
        self.foxi_mode = 'null'
        self.run_foxi
        self.chains_directory = 'foxichains/'
        self.priors_directory = 'foxipriors/'
        self.output_directory = 'foxioutput/'
        self.plots_directory = 'foxiplots/'
        self.source_directory = 'foxisource/'   
        self.gaussian_forecast


    def set_chains(self,name_of_chains): 
    # Set the file name of the chains of the current data set
        self.current_data_chains = name_of_chains


    def set_model_name_list(self,model_name_list): 
    # Set the name list of models which can then be referred to by number for simplicity
        self.model_name_list = model_name_list


    def decisivity_lnB_utility_functions(self,fiducial_point_vector,forecast_data_function,prior_column_numbers,number_of_points,error_vector): 
    # The decisivity and lnB utility functions defined in arxiv:1639.3933, but first used in arxiv:1012.3195
    """
    Inputs to decisivity_utility_function:       


    fiducial_point_vector            =  Forecast distribution fiducial central values


    forecast_data_function           =  A function like gaussian_forecast


    prior_column_numbers             =  A list of the numbers of the columns in the prior that correspond to
                                        the parameters of interest, starting with 0. These should be in the same
                                        order as all other structures.


    number_of_points                 =  The number of points specified to be read off from the prior values.


    error_vector                     =  Fixed predicted future measurement errors corresponding to the fiducial_point_vector 


    """
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
                for line in file and running_total < number_of_points:
                    columns = line.split()
                    prior_point_vector = [] 
                    for number in prior_column_numbers: # Take a prior point 
                        prior_point_vector.append(float(columns[number]))
                    E[i] += forecast_data_function(prior_point_vector,fiducial_point_vector,error_vector) 
                    # Calculate the forecast probability and therefore the contribution to the Bayesian evidence for each model
                         
        for j in range(0,len(self.model_name_list)-1):
            lnB[j] = np.log10(E[j]) - np.log10(E[0]) # Compute log Bayes factor utility
            if np.log10(E[j]) - np.log10(E[0]) <= -5.0:
                if j > 0: decisivity[j] = 1.0 # Compute decisivity utility (for each new forecast distribution this is either 1 or 0) 
 
        return [lnB,decisivity] # Output both utilities        


    def jensen_shannon_div_utility_function(self): 
    # A utility based on the total Jensen-Shannon divergence introduced in arxiv:1606.06758
        return 1.0

   
    def gaussian_forecast(self,prior_point_vector,fiducial_point_vector,error_vector):
    # Posterior probability at some point - prior_point_vector - in arbirary dimensions using a Gaussian 
    # forecast distribution given a fiducial point vector and error vector
        prior_point_vector = np.asarray(prior_point_vector) 
        fiducial_point_vector = np.ararray(fiducial_point_vector)      
        error_vector = np.asarray(error_vector)
        return (1.0/np.sqrt(2.0*np.pi))*np.exp(-(0.5*sum(((prior_point_vector-fiducial_point_vector)/error_vector)**2))-sum(np.log(error_vector)))


    def run_foxi(self,foxi_mode='deci_ExlnB',chains_column_numbers,prior_column_numbers,number_of_points,number_of_prior_points,error_vector): 
    # This is the main algorithm to compute expected utilites and compute other quantities
        ''' 
        Quick usage and settings:

        foxi_mode                   = 'Exjsd'             - Sets up foxi to compute just the total expected
                                                            Jensen-Shannon divergence between all models in 
                                                            model_name_list.
                                      'deci_ExlnB'        - Sets up foxi to compute both the decisivity and 
                                                            expected log Bayes factor utilities where we
                                                            set model_name_list[0] as the reference model.
                                      'deci_ExlnB_Exjsd'  - Sets up foxi to compute 'deci_ExlnB' and 'Exjsd'.

                                      < Default >         - This is set to 'deci_ExlnB'. 
                 

        chains_column_numbers      =  A list of the numbers of the columns in the chains that correspond to
                                      the parameters of interest, starting with 0. These should be in the same
                                      order as all other structures.

       
        number_of_points           =  The number of points specified to be read off from the current data chains.


        number_of_prior_points     =  The number of points specified to be read off from the prior.


        error_vector               =  Fixed predicted future measurement errors corresponding to the fiducial_point_vector.


        '''
        self.flashy_foxi() # Display propaganda      

        forecast_data_function = gaussian_forecast # Change the forecast data function here (if desired)

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
            for line in file and running_total < number_of_points:
            # Continue for as long as needed
                columns = line.split()
                fiducial_point_vector = [] 
                for number in chains_column_numbers:
                    fiducial_point_vector.append(float(columns[number]))
                if foxi_mode == 'deci_ExlnB': # Decide on which quantity(ies) to calculate
                    [lnB,deci] = self.decisivity_lnB_utility_functions(fiducial_point_vector,forecast_data_function,prior_column_numbers,number_of_prior_points,error_vector)
                    decisivity = decisivity + deci 
                    expected_lnB = expected_lnB + lnB # Do quick vector additions to update both expected utilities
        
        if foxi_mode == 'deci_ExlnB':
            output_data_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiout_deci_ExlnB.txt",'w') 
            output_data_file.write(str(self.model_name_list) + ' [<lnB>_1, <lnB>_2, ...] = ' + str(expected_lnB) + "\n")
            output_data_file.write(str(self.model_name_list) + ' [DECI_1, DECI_2, ...] = ' + str(decisivity))
            output_data_file.close()


    def flashy_foxi(): # Just some front page propaganda...
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


