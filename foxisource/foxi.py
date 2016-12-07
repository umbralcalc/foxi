import imports

class foxi:
# Initialize the 'foxi' method class
    

    def __init__(self,path_to_foxi_directory):
        self.path_to_foxi_directory = path_to_foxi_directory # Set this upon declaration
        self.current_data_chains = ''
        self.model_name_list = [] 
        self.set_chains
        self.set_model_name_list
        self.differential_decisivity_utility_function
        self.differential_expected_lnB_utility_function
        self.differential_jensen_shannon_div_utility_function
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


    def differential_decisivity_utility_function(self): 
    # Differential contribution from the decisivity utility function defined in arxiv:1639.3933, but first used in arxiv:1012.3195


    def differential_expected_lnB_utility_function(self): 
    # Differential contribution from the expected ln Bayes factor utility function defined in arxiv:1012.3195 and arxiv:1639.3933 


    def differential_jensen_shannon_div_utility_function(self): 
    # Differential contribution from a utility based on the total Jensen-Shannon divergence introduced in arxiv:1606.06758

   
    def gaussian_forecast(self,x_vector,fiducial_point_vector,error_vector):
    # Posterior probability at some point x in arbirary dimensions using a Gaussian 
    # forecast distribution given a fiducial point vector and error vector
        return (1.0/np.sqrt(2.0*np.pi))*np.exp(-(0.5*sum(((x_vector-fiducial_point_vector)/error_vector)**2))-sum(np.log(error_vector)))


    def run_foxi(self,foxi_mode='deci',chains_column_numbers,number_of_points,fiducial_widths): 
    # This is the main algorithm to compute expected utilites and compute other quantities
        ''' 
        Quick usage and settings:

        foxi_mode                   = 'deci'            - Sets up foxi to compute just the decisivity 
                                                          between all models in model_name_list, setting 
                                                          model_name_list[0] as the reference model.
                                      'ExlnB'           - Sets up foxi to compute just the expected ln
                                                          Bayes factor between all models in model_name_list, 
                                                          setting model_name_list[0] as the reference model.
                                      'jsd'             - Sets up foxi to compute just the total Jensen-
                                                          Shannon divergence between all models in 
                                                          model_name_list.
                                      'deci_ExlnB'      - Sets up foxi to compute both 'deci' and 'ExlnB'.
                                      'deci_ExlnB_jsd'  - Sets up foxi to compute 'deci', 'ExlnB' and 'jsd'.

                                      < Default >       - This is set to 'deci' 
                 

        chains_column_numbers      =  A list of the numbers of the columns in the chains that correspond to
                                      the parameters of interest, starting with 0. These should be in the same
                                      order as all other structures.

        '''
        running_total = 0 # Initialize a running total of points read in from the chains
        with open(self.path_to_foxi_directory + '/' + self.chains_directory + self.current_data_chains) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the chains
            for line in file and running_total < number_of_points:
            # Continue for as long as needed
                columns = line.split()
                fiducial_point_vector = [] 
                for number in chains_column_numbers:
                    fiducial_point_vector.append(float(columns[number]))



    def flashy_foxi(): # Just some front page propaganda...
        print(' >>>>>>>>>>                               ')
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


