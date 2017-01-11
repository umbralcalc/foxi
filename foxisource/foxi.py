import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text',usetex=True)
import pylab as pl
import corner

class foxi:
# Initialize the 'foxi' method class
    

    def __init__(self,path_to_foxi_directory):
        self.path_to_foxi_directory = path_to_foxi_directory # Set this upon declaration
        self.current_data_chains = ''
        self.model_name_list = [] 
        self.set_chains
        self.set_model_name_list
        self.utility_functions
        self.run_foxi
        self.plot_foxiplots
        self.chains_directory = 'foxichains/' 
        self.priors_directory = 'foxipriors/'
        self.output_directory = 'foxioutput/'
        self.plots_directory = 'foxiplots/'
        self.source_directory = 'foxisource/'
        # These can be changed above if they are annoying :-) I did get carried away...
        self.gaussian_forecast
        self.flashy_foxi
        self.set_column_types
        self.column_types = []
        self.column_functions 
        self.column_types_are_set = False
        self.add_axes_labels
        self.axes_labels = []
        self.fontsize = 15


    def set_chains(self,name_of_chains): 
    # Set the file name of the chains of the current data set
        self.current_data_chains = name_of_chains


    def set_model_name_list(self,model_name_list): 
    # Set the name list of models which can then be referred to by number for simplicity
        self.model_name_list = model_name_list


    def set_column_types(self,column_types):
    # Input a list of strings to set the format of each column in both prior and data chains i.e. flat, log, log10, ...
        self.column_types = column_types
        self.column_types_are_set = True # All columns are as input unless this is True


    def column_functions(self,i,input_value):
    # Choose to transform whichever column according to its format
        if self.column_types[i] == 'flat': return input_value
        if self.column_types[i] == 'log': return np.exp(input_value)
        if self.column_types[i] == 'log10': return 10.0**(input_value)


    def add_axes_labels(self,list_of_labels,fontsize):
    # Function to add axes LaTeX labels for foxiplots
        self.axes_labels = list_of_labels
        self.fontsize = fontsize


    def utility_functions(self,fiducial_point_vector,chains_column_numbers,prior_column_numbers,forecast_data_function,number_of_points,number_of_prior_points,error_vector):
    # The Kullback-Leibler divergence utility function defined in arxiv:1639.3933
        '''
        DKL_utility_function: 


        fiducial_point_vector                  =  Forecast distribution fiducial central values

        
        chains_column_numbers                  =  A list of the numbers of the columns in the chains that correspond to
                                                  the parameters of interest, starting with 0. These should be in the same
                                                  order as all other structures 


        prior_column_numbers                   =  A 2D list [i][j] of the numbers of the columns j in the prior for model i that 
                                                  correspond to the parameters of interest, starting with 0. These should 
                                                  be in the same order as all other structures      


        forecast_data_function                 =  A function like gaussian_forecast

        
        number_of_points                       =  The number of points specified to be read off from the current data chains

        
        number_of_prior_points                 =  The number of points specified to be read off from the prior


        error_vector                           =  Fixed predicted future measurement errors corresponding to the fiducial_point_vector 


        '''


        decisivity = np.zeros(len(self.model_name_list))
        # Initialize an array of values of the decisivity for the number of models (reference model is element 0)
        E = np.zeros(len(self.model_name_list))
        # Initialize an array of values of the Bayesian evidence for the number of models (reference model is element 0)
        abslnB = np.zeros(len(self.model_name_list))
        # Initialize an array of values of the absolute log Bayes factor for the number of models (reference model is element 0)
        DKL = 0.0
        # Initialise the integral count Kullback-Leibler divergence at 0.0
        forecast_data_function_normalisation = 0.0
        # Initialise the normalisation for the forecast data

        running_total = 0 # Initialize a running total of points read in from the chains

        with open(self.path_to_foxi_directory + '/' + self.chains_directory + self.current_data_chains) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the chains
            for line in file:
                columns = line.split()
                fiducial_point_vector_for_integral = [] 
                for j in range(0,len(chains_column_numbers)):
                    if self.column_types_are_set == True: 
                        fiducial_point_vector_for_integral.append(self.column_functions(j,float(columns[chains_column_numbers[j]])))
                    if self.column_types_are_set == False: 
                        fiducial_point_vector_for_integral.append(float(columns[chains_column_numbers[j]])) # All columns are flat formats unless this is True
                forecast_data_function_normalisation += forecast_data_function(fiducial_point_vector_for_integral,fiducial_point_vector,error_vector)
                running_total+=1 # Also add to the running total                         
                if running_total >= number_of_points: break # Finish once reached specified number points

        for i in range(0,len(self.model_name_list)):
            
            running_total = 0 # Initialize a running total of points read in from each prior

            with open(self.path_to_foxi_directory + '/' + self.priors_directory + 'bayesinf_' + self.model_name_list[i] + '.txt') as file:
            # Compute quantities in loop dynamically as with open(..) reads off the prior values
                for line in file:
                    columns = line.split()
                    prior_point_vector = [] 
                    for j in range(0,len(prior_column_numbers[i])): # Take a prior point 
                        if self.column_types_are_set == True: 
                            prior_point_vector.append(self.column_functions(j,float(columns[prior_column_numbers[i][j]])))
                        if self.column_types_are_set == False: 
                            prior_point_vector.append(float(columns[prior_column_numbers[i][j]])) # All columns are as input unless this is True
                    E[i] += forecast_data_function(prior_point_vector,fiducial_point_vector,error_vector)/float(number_of_prior_points)
                    # Calculate the forecast probability and therefore the contribution to the Bayesian evidence for each model
                    running_total+=1 # Also add to the running total                         
                    if running_total >= number_of_prior_points: break # Finish once reached specified number of prior points

        for j in range(0,len(self.model_name_list)):      
        # Summation over the model priors    
            if E[j] == 0.0:
                if E[0] == 0.0:
                    abslnB[j] = 0.0
                else:
                    abslnB[j] = 1000.0 # Maximal permitted value for an individual sample
            if E[0] == 0.0:
                if E[j] == 0.0:
                    abslnB[j] = 0.0
                else:
                    abslnB[j] = 1000.0 # Maximal permitted value for an individual sample
            else:
                abslnB[j] = abs(np.log(E[j]) - np.log(E[0])) # Compute absolute log Bayes factor utility
            # The block above deals with unruly values like 0.0 inside the logarithms

            if abs(np.log(E[j]) - np.log(E[0])) >= 5.0:
                if j > 0: decisivity[j] = 1.0 # Compute decisivity utility (for each new forecast distribution this is either 1 or 0) 

        running_total = 0 # Initialize a running total of points read in from the chains

        with open(self.path_to_foxi_directory + '/' + self.chains_directory + self.current_data_chains) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the chains
            for line in file:
                columns = line.split()
                fiducial_point_vector_for_integral = [] 
                for j in range(0,len(chains_column_numbers)):
                    if self.column_types_are_set == True: 
                        fiducial_point_vector_for_integral.append(self.column_functions(j,float(columns[chains_column_numbers[j]])))
                    if self.column_types_are_set == False: 
                        fiducial_point_vector_for_integral.append(float(columns[chains_column_numbers[j]])) # All columns are flat formats unless this is True

                logargument = float(number_of_points)*forecast_data_function(fiducial_point_vector_for_integral,fiducial_point_vector,error_vector)/forecast_data_function_normalisation
                dkl = (forecast_data_function(fiducial_point_vector_for_integral,fiducial_point_vector,error_vector)/forecast_data_function_normalisation)*np.log(logargument)
                if np.isnan(dkl) == False and logargument > 0.0: DKL += dkl # Conditions to avoid problems in the logarithm

                running_total+=1 # Also add to the running total                         
                if running_total >= number_of_points: break # Finish once reached specified number points

        return [abslnB,decisivity,DKL,np.log(E)] # Output utilities and raw ln-evidences

   
    def gaussian_forecast(self,prior_point_vector,fiducial_point_vector,error_vector):
    # Posterior probability at some point - prior_point_vector - in arbirary dimensions using a Gaussian 
    # forecast distribution given a fiducial point vector and error vector
        prior_point_vector = np.asarray(prior_point_vector) 
        fiducial_point_vector = np.asarray(fiducial_point_vector)      
        error_vector = np.asarray(error_vector)
        return (1.0/((2.0*np.pi)**(0.5*len(error_vector))))*np.exp(-0.5*(sum(((prior_point_vector-fiducial_point_vector)/error_vector)**2))+(2.0*sum(np.log(error_vector))))


    def run_foxi(self,chains_column_numbers,prior_column_numbers,number_of_points,number_of_prior_points,error_vector,foxiplot_data=False): 
    # This is the main algorithm to compute expected utilites and compute other quantities
        ''' 
        Quick usage and settings:
                 

        chains_column_numbers      =  A list of the numbers of the columns in the chains that correspond to
                                      the parameters of interest, starting with 0. These should be in the same
                                      order as all other structures


        prior_column_numbers       =  A 2D list [i][j] of the numbers of the columns j in the prior for model i that 
                                      correspond to the parameters of interest, starting with 0. These should 
                                      be in the same order as all other structures   


        number_of_points           =  The number of points specified to be read off from the current data chains


        number_of_prior_points     =  The number of points specified to be read off from the prior


        error_vector               =  Fixed predicted future measurement errors corresponding to the fiducial_point_vector


        foxiplot_data              =  Boolean to decide whether or not to output data for plotting the utilities
                                      with the foxiplot tool    


        '''


        self.flashy_foxi() # Display propaganda      

        forecast_data_function = self.gaussian_forecast # Change the forecast data function here (if desired)

        running_total = 0 # Initialize a running total of points read in from the chains

        deci = np.zeros(len(self.model_name_list))
        # Initialize an array of values of the decisivity utility function for the number of models (reference model is element 0)
        abslnB = np.zeros(len(self.model_name_list))
        # Initialize an array of values of the absolute log Bayes factor for the number of models (reference model is element 0)
        decisivity = np.zeros(len(self.model_name_list))
        # Initialize an array of values of the full decisivity for the number of models (reference model is element 0)
        expected_abslnB = np.zeros(len(self.model_name_list))
        # Initialize an array of values of the expected absolute log Bayes factor for the number of models (reference model is element 0)
        expected_DKL = 0.0
        # Initialize the count for expected Kullback-Leibler divergence     

        if foxiplot_data == True:
            plot_data_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_data.txt",'w')
            # Initialize output files if outputting plot data

        with open(self.path_to_foxi_directory + '/' + self.chains_directory + self.current_data_chains) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the chains
            for line in file:
                columns = line.split()
                fiducial_point_vector = [] 
                for j in range(0,len(chains_column_numbers)):
                    if self.column_types_are_set == True: 
                        fiducial_point_vector.append(self.column_functions(j,float(columns[chains_column_numbers[j]])))
                    if self.column_types_are_set == False: 
                        fiducial_point_vector.append(float(columns[chains_column_numbers[j]])) # All columns are flat formats unless this is True
                [abslnB,deci,DKL,lnE] = self.utility_functions(fiducial_point_vector,chains_column_numbers,prior_column_numbers,forecast_data_function,number_of_points,number_of_prior_points,error_vector)
                decisivity = decisivity + deci 
                expected_abslnB = expected_abslnB + abslnB # Do quick vector additions to update both expected utilities
                expected_DKL += DKL
                
                if foxiplot_data == True:
                    for value in fiducial_point_vector:
                        plot_data_file.write(str(value) + "\t")
                    plot_data_file.write(str(running_total) + "\t") # This bit helps identify a gap between output types on a line
                    for value in abslnB:
                        plot_data_file.write(str(value) + "\t")
                    plot_data_file.write(str(running_total) + "\t") # This bit helps identify a gap between output types on a line
                    for value in lnE:
                        plot_data_file.write(str(value) + "\t")
                    plot_data_file.write(str(running_total) + "\t") # This bit helps identify a gap between output types on a line
                    plot_data_file.write(str(DKL) + "\n")
                    # Write data to file if requested                    

                running_total+=1 # Also add to the running total
                if running_total >= number_of_points: break # Finish once reached specified number of data points 
        
        expected_abslnB /= float(number_of_points)
        decisivity /= float(number_of_points) 
        expected_DKL /= float(number_of_points) # Normalise outputs
        output_data_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiout.txt",'w') 
        output_data_file.write(str(self.model_name_list) + ' [<|lnB|>_1, <|lnB|>_2, ...] = ' + str(expected_abslnB) + "\n")
        output_data_file.write(str(self.model_name_list) + ' [DECI_1, DECI_2, ...] = ' + str(decisivity) + "\n")
        output_data_file.write('<DKL> = ' + str(expected_DKL))
        output_data_file.close()

        if foxiplot_data == True:
            plot_data_file.close()    


    def plot_foxiplots(self,filename_choice,column_numbers,ranges,number_of_bins,number_of_samples,label_values):
    # Plot data from files using corner and output to the directory foxioutput/
        ''' 
        Quick usage and settings:
                 
        
        filename_choice            =  Input the name of the file in foxioutput/ to be plotted 
       
 
        column_numbers             =  A list of column numbers (starting at 0) that will be read
                                      in from filename_choice and output using corner plot


        ranges                     =  A tuple [(),(),...] with the plot ranges for all of the
                                      variables that are plotted 


        number_of_bins             =  Integer input for the number of bins for all axes


        number_of_samples          =  Integer number of samples to be plotted from the file


        label_values               =  Label values of interest - a list of these values with
                                      an element for each subplot in corner, otherwise should
                                      have value -> None -> for that element


        '''


        points = []
        # Initialise empty lists for samples

        running_total = 0 # Initialize a running total of points read in from the data file
        
        with open(self.path_to_foxi_directory + "/" + self.output_directory + filename_choice) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the chains
            for line in file:
                columns = line.split()
                list_of_values = []
                for number in column_numbers: 
                    list_of_values.append(float(columns[number]))
                points.append(list_of_values)
                # Read in data points from the requested columns
                running_total+=1 # Also add to the running total
                if running_total >= number_of_samples: break # Finish once reached specified number of data points 

        points = np.asarray(points)
        # Change to arrays for easy manipulation

        figure = corner.corner(points,bins=number_of_bins,color='r',range=ranges,labels=self.axes_labels, label_kwargs={"fontsize": self.fontsize}, plot_contours=False,truths=label_values)
        # Use corner to plot the samples 

        figure.savefig(self.path_to_foxi_directory + "/" + self.plots_directory + "foxiplot.pdf", format='pdf',bbox_inches='tight',pad_inches=0.1,dpi=300)
        # Saving output to foxiplots/


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


