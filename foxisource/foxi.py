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
        self.utility_functions
        self.run_foxi
        self.run_foxiplots
        self.plot_foxiplots
        self.chains_directory = 'foxichains/'
        self.priors_directory = 'foxipriors/'
        self.output_directory = 'foxioutput/'
        self.plots_directory = 'foxiplots/'
        self.source_directory = 'foxisource/'   
        self.gaussian_forecast
        self.flashy_foxi
        self.flashy_foxiplots
        self.set_column_types
        self.column_types = []
        self.column_functions 
        self.column_types_are_set = False
        self.add_axes_labels
        self.axes = plt.gca()


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


    def add_axes_labels(self, paramname_tex_x, paramname_tex_y, fontsize):
    # Function to add axes LaTeX labels for foxiplots
        self.axes.set_xlabel(paramname_tex_x,fontsize=float(fontsize))
        self.axes.set_ylabel(paramname_tex_y,fontsize=float(fontsize))


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

        return [abslnB,decisivity,DKL] # Output utilities

   
    def gaussian_forecast(self,prior_point_vector,fiducial_point_vector,error_vector):
    # Posterior probability at some point - prior_point_vector - in arbirary dimensions using a Gaussian 
    # forecast distribution given a fiducial point vector and error vector
        prior_point_vector = np.asarray(prior_point_vector) 
        fiducial_point_vector = np.asarray(fiducial_point_vector)      
        error_vector = np.asarray(error_vector)
        return (1.0/((2.0*np.pi)**(0.5*len(error_vector))))*np.exp(-0.5*(sum(((prior_point_vector-fiducial_point_vector)/error_vector)**2))+(2.0*sum(np.log(error_vector))))


    def run_foxi(self,chains_column_numbers,prior_column_numbers,number_of_points,number_of_prior_points,error_vector): 
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
                [abslnB,deci,DKL] = self.utility_functions(fiducial_point_vector,chains_column_numbers,prior_column_numbers,forecast_data_function,number_of_points,number_of_prior_points,error_vector)
                decisivity = decisivity + deci 
                expected_abslnB = expected_abslnB + abslnB # Do quick vector additions to update both expected utilities
                expected_DKL += DKL
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


    def run_foxiplots(self,chains_column_numbers,prior_column_numbers,number_of_points,number_of_prior_points,error_vector,xy_column_numbers,xmin,xmax,ymin,ymax,kind_of_interpolation): 
    # This is the plot data outputting tool for the utilites
        ''' 
        Quick usage and settings:
                 

        chains_column_numbers      =  A list of the numbers of the columns in the chains that correspond to
                                      the parameters of interest, starting with 0. These should be in the same
                                      order as all other structures


        prior_column_numbers       =  A 2D list [i][j] of the numbers of the columns j in the prior for model i that 
                                      correspond to the parameters of interest, starting with 0. These should 
                                      be in the same order as all other structures. In run_foxiplots one should
                                      only have two models since plotting is computationally expensive, so i = 0,1
                                      where 0 still corresponds to the reference model


        number_of_points           =  The number of points specified to be read off from the current data chains


        number_of_prior_points     =  The number of points specified to be read off from the prior


        error_vector               =  Fixed predicted future measurement errors corresponding to the fiducial_point_vector

        
        xy_column_numbers          =  A 2-element list with the column numbers of the (x,y) values to plot. Element 0 => x
                                      and element 1 => y


        xmin, xmax                 =  Limits set on the x-axis for the plot data

        
        ymin, ymax                 =  Limits set on the y-axis for the plot data


        kind_of_interpolation      =  Either 'nn' or 'linear' for nearest-neighbour or linear interpolation schemes        


        '''

        self.flashy_foxiplots() # Display propaganda      

        forecast_data_function = self.gaussian_forecast # Change the forecast data function here (if desired)

        running_total = 0 # Initialize a running total of points read in from the chains

        deci = np.zeros(len(self.model_name_list))
        # Initialize an array of values of the decisivity utility function for the number of models (reference model is element 0)
        abslnB = np.zeros(len(self.model_name_list))
        # Initialize an array of values of the absolute log Bayes factor for the number of models (reference model is element 0)
        
        DKL_values = []
        abslnB_values = []
        x_values = []
        y_values = []
        # Initialise lists of values for the utilities and (x,y) values to be dumped    

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
                [abslnB,deci,DKL] = self.utility_functions(fiducial_point_vector,chains_column_numbers,prior_column_numbers,forecast_data_function,number_of_points,number_of_prior_points,error_vector)
                abslnB_values.append(abslnB[1])  
                DKL_values.append(DKL) # Add computed utilities to the lists
                x_values.append(fiducial_point_vector[xy_column_numbers[0]])
                y_values.append(fiducial_point_vector[xy_column_numbers[1]]) # Add corresponding (x,y) values to lists
                running_total+=1 # Also add to the running total
                if running_total >= number_of_points: break # Finish once reached specified number of data points 
   
        x_plot_values = pl.linspace(xmin,xmax, 100) 
        y_plot_values = pl.linspace(ymin,ymax, 100)
        # Generate a range of (x,y) plot values         

        z_plot_values_abslnB = mpl.mlab.griddata(x_values,y_values,abslnB_values,x_plot_values,y_plot_values,interp=kind_of_interpolation)
        z_plot_values_DKL = mpl.mlab.griddata(x_values,y_values,DKL_values,x_plot_values,y_plot_values,interp=kind_of_interpolation)
        # Interpolate to obtain plot values for output

        plot_data_file_abslnB = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_abslnB_data.txt",'w')
        plot_data_file_DKL = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_DKL_data.txt",'w') 

        for i in range(0,len(x_plot_values)):
            for j in range(0,len(y_plot_values)):
                plot_data_file_abslnB.write(str(x_plot_values[i]) + "\t" + str(y_plot_values[j]) + "\t" + str(z_plot_values_abslnB[i,j]) + "\n")
                plot_data_file_DKL.write(str(x_plot_values[i]) + "\t" + str(y_plot_values[j]) + "\t" + str(z_plot_values_DKL[i,j]) + "\n")
                # Output plot data to the files

        plot_data_file_abslnB.close()
        plot_data_file_DKL.close()


    def plot_foxiplots(self,filename_choice,xmin,xmax,ymin,ymax,zmin,zmax):
    # Plot data from files in the directory foxioutput/
        ''' 
        Quick usage and settings:
                 
        
        filename_choice            =  Input the name of the file in foxioutput/ to be plotted 


        xmin, xmax                 =  Limits set on the x-axis for the plot data

        
        ymin, ymax                 =  Limits set on the y-axis for the plot data


        zmin, zmax                 =  Limits set on the z-axis for the plot data        


        '''
        x_values = []
        y_values = []
        z_values = pl.zeros([100, 100])
        # Initialise lists of (x,y) and a 2D array of z values  

        x_before = 0.0
        y_before = 0.0 
        i = 0
        j = 0     

        with open(self.path_to_foxi_directory + "/" + self.output_directory + filename_choice) as file:
            for line in file:
                digits = line.split()
                z_values[j,i] = float(digits[2])
                if x_before != float(digits[0]):
                    if len(x_values) < 100: x_values.append(float(digits[0]))
                    if len(x_values) != 1: i += 1
                    if i > 99: i = 0
                    # Iterate on loop in x values
                x_before = float(digits[0])
                if y_before != float(digits[1]):
                    if len(y_values) < 100: y_values.append(float(digits[1]))
                    if len(y_values) != 1: j += 1
                    if j > 99: j = 0
                    # Iterate on alternate loop in y values
                y_before = float(digits[1])  
        # Reading in a list of values into a 2D array of values: z_values
        
        pl.xlim(xmin,xmax)
        pl.ylim(ymin,ymax)
        # Add in extra axes limits if required

        plot_fig = pl.pcolormesh(x_values, y_values, z_values, vmin=zmin, vmax=zmax, cmap='BuPu')
        pl.colorbar()
        # Plot the values

        pl.savefig(self.path_to_foxi_directory + "/" + self.plots_directory + "foxiplot.pdf", format='pdf',bbox_inches='tight',pad_inches=0.1,dpi=300)
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


    def flashy_foxiplots(self): # Just some more front page propaganda...
        print('                                                                                          ')
        print('>>>>>>>>>>>                                                                               ')
        print('>>>       >>                                       >>                   >>                ')
        print('>>                                                 >>                   >>                ')
        print('>>                                  >>             >>                   >>                ')
        print('>>>>>>>>>    >>>>>>>    >>>    >>>       >>>>>>>>  >>       >>>>>>>     >>       >>>>>>   ') 
        print('>>          >>     >>     >>  >>    >>   >>    >>  >>      >>     >>    >>>>>>   >>       ')
        print('>>         >>       >>     >>>>     >>   >>    >>  >>     >>       >>   >>        >>      ')
        print('>>         >>       >>    >>>>      >>   >>    >>  >>     >>       >>   >>          >>    ')
        print('>>          >>     >>    >>  >>     >>   >>>>>>    >>      >>     >>    >>           >>   ')
        print('>>           >>>>>>>   >>>    >>>   >>>  >>         >>>>>>  >>>>>>>      >>>>>>  >>>>>    ')
        print('                                         >>                                               ')
        print('                                         >>                                               ')
        print('                                         >>                                               ')
        print('                                                                                          ')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('                              Author: Robert J. Hardwick                                  ')
        print('                             DISTRIBUTED UNDER MIT LICENSE                                ')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('                                                                                          ')


