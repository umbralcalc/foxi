import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text',usetex=True)
rc('text.latex',preamble=r'\usepackage{mathrsfs}')
import pylab as pl
import matplotlib.cm as cm
import corner
import math 
import scipy.stats as sps
from statsmodels.nonparametric.kernel_density import KDEMultivariate as kde


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
        self.rerun_foxi
        self.run_foxifish
        self.analyse_foxiplot_line
        self.plot_foxiplots
        self.chains_directory = 'foxichains/' 
        self.priors_directory = 'foxipriors/'
        self.output_directory = 'foxioutput/'
        self.plots_directory = 'foxiplots/'
        self.source_directory = 'foxisource/'
        # These can be changed above if they are annoying :-) I did get carried away...
        self.gaussian_forecast
        self.set_fisher_matrix
        self.fisher_matrix_forecast
        self.flashy_foxi
        self.column_types = []
        self.prior_column_types = []
        self.chains_data = [] 
        self.chains_weights = []
        self.column_functions 
        self.column_types_are_set = False
        self.prior_column_types_are_set = False
        self.chains_weights_are_set = False
        self.add_axes_labels
        self.axes_labels = []
        self.fontsize = 15
        self.decide_on_best_computation
        self.decide_on_best_computation_with_fisher
        self.number_of_forecast_samples = 1000 # Number of samples to take from the forecast distribution 
        self.flat_function
        self.exp_function
        self.power10_function
        self.analytic_estimates
        self.FPS_derivative
        self.dictionary_of_column_types = {'flat': self.flat_function, 'log': self.exp_function, 'log10': self.power10_function}    


    def set_chains(self,name_of_chains,chains_column_numbers,number_of_points,weights_column=None,column_types=None): 
    # Set the file name of the chains of the current data set
        self.current_data_chains = name_of_chains

        self.chains_column_numbers = chains_column_numbers
        # A list of the numbers of the columns in the chains that correspond to the parameters of interest, starting with 0. These should be in the same order as all other structures

        self.number_of_points = number_of_points # The number of points specified to be read off from the current data chains
  
        if weights_column: 
            self.current_data_chains_weights_column = weights_column
            self.chains_weights_are_set = True
        # Set a number (starting at 0) for the column of the weights in the chains (if there is one)

        if column_types:
            self.column_types = column_types # Input a list of strings to set the format of each column in data chains i.e. flat, log, log10, ...
            self.column_types_are_set = True # All columns are as input unless this is True

        running_total = 0 # Initialize a running total of points read in from the chains

        with open(self.path_to_foxi_directory + '/' + self.chains_directory + self.current_data_chains) as file:
        # Loop dynamically with open(..) reads off the chains
            for line in file:
                columns = line.split()
                columns = np.asarray(columns).astype(np.float)              

                if self.chains_weights_are_set == True: 
                    self.chains_weights.append(columns[self.current_data_chains_weights_column])
                else:
                    self.chains_weights.append(1.0)
                # Set the chain weights either to the alloted column or to the default of 1.0 at each point

                if self.column_types_are_set == True: 
                    fiducial_point_vector = self.column_functions(np.asarray(columns[chains_column_numbers]))
                else: 
                    fiducial_point_vector = np.asarray(columns[chains_column_numbers]) # All columns are flat formats unless this is True

                self.chains_data.append(fiducial_point_vector)

                running_total+=1 # Also add to the running total
                if running_total >= number_of_points: break # Finish once reached specified number of data points

    
    def flat_function(self,x):
        return x

    
    def exp_function(self,x):
        return np.exp(x)


    def power10_function(self,x):
        return 10.0**(x)


    def set_model_name_list(self,model_name_list,prior_column_numbers,number_of_prior_points,prior_column_types=None): 
    # Set the name list of models which can then be referred to by number for simplicity
        self.model_name_list = model_name_list
        # Input the .txt or .dat file names in the foxipriors/ directory
        self.number_of_category_A_points = [0 for i in model_name_list]
        self.number_of_category_B_points = [0 for i in model_name_list]
        self.number_of_category_C_points = [0 for i in model_name_list]
        self.number_of_category_D_points = [0 for i in model_name_list]
        # Initialise a count for each model in their category of points (see arxiv:1803.09491)
        self.number_of_maxed_evidences = [0 for i in model_name_list]
        # Initialise the count for the number of maxed evidences - the number of evidences in each model point for which
        # the numerical precision is reached at abslnB >= 1000 
        self.prior_data = [[] for i in model_name_list]
        # Define prior data storage

        self.prior_column_numbers = prior_column_numbers
        # A 2D list [i][j] of the numbers of the columns j in the prior for model i that correspond to the parameters of interest, starting with 0. These should be in the same order as all other structures 

        self.number_of_prior_points = number_of_prior_points # The number of points specified to be read off from the prior

        if prior_column_types:
            self.prior_column_types = prior_column_types # Input a list of strings to set the format of each column in prior samples i.e. flat, log, log10, ...
            self.prior_column_types_are_set = True # All columns are as input unless this is True

        self.density_functions = []
        # Initialise an empty list of density functions that may or may not be used

        for i in range(0,len(self.model_name_list)):  
         
            running_total = 0 # Initialize a running total of points read in from each prior      

            with open(self.path_to_foxi_directory + '/' + self.priors_directory + self.model_name_list[i]) as file:
            # Compute quantities in loop dynamically as with open(..) reads off the prior values
                for line in file:
                    columns = line.split()
                    columns = np.asarray(columns).astype(np.float)

                    if self.prior_column_types_are_set == True: 
                        prior_point_vector = self.column_functions(np.asarray(columns[prior_column_numbers[i]]),prior=True)
                    else: 
                        prior_point_vector = np.asarray(columns[prior_column_numbers[i]]) # All columns are as input unless this is True
                    
                    self.prior_data[i].append(np.asarray(prior_point_vector)) # Store the data for the KDE
                                      
                    # Calculate the forecast probability and therefore the contribution to the Bayesian evidence for each model
                    running_total+=1 # Also add to the running total                         
                    if running_total >= number_of_prior_points: break # Finish once reached specified number of prior points

            c_string = ''
            for j in range(0,len(prior_column_numbers[i])): c_string += 'c'
            # Silly notational detail to get the kde to work for continuous variables in all dimensions    

            self.density_functions.append(kde(self.prior_data[i],var_type=c_string)) # Append a new Density Estimation function to the list for use later
            # Kernel Density Estimation uses statsmodels.nonparametric toolkit to estimate the 
            # density locally at the fiducial point itself, ensuring that there are no lost points
            # within each prior volume 'hull'

        
        # When using KDE method we quote the bandwidths used for each dimension for smoothing
        print('Using the statsmodels module: http://www.statsmodels.org/stable/index.html')
        print('The Kernel Density Bandwidth for each model listed in each dimension:' + '\n')
        for i in range(0,len(self.model_name_list)):
             print('Model ' + str(i) + ': ' + str(self.density_functions[i].bw))


    def set_number_of_forecast_samples(self,number_of_forecast_samples):
    # Change the number of samples to take from the forecast likelihood distribution when computing the evidence integrals
        self.number_of_forecast_samples = number_of_forecast_samples

    
    def set_fisher_matrix(self,fisher_matrix_function):
    # Set a function of the fiducial point vector which returns the forecast Fisher matrix for the future likelihood
        self.fisher_matrix_function = fisher_matrix_function


    def column_functions(self,input_value,prior=False):
    # Choose to transform whichever column according to its format
        if prior == False:
            return [self.dictionary_of_column_types[self.column_types[i]](input_value[i]) for i in range(0,len(self.column_types))]
        else:
            return [self.dictionary_of_column_types[self.prior_column_types[i]](input_value[i]) for i in range(0,len(self.prior_column_types))]


    def add_axes_labels(self,list_of_labels,fontsize):
    # Function to add axes LaTeX labels for foxiplots
        self.axes_labels = list_of_labels
        self.fontsize = fontsize


    def analyse_foxiplot_line(self,line_1,as_a_function=False):
    # Reads first line from a foxiplot data file and outputs the number of models and dimension of data (this is just to check the input)
        dim_data = int(line_1[0])
        num_models = int(line_1[1])
        self.mix_models_was_used = int(line_1[2])
        
        if as_a_function == False:
            if self.mix_models_was_used == False: 
                print("Just read in an ordinary foxiplot data file with data dimension " + str(dim_data) + " and the number of model pairs is " + str(num_models))
            if self.mix_models_was_used == True:
                print("Just read in a `mix_models' foxiplot data file with data dimension " + str(dim_data) + " and the number of model pairs is " + str(num_models))

        return [dim_data,num_models]


    def FPS_derivative(self,order,stepsize,f_lower2,f_lower1,f_centre,f_upper1,f_upper2):
    # Numerically evaluates the (first or second) derivative in a Five Point Stencil (FPS) using the coefficients provided by: http://web.media.mit.edu/~crtaylor/calculator.html
        if order == 1: return (f_lower2 - (8.0*f_lower1) + (8.0*f_upper1) - f_upper2)/(12.0*stepsize)
        if order == 2: return (-f_lower2 + (16.0*f_lower1) - (30.0*f_centre) + (16.0*f_upper1) - f_upper2)/(12.0*(stepsize**2.0))


    def analytic_estimates(self,toy_model_space,chains_best_fit_point,chains_fisher_matrix,stepsizes):
    # Additional function which estimates (using the Fisher matrix function introduced with 'set_fisher_matrix') and outputs the expected 
    # Kullback-Liebler divergence and total sum over expected log-Bayes factors in the model space specified by the user at input... 
    # Run this separately from the main foxi algorithm, and note that it takes far less time to complete so can be a useful
    # check on the accuracy of foxi computations
        
        '''
        Inputs:


        toy_model_space                =    User-specified model space - a collection of points with coordinates given in a list [[dim1,dim2,..],[dim1,dim2,..],...] which 
                                            corresponds to a collection of models represented with dirac delta function priors. This is useful to get an approximate picture
                                            of how a deformation to the space of models can change the expected ln Bayes factor utility


        chains_best_fit_point          =    User-specified point in a list [dim1, dim2, ...] representing the best fit value from the data chains


        chains_fisher_matrix           =    User-specified square matrix given in list form [[dim1,dim2,..],[dim1,dim2,..],...] representing the approximate Fisher matrix
                                            of the current posterior. This should be readily obtained from a decent MCMC algorithm from which the chains were obtained

        
        stepsizes                      =    A list of stepsizes in each dimension with which to numerically evaluate the derivative of the forecast Fisher matrix


        '''

        toy_model_space = np.asarray(toy_model_space) 
        chains_best_fit_point = np.asarray(chains_best_fit_point)
        chains_fisher_matrix = np.asarray(chains_fisher_matrix)
        # Convert to arrays for quicker computation

        dimension_number = len(chains_fisher_matrix)
        # Find the number of dimensions

        forecast_fisher_centre = self.fisher_matrix_function(chains_best_fit_point)  
        # Calculate the Fisher matrix forecast at the best fit point of the chains       

        forecast_fisher_first_derivs = []
        forecast_fisher_first_derivs_function_values = []
        forecast_fisher_second_derivs = []
        forecast_fisher_cross_second_derivs = [[np.zeros((dimension_number,dimension_number)) for i in range(0,dimension_number)] for j in range(0,dimension_number)]
        # Initialise lists of derivatives
        
        stepsizes_list = []
        for i in range(0,dimension_number):
            stepsize_vector = np.zeros(dimension_number)
            stepsize_vector[i] = stepsizes[i]
            stepsizes_list.append(stepsize_vector)
        # Create list of stepsize vectors for quick addition

        for i in range(0,dimension_number):
            f_lower2 = self.fisher_matrix_function(chains_best_fit_point-(2.0*stepsizes_list[i])) 
            f_lower1 = self.fisher_matrix_function(chains_best_fit_point-(1.0*stepsizes_list[i]))
            f_upper1 = self.fisher_matrix_function(chains_best_fit_point+(1.0*stepsizes_list[i]))
            f_upper2 = self.fisher_matrix_function(chains_best_fit_point+(2.0*stepsizes_list[i]))
            # Loop over the number of dimensions and compute the forecast Fisher matrices of the stencil

            forecast_fisher_first_derivs_function_values.append([f_lower2,f_lower1,f_upper1,f_upper2])
            # Store the values in order to calculate the cross-derivatives later 

            forecast_fisher_first_derivs.append(self.FPS_derivative(1,stepsizes[i],f_lower2,f_lower1,forecast_fisher_centre,f_upper1,f_upper2))
            forecast_fisher_second_derivs.append(self.FPS_derivative(2,stepsizes[i],f_lower2,f_lower1,forecast_fisher_centre,f_upper1,f_upper2)) 
            # Set the value of the first and second derivative matrices - the latter corresponding to the D_iD_i derivative           

        for i in range(0,dimension_number):
            for j in range(i+1,dimension_number):  
            # Use a scheme that recomputes the minimum number of components for the cross-derivatives
                f_lower_lower = self.fisher_matrix_function(chains_best_fit_point-stepsizes_list[i]-stepsizes_list[j]) 
                f_upper_upper = self.fisher_matrix_function(chains_best_fit_point+stepsizes_list[i]+stepsizes_list[j])
                cross_derivative = (f_upper_upper+(2.0*forecast_fisher_centre)+f_lower_lower-forecast_fisher_first_derivs_function_values[i][1]-forecast_fisher_first_derivs_function_values[i][2]-forecast_fisher_first_derivs_function_values[j][1]-forecast_fisher_first_derivs_function_values[j][2])/(2.0*stepsizes[i]*stepsizes[j]) 
                forecast_fisher_cross_second_derivs[i][j] = cross_derivative 
                forecast_fisher_cross_second_derivs[j][i] = cross_derivative  
                # Add in the off-diagonal components of the derivative matrix                 

        for i in range(0,dimension_number): forecast_fisher_cross_second_derivs[i][i] = forecast_fisher_second_derivs[i]
        # Add in the diagonal component of the derivative matrix

        DKL_centre = 0.5*(np.log(np.linalg.det(np.linalg.inv(chains_fisher_matrix)*(chains_fisher_matrix+forecast_fisher_centre))) - dimension_number + np.trace(chains_fisher_matrix*np.linalg.inv(chains_fisher_matrix+forecast_fisher_centre))) 
        SUMexlnB_centre = 0.0
        exlnB_centre = [0.0 for npairs in range(0,len(toy_model_space)*(len(toy_model_space)-1)/2 + len(toy_model_space))]
        add_on = 0
        for na in range(0,len(toy_model_space)): 
            for nb in range(na,len(toy_model_space)):
                lnB_centre = np.abs(0.5*(np.dot((toy_model_space[nb]-chains_best_fit_point),np.dot(forecast_fisher_centre,(toy_model_space[nb]-chains_best_fit_point))) - np.dot((toy_model_space[na]-chains_best_fit_point),np.dot(forecast_fisher_centre,(toy_model_space[na]-chains_best_fit_point)))))
                SUMexlnB_centre += lnB_centre
                exlnB_centre[nb-na+add_on] = lnB_centre
            add_on += len(toy_model_space)-na
        # Evaluate the utilities at the central values - this includes summing over the expected Bayes factors all possible model pairs specified in the toy model space

        exDKLest = DKL_centre 
        SUMexlnBest = SUMexlnB_centre
        exDKLest_som = 0.0 
        SUMexlnBest_som = 0.0
        exlnBest = [exlnB_centre[npairs] for npairs in range(0,len(exlnB_centre))]
        exlnBest_som = [0.0 for npairs in range(0,len(exlnB_centre))]
        # Initialise sums for the estimated expected utilities with their central values and the second centred-moments (som) to zero

        for i in range(0,dimension_number):
            for j in range(0,dimension_number):
            # Compute the expected utilty estimates with the fisher matrices
            # As above, this is a super-slow method of looping - more efficient to use tensor contractions in python but 
            # will leave this to a later update in the case where the number of dimensions is very large
                traceterm = np.trace((np.linalg.inv(chains_fisher_matrix+forecast_fisher_centre)*forecast_fisher_cross_second_derivs[i][j]) - ((np.linalg.inv(chains_fisher_matrix+forecast_fisher_centre)**2.0)*(forecast_fisher_first_derivs[i]*forecast_fisher_first_derivs[j])) +((2.0*chains_fisher_matrix)*(np.linalg.inv(chains_fisher_matrix+forecast_fisher_centre)**3.0)*(forecast_fisher_first_derivs[i]*forecast_fisher_first_derivs[j])) - (chains_fisher_matrix*(np.linalg.inv(chains_fisher_matrix+forecast_fisher_centre)**2.0)*forecast_fisher_cross_second_derivs[i][j])) + (np.linalg.inv(chains_fisher_matrix)*(np.linalg.inv(np.linalg.inv(chains_fisher_matrix)+ np.linalg.inv(forecast_fisher_centre))**2.0))[i][j]
                traceterm_som = np.trace(((np.linalg.inv(chains_fisher_matrix+forecast_fisher_centre)*forecast_fisher_first_derivs[i]) - (chains_fisher_matrix*(np.linalg.inv(chains_fisher_matrix+forecast_fisher_centre)**2.0)*forecast_fisher_first_derivs[i])))*np.trace(((np.linalg.inv(chains_fisher_matrix+forecast_fisher_centre)*forecast_fisher_first_derivs[j]) - (chains_fisher_matrix*(np.linalg.inv(chains_fisher_matrix+forecast_fisher_centre)**2.0)*forecast_fisher_first_derivs[j])))
                exDKLest += 0.25*np.linalg.inv(chains_fisher_matrix)[i][j]*traceterm
                exDKLest_som += 0.25*np.linalg.inv(chains_fisher_matrix)[i][j]*traceterm_som
                # Compute the estimated expected Kullback-Liebler divergence ij-component and then add it to the total contribution

                SUMexlnBestij = 0.0
                SUMexlnBestij_som = 0.0
                add_on = 0
                for na in range(0,len(toy_model_space)): 
                    for nb in range(na,len(toy_model_space)):
                        lnBestij = np.abs((4.0*np.dot(forecast_fisher_first_derivs[i],(toy_model_space[na]-toy_model_space[nb])))[j]+np.dot((toy_model_space[nb]-chains_best_fit_point),np.dot(forecast_fisher_cross_second_derivs[i][j],(toy_model_space[nb]-chains_best_fit_point)))-np.dot((toy_model_space[na]-chains_best_fit_point),np.dot(forecast_fisher_cross_second_derivs[i][j],(toy_model_space[na]-chains_best_fit_point))))
                        lnBestij_som = np.abs(((2.0*np.dot(forecast_fisher_centre,toy_model_space[na]-toy_model_space[nb])[i])+(np.dot(toy_model_space[nb]-chains_best_fit_point,np.dot(forecast_fisher_first_derivs[i],toy_model_space[nb]-chains_best_fit_point)))-(np.dot(toy_model_space[na]-chains_best_fit_point,np.dot(forecast_fisher_first_derivs[i],toy_model_space[na]-chains_best_fit_point))))*((2.0*np.dot(forecast_fisher_centre,toy_model_space[na]-toy_model_space[nb])[j])+(np.dot(toy_model_space[nb]-chains_best_fit_point,np.dot(forecast_fisher_first_derivs[j],toy_model_space[nb]-chains_best_fit_point)))-(np.dot(toy_model_space[na]-chains_best_fit_point,np.dot(forecast_fisher_first_derivs[j],toy_model_space[na]-chains_best_fit_point)))))    
                # Compute the estimated expected ln Bayes factor (and its second moment contribution) averaged over the whole model space ij-component and then add it to the total contribution
                        exlnBest[nb-na+add_on] += 0.25*np.linalg.inv(chains_fisher_matrix)[i][j]*lnBestij
                        exlnBest_som[nb-na+add_on] += 0.25*np.linalg.inv(chains_fisher_matrix)[i][j]*lnBestij_som
                        SUMexlnBestij += lnBestij
                        SUMexlnBestij_som += lnBestij_som
                    add_on += len(toy_model_space)-na
                
                SUMexlnBest += 0.25*np.linalg.inv(chains_fisher_matrix)[i][j]*SUMexlnBestij
                SUMexlnBest_som += 0.25*np.linalg.inv(chains_fisher_matrix)[i][j]*SUMexlnBestij_som
                # Add this ij-contribution to the total   

        print('<DKL> estimate: ' + str(exDKLest) + ' + O(F^{-2}) terms') 
        print('< (DKL-<DKL>)^2 > estimate: ' + str(exDKLest_som) + ' + O(F^{-2}) terms') 
        print('Sum_i <|lnB|>_i estimate: ' + str(SUMexlnBest) + ' + O(F^{-2}) terms') 
        print('Sum_i < (|lnB|_i - <|lnB|>_i)^2 > estimate: ' + str(SUMexlnBest_som) + ' + O(F^{-2}) terms')
        for npairs in range(0,len(exlnBest)):
            print('<|lnB|>_' + str(npairs) + ' estimate: ' + str(exlnBest[npairs]) + ' + O(F^{-2}) terms')
            print('< (|lnB|_' + str(npairs) + ' - <|lnB|>_' + str(npairs) + ')^2 >' + ' estimate: ' + str(exlnBest_som[npairs]) + ' + O(F^{-2}) terms')
        # Return the results in terminal output form


    def decide_on_best_computation(self,ML_point,samples_model_ML,kde_model_ML,samples_model_evidence,error_vector,ML_threshold,fiducial_point_vector,forecast_data_function,model_index):
    # Decide on and output the best computation available for both the Maximum Likelihood and Evidence for a given model     
        if ML_point*np.exp(-ML_threshold) < kde_model_ML:
            if ML_point*np.exp(-ML_threshold) < samples_model_ML:
            # If kde_model_ML and samples_model_ML are both over the threshold we should compare the kernel bandwidth to the 
            # fiducial likelihood error vector in each dimension to decide if the prior samples can give a decent estimate. 
            # Otherwise we simply use the kde method to estimate the evidence and Maximum Likelihood
                use_sampling = True
                if any(value > 0.0 for value in np.asarray(np.sqrt(self.density_functions[model_index].bw)) - np.asarray(error_vector)):
                    use_sampling = False

                if use_sampling == False:
                    model_ML = kde_model_ML
                    model_evidence = kde_model_ML

                    model_evidence = np.sum(self.density_functions[model_index].pdf(forecast_data_function(fiducial_point_vector,fiducial_point_vector,error_vector,samples=self.number_of_forecast_samples)))/float(self.number_of_forecast_samples)
                    # Compute the evidence using samples from the forecast likelihood distribution and the local prior density

                if use_sampling == True:
                    model_ML = samples_model_ML
                    model_evidence = samples_model_evidence 
                self.number_of_category_B_points[model_index] += 1
                # Update the number of category B points for this model 
            else:
            # It must be true that kde_model_ML > samples_model_ML therefore we use the KDE method to avoid undersampling problems
                model_ML = kde_model_ML
                model_evidence = kde_model_ML
                self.number_of_category_D_points[model_index] += 1
                # Update the number of category D points for this model
        else:
            if ML_point*np.exp(-ML_threshold) < samples_model_ML:
            # It must be true that samples_model_ML > kde_model_ML therefore we use the sampling method to avoid issues with approximating
            # the fiducial liklelihood as a single point in parameter space
                model_ML = samples_model_ML
                model_evidence = samples_model_evidence
                self.number_of_category_A_points[model_index] += 1
                # Update the number of category A points for this model
            else:
            # Everything is ruled out so just go with the sampled versions of each quantity  
                model_ML = samples_model_ML
                model_evidence = samples_model_evidence
                self.number_of_category_C_points[model_index] += 1
                # Update the number of category C points for this model

        return [model_ML,model_evidence]    


    def decide_on_best_computation_with_fisher(self,ML_point,samples_model_ML,kde_model_ML,samples_model_evidence,fisher_matrix,ML_threshold,fiducial_point_vector,forecast_data_function,model_index):
    # Decide on and output the best computation available for both the Maximum Likelihood and Evidence for a given model now using the Fisher matrix    
        if ML_point*np.exp(-ML_threshold) < kde_model_ML:
            if ML_point*np.exp(-ML_threshold) < samples_model_ML:
            # If kde_model_ML and samples_model_ML are both over the threshold we should compare the kernel bandwidth to the 
            # future likelihood inverse Fisher matrix diagnoal in each dimension to decide if the prior samples can give a decent estimate. 
            # Otherwise we simply use the kde method to estimate the evidence and Maximum Likelihood
                use_sampling = True
                if any(value > 0.0 for value in np.asarray(np.sqrt(self.density_functions[model_index].bw)) - (1.0/fisher_matrix.diagonal())):
                    use_sampling = False

                if use_sampling == False:
                    model_ML = kde_model_ML
                    model_evidence = kde_model_ML

                    model_evidence = np.sum(self.density_functions[model_index].pdf(forecast_data_function(fiducial_point_vector,fiducial_point_vector,fisher_matrix,samples=self.number_of_forecast_samples)))/float(self.number_of_forecast_samples)
                    # Compute the evidence using samples from the forecast likelihood distribution and the local prior density

                if use_sampling == True:
                    model_ML = samples_model_ML
                    model_evidence = samples_model_evidence 
                self.number_of_category_B_points[model_index] += 1
                # Update the number of category B points for this model 
            else:
            # It must be true that kde_model_ML > samples_model_ML therefore we use the KDE method to avoid undersampling problems
                model_ML = kde_model_ML
                model_evidence = kde_model_ML
                self.number_of_category_D_points[model_index] += 1
                # Update the number of category D points for this model
        else:
            if ML_point*np.exp(-ML_threshold) < samples_model_ML:
            # It must be true that samples_model_ML > kde_model_ML therefore we use the sampling method to avoid issues with approximating
            # the fiducial liklelihood as a single point in parameter space
                model_ML = samples_model_ML
                model_evidence = samples_model_evidence
                self.number_of_category_A_points[model_index] += 1
                # Update the number of category A points for this model
            else:
            # Everything is ruled out so just go with the sampled versions of each quantity  
                model_ML = samples_model_ML
                model_evidence = samples_model_evidence
                self.number_of_category_C_points[model_index] += 1
                # Update the number of category C points for this model

        return [model_ML,model_evidence] 


    def utility_functions(self,fiducial_point_vector,chains_column_numbers,prior_column_numbers,forecast_data_function,number_of_points,number_of_prior_points,error_vector,mix_models=False,ML_threshold=5.0):
# The function which computes all of the utilities at each fiducial point

        ''' 
        Definitions:


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

        
        mix_models                             =  Boolean - True outputs U in all combinations of model comparison i.e. {i not j} U(M_i-M_j)
                                                          - False outputs U for all models wrt the reference model i.e. {i=1,...,N} U(M_i-M_0)  


        ML_threshold                           =  The threshold in the MLE of a model prior to be classed as `ruled out' with
                                                  respect to the Maximum Likelihood 


        '''

        if mix_models == False:
            number_of_model_pairs = len(self.model_name_list)
            # The number of models where the utility reference model is element 0
        if mix_models == True:
            number_of_model_pairs = len(self.model_name_list) + ((len(self.model_name_list))*(len(self.model_name_list)-1)/2)
            # The number of possible model pairs to be computed, making use of Binomial Theorem  

        decisivity = np.zeros(number_of_model_pairs)
        # Initialize an array of values of the decisivity for the number of model pairs
        E = np.zeros(number_of_model_pairs)
        # Initialize an array of values of the Bayesian evidence for the number of model pairs
        abslnB = np.zeros(number_of_model_pairs)
        # Initialize an array of values of the absolute log Bayes factor for the number of model pairs
        model_valid_ML = np.zeros(len(self.model_name_list))
        # Initialize an array of binary indicator variables - containing either valid (1.0) or invalid (0.0) points in the 
        # Maximum Likelihood average for each of the models 
        valid_ML = np.zeros(number_of_model_pairs)
        # Initialize an array of binary indicator variables - containing either valid (1.0) or invalid (0.0) points showing
        # in the case where either one of the two models has model_valid_ML[i] = 1.0 or 0.0, respectively

        ML_point = forecast_data_function(fiducial_point_vector,fiducial_point_vector,error_vector)
        # The maximum Maximum Likelihood point is trivial to find
        DKL = 0.0
        # Initialise the integral count Kullback-Leibler divergence at 0.0
        forecast_data_function_normalisation = 0.0
        # Initialise the normalisation for the forecast data

        running_total = 0 # Initialize a running total of points read in from the chains

        forecast_data_function_normalisation = np.sum(np.asarray(self.chains_weights)*forecast_data_function(np.asarray(self.chains_data),fiducial_point_vector,error_vector))
        # Compute the future posterior normalisation for the dkl value

        logargument = np.sum(np.asarray(self.chains_weights))*np.asarray(forecast_data_function(np.asarray(self.chains_data),fiducial_point_vector,error_vector)/forecast_data_function_normalisation)

        dkl = np.asarray((np.asarray(self.chains_weights)*forecast_data_function(np.asarray(self.chains_data),fiducial_point_vector,error_vector)/forecast_data_function_normalisation)*np.log(logargument))
        dkl[np.isnan(dkl)] = 0.0
        dkl[np.isinf(dkl)] = 0.0
        # Remove potential infinite and NAN values

        DKL = np.sum(dkl)
        # Compute the dkl

        for i in range(0,len(self.model_name_list)):
            
            running_total = 0 # Initialize a running total of points read in from each prior
 
            kde_model_ML = self.density_functions[i].pdf(fiducial_point_vector)*forecast_data_function(fiducial_point_vector,fiducial_point_vector,error_vector)
            # Kernel Density Estimation using the statsmodels.nonparametric toolkit to estimate the 
            # density locally at the fiducial point itself. This can protect against finite sampling scale effects

            samples_model_ML = 0.0
            # Initialise the maximum likelihood to be obtained from the model samples
               
            E[i] = np.sum(forecast_data_function(np.asarray(self.prior_data[i]),fiducial_point_vector,error_vector)/float(number_of_prior_points))
            # Calculate the forecast probability and therefore the contribution to the Bayesian evidence for each model

            samples_model_ML = np.max(forecast_data_function(np.asarray(self.prior_data[i]),fiducial_point_vector,error_vector))
            # Compute the maximum likelihood to be obtained from the model samples                    

            [model_ML,E[i]] = self.decide_on_best_computation(ML_point,samples_model_ML,kde_model_ML,E[i],error_vector,ML_threshold,fiducial_point_vector,forecast_data_function,i)
            # Use a specified procedure to decide on which of the methods is best to use for computation

            if ML_point*np.exp(-ML_threshold) < model_ML: model_valid_ML[i] = 1.0 
            # Decide on whether the Maximum Likelihood point in the prior space is large enough to satisfy ML averageing for each 
            # model, having specified a threshold of acceptance in effective 'n-sigma' 

        if mix_models == True: 
            avoid_repeat = 0
            # Initialize a variable to iterate through the loop to avoid overcounting in `mix_model mode'
            count_elements = 0
            # Count the elements to the arrays listing U for each model pair on the fly
 

        for j in range(0,len(self.model_name_list)):      
        # Summation over the model priors 
            if mix_models == False: 
                if model_valid_ML[j] == 1.0 or model_valid_ML[0] == 1.0:
                    valid_ML[j] = 1.0
                # Decide on whether the point is to be included within the average utility for this
                # specific model pair (model j and the reference model 0)
  
                if E[j] == 0.0:
                    self.number_of_maxed_evidences[j] += 1
                    # Update the number of maxed evidences for this model

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
 
            if mix_models == True: 
                if E[j] == 0.0:
                    self.number_of_maxed_evidences[j] += 1
                    # Update the number of maxed evidences for this model

                for k in range(avoid_repeat,len(self.model_name_list)):      
                # Two-fold summation over the model priors if in `mix_model mode' 
                    if model_valid_ML[k] == 1.0 or model_valid_ML[j] == 1.0:
                        valid_ML[count_elements] = 1.0
                    # Decide on whether the point is to be included within the average utility for this
                    # specific model pair (model k and the reference model j)

                    if E[j] == 0.0:
                        if E[k] == 0.0:
                            abslnB[count_elements] = 0.0
                        else:
                            abslnB[count_elements] = 1000.0 # Maximal permitted value for an individual sample
                    if E[k] == 0.0:
                        if E[j] == 0.0:
                            abslnB[count_elements] = 0.0
                        else:
                            abslnB[count_elements] = 1000.0 # Maximal permitted value for an individual sample
                    else:
                        abslnB[count_elements] = abs(np.log(E[k])-np.log(E[j])) # Compute absolute log Bayes factor utility
                    # The block above deals with unruly values like 0.0 inside the logarithms

                    if abs(np.log(E[k])-np.log(E[j])) >= 5.0:
                        if j > 0: decisivity[count_elements] = 1.0 # Compute decisivity utility (for each new forecast distribution this is either 1 or 0) 
                    count_elements+=1 # Iterate the element counter for the arrays           

                avoid_repeat+=1 # Iterate the avoidance of repeating a model pair in the two-fold loop 
        
        
        running_total = 0 # Initialize a running total of points read in from the chains

        return [abslnB,decisivity,DKL,np.log(E),valid_ML] 
        # Output utilities, raw log evidences and binary indicator variables in 'valid_ML' to either validate (1.0) or invalidate (0.0) the fiducial 
        # point in the case of each model pair - this is used in the Maximum Likelihood average procedure in rerun_foxi

   
    def gaussian_forecast(self,prior_point_vector,fiducial_point_vector,error_vector,samples=None):
    # Likelihood at some point - prior_point_vector - in arbirary dimensions using a Gaussian 
    # forecast distribution given a fiducial point vector and error vector
        prior_point_vector = np.asarray(prior_point_vector) 
        fiducial_point_vector = np.asarray(fiducial_point_vector)  
        error_vector = np.asarray(error_vector)
        covmat = np.identity(len(error_vector))*(error_vector**2)

        if samples:
            return np.random.multivariate_normal(fiducial_point_vector,covmat,samples)
            # Return a number of samples from the forecast distribution if sample number is chosen
        else:
            return sps.multivariate_normal(fiducial_point_vector,covmat).pdf(prior_point_vector)
            # Return a density if sample number is not chosen


    def fisher_matrix_forecast(self,prior_point_vector,fiducial_point_vector,fisher_matrix,samples=None):
    # Likelihood at some point - prior_point_vector - in arbirary dimensions using a Gaussian 
    # forecast distribution given a fiducial point vector and the Fisher matrix
        prior_point_vector = np.asarray(prior_point_vector) 
        fiducial_point_vector = np.asarray(fiducial_point_vector)  
        covmat = np.linalg.inv(fisher_matrix)

        if samples:
            return np.random.multivariate_normal(fiducial_point_vector,covmat,samples)
            # Return a number of samples from the forecast distribution if sample number is chosen
        else:
            return sps.multivariate_normal(fiducial_point_vector,covmat).pdf(prior_point_vector)
            # Return a density if sample number is not chosen


    def fisher_utility_functions(self,fiducial_point_vector,chains_column_numbers,prior_column_numbers,forecast_data_function,number_of_points,number_of_prior_points,fisher_matrix,mix_models=False,ML_threshold=5.0):
# A modification to 'utility_functions' that now allows for Fisher matrices to be used

        ''' 
        Definitions:


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


        fisher_matrix                          =  The specified Fisher matrix for this fiducial_point_vector  

        
        mix_models                             =  Boolean - True outputs U in all combinations of model comparison i.e. {i not j} U(M_i-M_j)
                                                          - False outputs U for all models wrt the reference model i.e. {i=1,...,N} U(M_i-M_0)  


        ML_threshold                           =  The threshold in the MLE of a model prior to be classed as `ruled out' with
                                                  respect to the Maximum Likelihood 


        '''

        if mix_models == False:
            number_of_model_pairs = len(self.model_name_list)
            # The number of models where the utility reference model is element 0
        if mix_models == True:
            number_of_model_pairs = len(self.model_name_list) + ((len(self.model_name_list))*(len(self.model_name_list)-1)/2)
            # The number of possible model pairs to be computed, making use of Binomial Theorem  

        decisivity = np.zeros(number_of_model_pairs)
        # Initialize an array of values of the decisivity for the number of model pairs
        E = np.zeros(number_of_model_pairs)
        # Initialize an array of values of the Bayesian evidence for the number of model pairs
        abslnB = np.zeros(number_of_model_pairs)
        # Initialize an array of values of the absolute log Bayes factor for the number of model pairs
        model_valid_ML = np.zeros(len(self.model_name_list))
        # Initialize an array of binary indicator variables - containing either valid (1.0) or invalid (0.0) points in the 
        # Maximum Likelihood average for each of the models 
        valid_ML = np.zeros(number_of_model_pairs)
        # Initialize an array of binary indicator variables - containing either valid (1.0) or invalid (0.0) points showing
        # in the case where either one of the two models has model_valid_ML[i] = 1.0 or 0.0, respectively

        ML_point = forecast_data_function(fiducial_point_vector,fiducial_point_vector,fisher_matrix)
        # The maximum Maximum Likelihood point is trivial to find
        DKL = 0.0
        # Initialise the integral count Kullback-Leibler divergence at 0.0
        forecast_data_function_normalisation = 0.0
        # Initialise the normalisation for the forecast data

        running_total = 0 # Initialize a running total of points read in from the chains

        forecast_data_function_normalisation = np.sum(np.asarray(self.chains_weights)*forecast_data_function(np.asarray(self.chains_data),fiducial_point_vector,fisher_matrix))
        # Compute the future posterior normalisation for the dkl value

        logargument = np.sum(np.asarray(self.chains_weights))*np.asarray(forecast_data_function(np.asarray(self.chains_data),fiducial_point_vector,fisher_matrix)/forecast_data_function_normalisation)

        dkl = np.asarray((np.asarray(self.chains_weights)*forecast_data_function(np.asarray(self.chains_data),fiducial_point_vector,fisher_matrix)/forecast_data_function_normalisation)*np.log(logargument))
        dkl[np.isnan(dkl)] = 0.0
        dkl[np.isinf(dkl)] = 0.0
        # Remove potential infinite and NAN values

        DKL = np.sum(dkl)
        # Compute the dkl

        for i in range(0,len(self.model_name_list)):
            
            running_total = 0 # Initialize a running total of points read in from each prior
 
            kde_model_ML = self.density_functions[i].pdf(fiducial_point_vector)*forecast_data_function(fiducial_point_vector,fiducial_point_vector,fisher_matrix)
            # Kernel Density Estimation using the statsmodels.nonparametric toolkit to estimate the 
            # density locally at the fiducial point itself. This can protect against finite sampling scale effects

            samples_model_ML = 0.0
            # Initialise the maximum likelihood to be obtained from the model samples
               
            E[i] = np.sum(forecast_data_function(np.asarray(self.prior_data[i]),fiducial_point_vector,fisher_matrix)/float(number_of_prior_points))
            # Calculate the forecast probability and therefore the contribution to the Bayesian evidence for each model

            samples_model_ML = np.max(forecast_data_function(np.asarray(self.prior_data[i]),fiducial_point_vector,fisher_matrix))
            # Compute the maximum likelihood to be obtained from the model samples                    

            [model_ML,E[i]] = self.decide_on_best_computation_with_fisher(ML_point,samples_model_ML,kde_model_ML,E[i],fisher_matrix,ML_threshold,fiducial_point_vector,forecast_data_function,i)
            # Use a specified procedure to decide on which of the methods is best to use for computation

            if ML_point*np.exp(-ML_threshold) < model_ML: model_valid_ML[i] = 1.0 
            # Decide on whether the Maximum Likelihood point in the prior space is large enough to satisfy ML averageing for each 
            # model, having specified a threshold of acceptance in effective 'n-sigma' 

        if mix_models == True: 
            avoid_repeat = 0
            # Initialize a variable to iterate through the loop to avoid overcounting in `mix_model mode'
            count_elements = 0
            # Count the elements to the arrays listing U for each model pair on the fly
 

        for j in range(0,len(self.model_name_list)):      
        # Summation over the model priors 
            if mix_models == False: 
                if model_valid_ML[j] == 1.0 or model_valid_ML[0] == 1.0:
                    valid_ML[j] = 1.0
                # Decide on whether the point is to be included within the average utility for this
                # specific model pair (model j and the reference model 0)
  
                if E[j] == 0.0:
                    self.number_of_maxed_evidences[j] += 1
                    # Update the number of maxed evidences for this model

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
 
            if mix_models == True: 
                if E[j] == 0.0:
                    self.number_of_maxed_evidences[j] += 1
                    # Update the number of maxed evidences for this model

                for k in range(avoid_repeat,len(self.model_name_list)):      
                # Two-fold summation over the model priors if in `mix_model mode' 
                    if model_valid_ML[k] == 1.0 or model_valid_ML[j] == 1.0:
                        valid_ML[count_elements] = 1.0
                    # Decide on whether the point is to be included within the average utility for this
                    # specific model pair (model k and the reference model j)

                    if E[j] == 0.0:
                        if E[k] == 0.0:
                            abslnB[count_elements] = 0.0
                        else:
                            abslnB[count_elements] = 1000.0 # Maximal permitted value for an individual sample
                    if E[k] == 0.0:
                        if E[j] == 0.0:
                            abslnB[count_elements] = 0.0 
                        else:
                            abslnB[count_elements] = 1000.0 # Maximal permitted value for an individual sample
                    else:
                        abslnB[count_elements] = abs(np.log(E[k])-np.log(E[j])) # Compute absolute log Bayes factor utility
                    # The block above deals with unruly values like 0.0 inside the logarithms

                    if abs(np.log(E[k])-np.log(E[j])) >= 5.0:
                        if j > 0: decisivity[count_elements] = 1.0 # Compute decisivity utility (for each new forecast distribution this is either 1 or 0) 
                    count_elements+=1 # Iterate the element counter for the arrays           

                avoid_repeat+=1 # Iterate the avoidance of repeating a model pair in the two-fold loop 
        
        
        running_total = 0 # Initialize a running total of points read in from the chains

        return [abslnB,decisivity,DKL,np.log(E),valid_ML] 
        # Output utilities, raw log evidences and binary indicator variables in 'valid_ML' to either validate (1.0) or invalidate (0.0) the fiducial 
        # point in the case of each model pair - this is used in the Maximum Likelihood average procedure in rerun_foxi


    def run_foxi(self,error_vector,mix_models=False): 
    # This is the main algorithm to compute the utilites and compute other quantities then output them to a file
        ''' 
        Quick usage and settings:  


        error_vector               =  Fixed predicted future measurement errors corresponding to the fiducial_point_vector   


        mix_models                 =  Boolean - True computes U in all combinations of model comparison i.e. {i not j} U(M_i-M_j)
                                              - False computes U for all models wrt the reference model i.e. {i=1,...,N} U(M_i-M_0)  


        '''


        self.flashy_foxi() # Display propaganda      

        forecast_data_function = self.gaussian_forecast # Change the forecast data function here (if desired)

        running_total = 0 # Initialize a running total of points read in from the chains

        if mix_models == False:
            deci = np.zeros(len(self.model_name_list))
            # Initialize an array of values of the decisivity utility function for the number of models (reference model is element 0)
            abslnB = np.zeros(len(self.model_name_list))
            # Initialize an array of values of the absolute log Bayes factor for the number of models (reference model is element 0)  

            plot_data_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_data.txt",'w')
            plot_data_file.write(str(len(self.chains_column_numbers)) + "\t" + str(len(self.model_name_list)) + "\t" + str(0) + "\n")
            # Initialize output file

        if mix_models == True:
            possible_model_pairs = len(self.model_name_list) + ((len(self.model_name_list))*(len(self.model_name_list)-1)/2)
            # The number of possible distinct model pairs, making use of Binomial Theorem
            deci = np.zeros(possible_model_pairs)
            # Initialize an array of values of the decisivity utility function for the possible number of distinct model pairs 
            abslnB = np.zeros(possible_model_pairs)
            # Initialize an array of values of the absolute log Bayes factor for the possible number of distinct model pairs 

            plot_data_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_data_mix_models.txt",'w')
            plot_data_file.write(str(len(self.chains_column_numbers)) + "\t" + str(possible_model_pairs) + "\t" + str(1) + "\n")
            # Initialize output file in the case where all possible model combinations are tried within each utility


        for j in range(0,len(self.chains_data)):
        # Compute utilities with stored chains 
        
            [abslnB,deci,DKL,lnE,valid_ML] = self.utility_functions(self.chains_data[j],self.chains_column_numbers,self.prior_column_numbers,forecast_data_function,self.number_of_points,self.number_of_prior_points,error_vector,mix_models=mix_models)
                
            plot_data_file.write(str(self.chains_weights[j]) + "\t" + "\t".join(map(str, np.asarray(self.chains_data[j]))) +  "\t"  +  str(j)  +  "\t"  +  "\t".join(map(str, np.asarray(abslnB))) +  "\t"  +  str(j)  +  "\t"  +  "\t".join(map(str, np.asarray(lnE))) +  "\t"  +  str(j)  +  "\t"  +  "\t".join(map(str, np.asarray(valid_ML))) +  "\t"  +  str(j)  +  "\t"  +  str(DKL) + "\n") 
                # Output fiducial point values etc... and other data to file if requested  
       
        plot_data_file.close() 

        points_category_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_data_number_points_categories.txt",'w')
        # Initialize output file containing the counts of each category of point (defined in arxiv:1803.09491)   

        points_category_file.write('Number of points in each category (defined in arxiv:1803.09491) for each model:' + '\n')
        for i in range(0,len(self.model_name_list)):
            points_category_file.write('Model ' + str(i) + ": " + "n_A = " + str(self.number_of_category_A_points[i]) + " " + "n_B = " + str(self.number_of_category_B_points[i]) + " " + "n_C = " + str(self.number_of_category_C_points[i]) + " " + "n_D = " + str(self.number_of_category_D_points[i]) + '\n')
            # Write output

        points_category_file.close()

        print('\n' + 'Number of maxed-out evidences for each model:' + '\n')
        for i in range(0,len(self.model_name_list)):
            print('Model ' + str(i) + ': ' + str(self.number_of_maxed_evidences[i]))
            # Print out the number of maxed-out evidences for each model at the end of the computation

    
    def run_foxifish(self,mix_models=False): 
    # This is the main algorithm which now includes a specified Fisher matrix
        ''' 
        Quick usage and settings:


        mix_models                 =  Boolean - True computes U in all combinations of model comparison i.e. {i not j} U(M_i-M_j)
                                              - False computes U for all models wrt the reference model i.e. {i=1,...,N} U(M_i-M_0)   
                                              
             
        '''


        self.flashy_foxifish() # Display propaganda... also, try to say 'flashy foxifish' really fast...     

        forecast_data_function = self.fisher_matrix_forecast # Change the forecast data function here (if desired)

        running_total = 0 # Initialize a running total of points read in from the chains

        if mix_models == False:
            deci = np.zeros(len(self.model_name_list))
            # Initialize an array of values of the decisivity utility function for the number of models (reference model is element 0)
            abslnB = np.zeros(len(self.model_name_list))
            # Initialize an array of values of the absolute log Bayes factor for the number of models (reference model is element 0)  

            plot_data_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_data.txt",'w')
            plot_data_file.write(str(len(self.chains_column_numbers)) + "\t" + str(len(self.model_name_list)) + "\t" + str(0) + "\n")
            # Initialize output file

        if mix_models == True:
            possible_model_pairs = len(self.model_name_list) + ((len(self.model_name_list))*(len(self.model_name_list)-1)/2)
            # The number of possible distinct model pairs, making use of Binomial Theorem
            deci = np.zeros(possible_model_pairs)
            # Initialize an array of values of the decisivity utility function for the possible number of distinct model pairs 
            abslnB = np.zeros(possible_model_pairs)
            # Initialize an array of values of the absolute log Bayes factor for the possible number of distinct model pairs 

            plot_data_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_data_mix_models.txt",'w')
            plot_data_file.write(str(len(self.chains_column_numbers)) + "\t" + str(possible_model_pairs) + "\t" + str(1) + "\n")
            # Initialize output file in the case where all possible model combinations are tried within each utility


        for j in range(0,len(self.chains_data)):
        # Compute utilities with stored chains 
        
            [abslnB,deci,DKL,lnE,valid_ML] = self.fisher_utility_functions(self.chains_data[j],self.chains_column_numbers,self.prior_column_numbers,forecast_data_function,self.number_of_points,self.number_of_prior_points,self.fisher_matrix_function(self.chains_data[j]),mix_models=mix_models)
                
            plot_data_file.write(str(self.chains_weights[j]) + "\t" + "\t".join(map(str, np.asarray(self.chains_data[j]))) +  "\t"  +  str(j)  +  "\t"  +  "\t".join(map(str, np.asarray(abslnB))) +  "\t"  +  str(j)  +  "\t"  +  "\t".join(map(str, np.asarray(lnE))) +  "\t"  +  str(j)  +  "\t"  +  "\t".join(map(str, np.asarray(valid_ML))) +  "\t"  +  str(j)  +  "\t"  +  str(DKL) + "\n") 
                # Output fiducial point values etc... and other data to file if requested  
       
        plot_data_file.close() 

        points_category_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_data_number_points_categories.txt",'w')
        # Initialize output file containing the counts of each category of point (defined in arxiv:1803.09491)   

        points_category_file.write('Number of points in each category (defined in arxiv:1803.09491) for each model:' + '\n')
        for i in range(0,len(self.model_name_list)):
            points_category_file.write('Model ' + str(i) + ": " + "n_A = " + str(self.number_of_category_A_points[i]) + " " + "n_B = " + str(self.number_of_category_B_points[i]) + " " + "n_C = " + str(self.number_of_category_C_points[i]) + " " + "n_D = " + str(self.number_of_category_D_points[i]) + '\n')
            # Write output

        points_category_file.close()

        print('\n' + 'Number of maxed-out evidences for each model:' + '\n')
        for i in range(0,len(self.model_name_list)):
            print('Model ' + str(i) + ': ' + str(self.number_of_maxed_evidences[i]))
            # Print out the number of maxed-out evidences for each model at the end of the computation


    def rerun_foxi(self,foxiplot_samples,number_of_foxiplot_samples,predictive_prior_types,TeX_output=False,model_name_TeX_input=[],as_a_function=False): 
    # This function takes in the foxiplot data file to compute secondary quantities like the centred second-moment of the utilities 
    # and also the utilities using Maximum Likelihood averageing are computed. This two-step sequence is intended to simplify computing
    # on a cluster
        ''' 
        Quick usage and settings:
                 

        foxiplot_samples               =  Name of an (un-edited) 'foxiplots_data.txt' file (or 'foxiplots_data_mix_models.txt' file) 
                                          that will be automatically read and interpreted


        number_of_foxiplot_samples     =  The number of points specified to be read off from the foxiplot data file


        predictive_prior_types         =  Takes a string as input, the choices are: 'flat' or 'log' - note that changing between 
                                          these can cause issues numerically if there are not enough points  

    
        TeX_output                     =  Boolean - True outputs the utilities in TeX table format

        
        model_name_TeX_input           =  An empty list which may be populated with model names written in TeX for the table

        
        as_a_function                  =  Boolean - True Converts 'rerun_foxi' into a function which returns the entire output as a nested list                                                   


        '''

        if as_a_function == False: self.flashy_foxi() # Display propaganda      

        running_total = 0 # Initialize a running total of points read in from the foxiplot data file
        
        line_for_analysis_1 = []
        # Initialise line to be used to analyse the foxiplot data file for correct reading

        with open(self.path_to_foxi_directory + '/' + self.output_directory + foxiplot_samples) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the foxiplot data file
            for line in file:
                columns = line.split()
                if running_total == 0:
                    line_for_analysis_1 = [float(value) for value in columns]
                running_total+=1 # Also add to the running total
                if running_total == 1: break # Finish once reached specified number of data points 
        # Read off lines to be used to analyse the foxiplot data file for correct reading
        
        [number_of_fiducial_point_dimensions,number_of_models] = self.analyse_foxiplot_line(line_for_analysis_1,as_a_function=as_a_function)
        # Read in lines, analyse and output useful information for reading the data file or return error

        predictive_prior_weight_total = 0.0
        # The normalisation for the predictive prior weights
        abslnB = np.zeros(number_of_models)
        # Initialize an array of values of the absolute log Bayes factor for the number of models (reference model is element 0)
        decisivity = np.zeros(number_of_models)
        # Initialize an array of values of the full decisivity for the number of models (reference model is element 0)
        expected_abslnB = np.zeros(number_of_models)
        # Initialize an array of values of the expected absolute log Bayes factor for the number of models (reference model is element 0)
        expected_abslnB_ML = np.zeros(number_of_models)
        # Initialize an the expected Maximum-Likelihood-averaged version of abslnB
        decisivity_ML = np.zeros(number_of_models)
        # Initialize an the expected Maximum-Likelihood-averaged version of the decisivity
        som_abslnB = np.zeros(number_of_models)
        # Initialize an array of values of the second-moment of the absolute log Bayes factor for the number of models (reference model is element 0)
        som_abslnB_ML = np.zeros(number_of_models)
        # Initialize som_abslnB for the Maximum Likelihood average scheme
        proportion_of_ML_failures = np.zeros(number_of_models)
        # Initialize the volume ratio of the fiducial point space associated to immediate disfavouring of both models due to the Bayesian evidence of both models being too low when compared with the Maximum Likelihood
        expected_DKL = 0.0
        # Initialize the count for expected Kullback-Leibler divergence   
        som_DKL = 0.0
        # Initialize the second-moment of the Kullback-Leibler divergence  
        total_valid_ML = np.zeros(number_of_models)
        # Initialize the additional normalisation factor for the Maximum Likelihood average  

        running_total = -1 # Initialize a running total of points read in from the foxiplot data file

        with open(self.path_to_foxi_directory + '/' + self.output_directory + foxiplot_samples) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the foxiplot data file
            for line in file:
                if running_total != -1:
                # Avoid reading the first line
                    columns = line.split()
                    fiducial_point_vector = [] 
                    valid_ML = []
                    invalid_ML =[]
                    DKL = float(columns[len(columns)-1])
                    chains_point_weight = float(columns[0]) # Find the weight to the point in the chains
                    for j in range(0,len(columns)):
                        if j < number_of_fiducial_point_dimensions+1 and j>0: 
                            fiducial_point_vector.append(float(columns[j]))
                        if j > number_of_fiducial_point_dimensions+1 and j < number_of_fiducial_point_dimensions+1 + number_of_models: 
                            if np.isnan(float(columns[j])) == False and np.isfinite(float(columns[j])) == True: 
                                abslnB[j - number_of_fiducial_point_dimensions-2] = float(columns[j])
                            else:
                                abslnB[j - number_of_fiducial_point_dimensions-2] = 1000.0 # Maximal permitted value for an individual sample
                            # Overall if else to avoid NANs and infinite values
                        if j > number_of_fiducial_point_dimensions + (2*number_of_models) + 3 and j < number_of_fiducial_point_dimensions + (3*number_of_models) + 4:  
                            valid_ML.append(float(columns[j]))
                            if float(columns[j]) == 0.0: invalid_ML.append(1.0)
                            if float(columns[j]) == 1.0: invalid_ML.append(0.0)
                    # Read in fiducial points, Maximum Likelihood points, invalid Maximum Likelihood points and utilities from foxiplot data file                          

                    predictive_prior_weight = chains_point_weight # Initialize the prior weight unit value
                    for i in range(0,len(fiducial_point_vector)):
                        if predictive_prior_types[i] == 'flat':
                            predictive_prior_weight *= 1.0
                        if predictive_prior_types[i] == 'log':
                            predictive_prior_weight *= 1.0/fiducial_point_vector[i]           
                    predictive_prior_weight_total += predictive_prior_weight
                    # Compute and store the predictive prior weights                 

                    deci = np.zeros(number_of_models)
                    # Initialize an array of values of the decisivity utility function for the number of models (reference model is element 0)
                    for i in range(0,number_of_models):
                        if abslnB[i] >= 5.0: deci[i] = 1.0
                    # Quickly compute the decisivity contribution using the absolute log Bayes factor for each model 

                    valid_ML = np.asarray(valid_ML)
                    invalid_ML = np.asarray(invalid_ML)
                    total_valid_ML += valid_ML*predictive_prior_weight
 
                    decisivity = decisivity + (deci*predictive_prior_weight) 
                    expected_abslnB = expected_abslnB + (abslnB*predictive_prior_weight) 
                    decisivity_ML = decisivity_ML + (valid_ML*predictive_prior_weight*deci) 
                    expected_abslnB_ML = expected_abslnB_ML + (valid_ML*predictive_prior_weight*abslnB) # Do quick vector additions to update the expected utilities
                    expected_DKL += (DKL*predictive_prior_weight)

                    proportion_of_ML_failures = proportion_of_ML_failures + (invalid_ML*predictive_prior_weight) 
                    # Compute the volume ratio of the fiducial point space associated to immediate disfavouring of both because their Bayesian evidences are too low when compared with the Maximum Likelihood

                running_total+=1 # Also add to the running total
                if running_total >= number_of_foxiplot_samples: break # Finish once reached specified number of data points 
        # Computed the expected utilities in the above loop

        decisivity /= predictive_prior_weight_total
        expected_abslnB /= predictive_prior_weight_total
        decisivity_ML /= total_valid_ML
        expected_abslnB_ML /= total_valid_ML
        expected_DKL /= predictive_prior_weight_total
        proportion_of_ML_failures /= predictive_prior_weight_total
        # Performing relevant normalisations

        running_total = -1 # Initialize a running total of points read in from the foxiplot data file

        with open(self.path_to_foxi_directory + '/' + self.output_directory + foxiplot_samples) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the foxiplot data file
            for line in file:
                if running_total != -1:
                # Avoid reading the first line
                    columns = line.split()
                    fiducial_point_vector = [] 
                    valid_ML = []
                    DKL = float(columns[len(columns)-1])
                    chains_point_weight = float(columns[0]) # Find the weight to the point in the chains
                    for j in range(0,len(columns)):
                        if j < number_of_fiducial_point_dimensions+1 and j>0: 
                            fiducial_point_vector.append(float(columns[j]))
                        if j > number_of_fiducial_point_dimensions+1 and j < number_of_fiducial_point_dimensions+1 + number_of_models: 
                            if np.isnan(float(columns[j])) == False and np.isfinite(float(columns[j])) == True: 
                                abslnB[j - number_of_fiducial_point_dimensions-2] = float(columns[j])
                            else:
                                abslnB[j - number_of_fiducial_point_dimensions-2] = 1000.0 # Maximal permitted value for an individual sample
                            # Overall if else to avoid NANs and infinite values
                        if j > number_of_fiducial_point_dimensions + (2*number_of_models) + 3 and j < number_of_fiducial_point_dimensions + (3*number_of_models) + 4:  
                            valid_ML.append(float(columns[j]))
                    # Read in fiducial points and utilities from foxiplot data file                      

                    predictive_prior_weight = chains_point_weight # Initialize the prior weight unit value
                    for i in range(0,len(fiducial_point_vector)):
                        if predictive_prior_types[i] == 'flat':
                            predictive_prior_weight *= 1.0
                        if predictive_prior_types[i] == 'log':
                            predictive_prior_weight *= 1.0/fiducial_point_vector[i]

                    predictive_prior_weights = [predictive_prior_weight for k in range(0,number_of_models)]
                    predictive_prior_weights = np.asarray(predictive_prior_weights)
                    # Create an array (one value per model) of prior weights for each sample

                    valid_ML = np.asarray(valid_ML)
                    som_abslnB += (((abslnB-expected_abslnB)**2)*predictive_prior_weights/predictive_prior_weight_total) 
                    som_abslnB_ML += (((abslnB-expected_abslnB_ML)**2)*predictive_prior_weights*(valid_ML/total_valid_ML)) # Do quick vector additions to update second-moment utilities
                    som_DKL += (((DKL-expected_DKL)**2)*predictive_prior_weight/predictive_prior_weight_total)

                running_total+=1 # Also add to the running total
                if running_total >= number_of_foxiplot_samples: break # Finish once reached specified number of data points 
        # Computed the second-moment utilities in the above loop

        if as_a_function == False:
            output_data_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_data_summary.txt",'w')
            output_data_file.write(' [<|lnB|>_1, <|lnB|>_2, ...] = ' + str(expected_abslnB) + "\n")
            output_data_file.write(' [DECI_1, DECI_2, ...] = ' + str(decisivity) + "\n")
            output_data_file.write('<DKL> = ' + str(expected_DKL) + "\n")
            output_data_file.write(' [<|lnB|_ML>_1, <|lnB|_ML>_2, ...] = ' + str(expected_abslnB_ML) + "\n") 
            output_data_file.write(' [r_ML|_1, r_ML|_2, ...] = ' + str(proportion_of_ML_failures) + "\n")
            output_data_file.write(' [< (|lnB|_1 - <|lnB|>_1)^2 >, < (|lnB|_2 - <|lnB|>_2)^2 >, ...] = ' + str(som_abslnB) + "\n")
            output_data_file.write(' [< (|lnB|_1 - <|lnB|>_1_ML)^2 >_ML, < (|lnB|_2 - <|lnB|>_2_ML)^2 >_ML, ...] = ' + str(som_abslnB_ML) + "\n")
            output_data_file.write(' [DECI_1|_ML, DECI_2|_ML, ...] = ' + str(decisivity_ML) + "\n")
            output_data_file.write('< (DKL-<DKL>)^2 > = ' + str(som_DKL))
            output_data_file.close()
            # Output data to `foxiplots_data_summary.txt'
        else:
            return [expected_abslnB,decisivity,expected_DKL,expected_abslnB_ML,proportion_of_ML_failures,som_abslnB,som_abslnB_ML,decisivity_ML,som_DKL]

        if TeX_output == True:
        # Output a TeX table!
            output_data_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_data_summary_TeX.txt",'w')
            output_data_file.write(r'\begin{table}' + '\n')
            output_data_file.write('\centering' + '\n')
            output_data_file.write(r'\begin{tabular}{|c|c|c|c|c|c|}' + '\n')
            output_data_file.write('\cline{2-6}' + '\n')
            output_data_file.write(r'\multicolumn{1}{c}{\cellcolor{red!55}} & \multicolumn{5}{|c|}{Data Name}  \\      \hline' + '\n')
            output_data_file.write(r'\backslashbox{${\cal M}_\beta$ - ${\cal M}_\gamma$}{$\langle  U\rangle$}  & $\langle \vert \ln {\rm B}_{\beta \gamma}\vert \rangle$ & $\langle \vert \ln {\rm B}_{\beta \gamma} \vert \rangle_{{}_{\rm ML}}$ & $\mathscr{D}_{\beta \gamma}$ & $\mathscr{D}_{\beta \gamma}\vert_{{}_{\rm ML}}$ & $r_{{}_\mathrm{ML}}$ \\     \hline\hline' + '\n')
            for k in range(0,number_of_models):
                if len(model_name_TeX_input) == 0:
                    if self.mix_models_was_used == False:
                        if round(np.sqrt(som_abslnB[k]),2) < round(expected_abslnB[k],2):
                            mid_string_1 = str(round(expected_abslnB[k],2)) + ' $\pm$ ' + str(round(np.sqrt(som_abslnB[k]),2))
                        else: 
                            mid_string_1 = '$' + str(round(expected_abslnB[k],2)) + '^{+' + str(round(np.sqrt(som_abslnB[k]),2)) + '}_{-' + str(round(expected_abslnB[k],2)) + '}' + '$'
                        if round(np.sqrt(som_abslnB_ML[k]),2) < round(expected_abslnB_ML[k],2):
                            mid_string_2 = str(round(expected_abslnB_ML[k],2)) + ' $\pm$ ' + str(round(np.sqrt(som_abslnB_ML[k]),2))
                        else:
                            mid_string_2 = '$' + str(round(expected_abslnB_ML[k],2)) + '^{+' + str(round(np.sqrt(som_abslnB_ML[k]),2)) + '}_{-' + str(round(expected_abslnB_ML[k],2)) + '}' + '$'

                        output_data_file.write('Model Pair ' + str(k) + '-' + str(0) + ' & ' + mid_string_1 + ' & ' + mid_string_2 + ' & ' + str(round(decisivity[k],2)) + ' & ' + str(round(decisivity_ML[k],2)) + ' & ' + str(round(proportion_of_ML_failures[k],2)) + r'  \\      \hline' + '\n')
                    
                    if self.mix_models_was_used == True:
                        number_of_individual_models = int(0.5 + np.sqrt((2.0*number_of_models)+0.25)) - 1  
                        # Compute the number of individual models 
                        if k == 0: 
                            target = number_of_individual_models-1
                            new_count = 0
                            previous_second_in_order = 0
                            primary_count = 0
                        # Initialise at k = 0

                        second_in_order = primary_count
                        if second_in_order != previous_second_in_order:
                            new_count = 0
                        first_in_order = second_in_order + new_count
                        previous_second_in_order = second_in_order
                        new_count += 1     

                        if k == target: 
                            primary_count += 1
                            target += number_of_individual_models-primary_count
                        # Compute the order in which the model pairs appear

                        if round(np.sqrt(som_abslnB[k]),2) < round(expected_abslnB[k],2):
                            mid_string_1 = str(round(expected_abslnB[k],2)) + ' $\pm$ ' + str(round(np.sqrt(som_abslnB[k]),2))
                        else: 
                            mid_string_1 = '$' + str(round(expected_abslnB[k],2)) + '^{+' + str(round(np.sqrt(som_abslnB[k]),2)) + '}_{-' + str(round(expected_abslnB[k],2)) + '}' + '$'
                        if round(np.sqrt(som_abslnB_ML[k]),2) < round(expected_abslnB_ML[k],2):
                            mid_string_2 = str(round(expected_abslnB_ML[k],2)) + ' $\pm$ ' + str(round(np.sqrt(som_abslnB_ML[k]),2))
                        else:
                            mid_string_2 = '$' + str(round(expected_abslnB_ML[k],2)) + '^{+' + str(round(np.sqrt(som_abslnB_ML[k]),2)) + '}_{-' + str(round(expected_abslnB_ML[k],2)) + '}' + '$'

                        output_data_file.write('Model Pair ' + str(first_in_order) + '-' + str(second_in_order) + ' & ' + mid_string_1 + ' & ' + mid_string_2 + ' & ' + str(round(decisivity[k],2)) + ' & ' + str(round(decisivity_ML[k],2)) + ' & ' + str(round(proportion_of_ML_failures[k],2)) + r'  \\      \hline' + '\n')

                else:
                    if self.mix_models_was_used == False:
                        if round(np.sqrt(som_abslnB[k]),2) < round(expected_abslnB[k],2):
                            mid_string_1 = str(round(expected_abslnB[k],2)) + ' $\pm$ ' + str(round(np.sqrt(som_abslnB[k]),2))
                        else: 
                            mid_string_1 = '$' + str(round(expected_abslnB[k],2)) + '^{+' + str(round(np.sqrt(som_abslnB[k]),2)) + '}_{-' + str(round(expected_abslnB[k],2)) + '}' + '$'
                        if round(np.sqrt(som_abslnB_ML[k]),2) < round(expected_abslnB_ML[k],2):
                            mid_string_2 = str(round(expected_abslnB_ML[k],2)) + ' $\pm$ ' + str(round(np.sqrt(som_abslnB_ML[k]),2))
                        else:
                            mid_string_2 = '$' + str(round(expected_abslnB_ML[k],2)) + '^{+' + str(round(np.sqrt(som_abslnB_ML[k]),2)) + '}_{-' + str(round(expected_abslnB_ML[k],2)) + '}' + '$'

                        output_data_file.write(model_name_TeX_input[k] + " - " + model_name_TeX_input[0] + ' & ' + mid_string_1 + ' & ' + mid_string_2 + ' & ' + str(round(decisivity[k],2)) + ' & ' + str(round(decisivity_ML[k],2)) + ' & ' + str(round(proportion_of_ML_failures[k],2)) + r'  \\      \hline' + '\n')

                    if self.mix_models_was_used == True:
                        number_of_individual_models = int(0.5 + np.sqrt((2.0*number_of_models)+0.25)) - 1  
                        # Compute the number of individual models 
                        if k == 0: 
                            target = number_of_individual_models-1
                            new_count = 0
                            previous_second_in_order = 0
                            primary_count = 0
                        # Initialise at k = 0

                        second_in_order = primary_count
                        if second_in_order != previous_second_in_order:
                            new_count = 0
                        first_in_order = second_in_order + new_count
                        previous_second_in_order = second_in_order
                        new_count += 1     

                        if k == target: 
                            primary_count += 1
                            target += number_of_individual_models-primary_count
                        # Compute the order in which the model pairs appear       

                        if round(np.sqrt(som_abslnB[k]),2) < round(expected_abslnB[k],2):
                            mid_string_1 = str(round(expected_abslnB[k],2)) + ' $\pm$ ' + str(round(np.sqrt(som_abslnB[k]),2))
                        else: 
                            mid_string_1 = '$' + str(round(expected_abslnB[k],2)) + '^{+' + str(round(np.sqrt(som_abslnB[k]),2)) + '}_{-' + str(round(expected_abslnB[k],2)) + '}' + '$'
                        if round(np.sqrt(som_abslnB_ML[k]),2) < round(expected_abslnB_ML[k],2):
                            mid_string_2 = str(round(expected_abslnB_ML[k],2)) + ' $\pm$ ' + str(round(np.sqrt(som_abslnB_ML[k]),2))
                        else:
                            mid_string_2 = '$' + str(round(expected_abslnB_ML[k],2)) + '^{+' + str(round(np.sqrt(som_abslnB_ML[k]),2)) + '}_{-' + str(round(expected_abslnB_ML[k],2)) + '}' + '$'

                        output_data_file.write(model_name_TeX_input[first_in_order] + " - " + model_name_TeX_input[second_in_order] + ' & ' + mid_string_1 + ' & ' + mid_string_2 + ' & ' + str(round(decisivity[k],2)) + ' & ' + str(round(decisivity_ML[k],2)) + ' & ' + str(round(proportion_of_ML_failures[k],2)) + r'  \\      \hline' + '\n')

            output_data_file.write('\end{tabular}' + '\n')
            output_data_file.write('\caption{Blah blah blah table...}' + '\n')
            output_data_file.write('\end{table}' + '\n')
            output_data_file.close()
            # Output data to `foxiplots_data_summary_TeX.txt' - TeX format


    def plot_foxiplots(self,filename_choice,column_numbers,ranges,number_of_bins,number_of_samples,label_values,corner_plot=False):
    # Plot data from files using either corner or 2D histogram and output to the directory foxiplots/
        ''' 
        Quick usage and settings:
                 
        
        filename_choice            =  Input the name of the file in foxioutput/ to be plotted 
       
 
        column_numbers             =  A list of column numbers (starting at 0) that will be read
                                      in from filename_choice and output the plot(s)


        ranges                     =  A tuple [(),(),...] with the plot ranges for all of the
                                      variables that are plotted 


        number_of_bins             =  Integer input for the number of bins for all axes


        number_of_samples          =  Integer number of samples to be plotted from the file


        label_values               =  Label values of interest - a list of these values with
                                      an element for each subplot in corner, otherwise should
                                      have value -> None -> for that element

        corner_plot                =  A Boolean: select 'True' for a corner plot of the utility
                                      values, select 'False' for a collection of 2D histograms  


        '''

        points = []
        # Initialise empty lists for samples

        running_total = -1 # Initialize a running total of points read in from the data file
        
        with open(self.path_to_foxi_directory + "/" + self.output_directory + filename_choice) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the chains
            for line in file:
                if running_total != -1:
                # Avoid reading the first line
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

        if corner_plot == False:
            for i in range(0,len(column_numbers)-1):
                y_points = [points[j][i] for j in range(0,number_of_samples)]
                x_points = [points[j][i+1] for j in range(0,number_of_samples)]
                y_step = (ranges[i][1]-ranges[i][0])/float(number_of_bins)
                x_step = (ranges[i+1][1]-ranges[i+1][0])/float(number_of_bins)
                # Estimate reasonable bin widths

                y_sides = [(y_step*float(j)) + ranges[i][0] for j in range(0,number_of_bins)]  
                x_sides = [(x_step*float(j)) + ranges[i+1][0] for j in range(0,number_of_bins)]       
                # Generate list of bins           

                z_values,x_sides,y_sides = np.histogram2d(y_points,x_points,bins=(x_sides,y_sides))
                # Generate histogram densities

                new_fig = plt.figure()
                axes = plt.gca()
                axes.set_ylim(ranges[i])
                axes.set_xlim(ranges[i+1])
                # Create a new figure for output with user-set axes limits

                x,y = np.meshgrid(x_sides,y_sides)
                # Generate a grid to plot from

                my_cmap = cm.get_cmap('BuPu')
                my_cmap.set_under('w') 
                # Making the lowest value 'true' white for style purposes          

                axes.pcolormesh(x,y,z_values,cmap=my_cmap)
                axes.scatter(x_points,y_points,alpha=0.1)
                # Generate the plot using pcolormesh and scatter
                plt.axis([ranges[i+1][0],ranges[i+1][1]-x_step,ranges[i][0],ranges[i][1]-y_step])
                # Fix edges of axes for neatness

                axes.set_xlabel(self.axes_labels[i+1])
                axes.set_ylabel(self.axes_labels[i])
                # Set user-defined labels

                for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] + axes.get_xticklabels() + axes.get_yticklabels()):
                    item.set_fontsize(self.fontsize)
                # User settings for the font size

                new_fig.savefig(self.path_to_foxi_directory + "/" + self.plots_directory + "foxiplot_2DHist_" + str(i) + ".pdf", format='pdf',bbox_inches='tight',pad_inches=0.1,dpi=300)
                # Saving output to foxiplots/

        if corner_plot == True:
            print('Using the corner module: https://pypi.python.org/pypi/corner/')
            figure = corner.corner(points,bins=number_of_bins,color='m',range=ranges,labels=self.axes_labels, label_kwargs={"fontsize": self.fontsize}, plot_contours=False,truths=label_values)
        # Use corner to plot the samples 

            figure.savefig(self.path_to_foxi_directory + "/" + self.plots_directory + "foxiplot.pdf", format='pdf',bbox_inches='tight',pad_inches=0.1,dpi=300)
        # Saving output to foxiplots/


    def plot_convergence(self,filename_choice,number_of_bins,total_number_of_samples,predictive_prior_types):
    # Make convergence plots of all utilities in a 'foxiplots_data' or 'foxiplots_data_mix_models' file and output to the directory foxiplots/
        ''' 
        Quick usage and settings:
                 
        
        filename_choice            =  Input the name of the file in foxioutput/ whose convergence plots are to be made  


        number_of_bins             =  Integer input for the number of bins for all axes


        predictive_prior_types     =  Takes a string as input, the choices are: 'flat' or 'log' - note that changing between 
                                      these can cause issues numerically if there are not enough points


        total_number_of_samples    =  Give the convergence plot up to a specified total number of samples



        '''

        tot_expected_abslnB = []
        tot_decisivity = []
        tot_expected_DKL = []
        tot_expected_abslnB_ML = []
        tot_proportion_of_ML_failures = []
        tot_decisivity_ML = []
        tot_number_of_samples = []
        # Initialise empty lists for plot values

        for ns in range(1,number_of_bins):
        # Count through the number of bins to re-evaluate the utilities
            number_of_samples_to_plot = int(float(ns)*(float(total_number_of_samples)/float(number_of_bins)))
            # Compute an integer number of samples to compute the utilities with each time

            [expected_abslnB,decisivity,expected_DKL,expected_abslnB_ML,proportion_of_ML_failures,som_abslnB,som_abslnB_ML,decisivity_ML,som_DKL] = self.rerun_foxi(filename_choice,number_of_samples_to_plot,predictive_prior_types,as_a_function=True)
            # Evaluate utilities each time using 'rerun_foxi'
            
            tot_number_of_samples.append(number_of_samples_to_plot)
            tot_expected_abslnB.append(expected_abslnB)
            tot_decisivity.append(decisivity)
            tot_expected_DKL.append(expected_DKL)
            tot_expected_abslnB_ML.append(expected_abslnB_ML)
            tot_proportion_of_ML_failures.append(proportion_of_ML_failures)
            tot_decisivity_ML.append(decisivity_ML)
            # Add to each point in the total list
            
        fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(6,6))
        # Create multi-tier plot of utilties   

        axes[0,0].plot(tot_number_of_samples,tot_expected_DKL)
        axes[0,0].set_title(r'$\langle D_{\rm KL}\rangle$', fontsize=self.fontsize)

        for nm in range(0,len(np.asarray(tot_expected_abslnB).T)):
        # Loop over the number of models to make each plot line
            axes[0,1].plot(tot_number_of_samples,np.asarray(tot_expected_abslnB).T[nm])
            axes[0,1].set_title(r'$\langle \vert \ln {\rm B}_{\beta \gamma}\vert \rangle$', fontsize=self.fontsize)
            axes[1,1].plot(tot_number_of_samples,np.asarray(tot_decisivity).T[nm])
            axes[1,1].set_title(r'$\mathscr{D}_{\beta \gamma}$', fontsize=self.fontsize)
            axes[0,2].plot(tot_number_of_samples,np.asarray(tot_expected_abslnB_ML).T[nm])
            axes[0,2].set_title(r'$\langle \vert \ln {\rm B}_{\beta \gamma} \vert \rangle_{{}_{\rm ML}}$', fontsize=self.fontsize)
            axes[1,2].plot(tot_number_of_samples,np.asarray(tot_decisivity_ML).T[nm])   
            axes[1,2].set_title(r'$\mathscr{D}_{\beta \gamma}\vert_{{}_{\rm ML}}$', fontsize=self.fontsize) 
            axes[1,0].plot(tot_number_of_samples,np.asarray(tot_proportion_of_ML_failures).T[nm])
            axes[1,0].set_title(r'$r_{{}_\mathrm{ML}}$', fontsize=self.fontsize)

        fig.subplots_adjust(hspace=0.4,wspace=0.4)
        fig.savefig(self.path_to_foxi_directory + "/" + self.plots_directory + "foxiplot_convergence.pdf", format='pdf',bbox_inches='tight',pad_inches=0.1,dpi=300)
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


    def flashy_foxifish(self): # Some more front page propaganda...
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
        print('    NOW WORKING WITH FISHER FORECASTS     ')
        print('                                          ')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


