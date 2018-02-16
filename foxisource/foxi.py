import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text',usetex=True)
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
        self.analyse_foxiplot_line
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
        self.set_prior_column_types
        self.column_types = []
        self.prior_column_types = []
        self.column_functions 
        self.column_types_are_set = False
        self.prior_column_types_are_set = False
        self.add_axes_labels
        self.axes_labels = []
        self.fontsize = 15
        self.decide_on_best_computation
        self.number_of_forecast_samples = 1000 # Number of samples to take from the forecast distribution 
        self.flat_function
        self.exp_function
        self.power10_function
        self.dictionary_of_column_types = {'flat': self.flat_function, 'log': self.exp_function, 'log10': self.power10_function}
        self.number_of_maxed_evidences = 0     


    def set_chains(self,name_of_chains): 
    # Set the file name of the chains of the current data set
        self.current_data_chains = name_of_chains

    
    def flat_function(self,x):
        return x

    
    def exp_function(self,x):
        return np.exp(x)


    def power10_function(self,x):
        return 10.0**(x)


    def set_model_name_list(self,model_name_list): 
    # Set the name list of models which can then be referred to by number for simplicity
        self.model_name_list = model_name_list
        # Input the .txt or .dat file names in the foxipriors/ directory
        self.number_of_category_A_points = [0 for i in model_name_list]
        self.number_of_category_B_points = [0 for i in model_name_list]
        self.number_of_category_C_points = [0 for i in model_name_list]
        self.number_of_category_D_points = [0 for i in model_name_list]
        # Initialise a count for each model in their category of points (see arxiv:XXXXXX)


    def set_column_types(self,column_types):
    # Input a list of strings to set the format of each column in data chains i.e. flat, log, log10, ...
        self.column_types = column_types
        self.column_types_are_set = True # All columns are as input unless this is True


    def set_prior_column_types(self,column_types):
    # Input a list of strings to set the format of each column in prior samples i.e. flat, log, log10, ...
        self.prior_column_types = column_types
        self.prior_column_types_are_set = True # All columns are as input unless this is True


    def set_number_of_forecast_samples(self,number_of_forecast_samples):
    # Change the number of samples to take from the forecast likelihood distribution when computing the evidence integrals
        self.number_of_forecast_samples = number_of_forecast_samples


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


    def analyse_foxiplot_line(self,line_1):
    # Reads first line from a foxiplot data file and outputs the number of models and dimension of data (this is just to check the input)
        dim_data = int(line_1[0])
        num_models = int(line_1[1])
        self.mix_models_was_used = int(line_1[2])
        
        if self.mix_models_was_used == False: 
            print("Just read in an ordinary foxiplot data file with data dimension " + str(dim_data) + " and the number of model pairs is " + str(num_models))
        if self.mix_models_was_used == True:
            print("Just read in a `mix_models' foxiplot data file with data dimension " + str(dim_data) + " and the number of model pairs is " + str(num_models))

        return [dim_data,num_models]


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


    def utility_functions(self,fiducial_point_vector,chains_column_numbers,prior_column_numbers,forecast_data_function,number_of_points,number_of_prior_points,error_vector,mix_models=False,ML_threshold=5.0):
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

        with open(self.path_to_foxi_directory + '/' + self.chains_directory + self.current_data_chains) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the chains
            for line in file:
                columns = line.split()
                columns = np.asarray(columns).astype(np.float)
 
                if self.column_types_are_set == True: 
                    fiducial_point_vector_for_integral = self.column_functions(np.asarray(columns[chains_column_numbers]))
                else: 
                    fiducial_point_vector_for_integral = np.asarray(columns[chains_column_numbers]) # All columns are flat formats unless this is True
                
                forecast_data_function_normalisation += forecast_data_function(fiducial_point_vector_for_integral,fiducial_point_vector,error_vector)
                # Compute the normalisation for the dkl value later on...                

                running_total+=1 # Also add to the running total                         
                if running_total >= number_of_points: break # Finish once reached specified number points

        for i in range(0,len(self.model_name_list)):
            
            running_total = 0 # Initialize a running total of points read in from each prior
 
            kde_model_ML = self.density_functions[i].pdf(fiducial_point_vector)*forecast_data_function(fiducial_point_vector,fiducial_point_vector,error_vector)
            # Kernel Density Estimation using the statsmodels.nonparametric toolkit to estimate the 
            # density locally at the fiducial point itself. This can protect against finite sampling scale effects

            samples_model_ML = 0.0
            # Initialise the maximum likelihood to be obtained from the model samples
               
            with open(self.path_to_foxi_directory + '/' + self.priors_directory + self.model_name_list[i]) as file:
            # Compute quantities in loop dynamically as with open(..) reads off the prior values
                for line in file:
                    columns = line.split()
                    columns = np.asarray(columns).astype(np.float)
                    
                    if self.prior_column_types_are_set == True: 
                        prior_point_vector = self.column_functions(np.asarray(columns[prior_column_numbers[i]]),prior=True)
                    else:
                        prior_point_vector = np.asarray(columns[prior_column_numbers[i]]) # All columns are as input unless this is True
                        
                    E[i] += forecast_data_function(prior_point_vector,fiducial_point_vector,error_vector)/float(number_of_prior_points)
                    # Calculate the forecast probability and therefore the contribution to the Bayesian evidence for each model

                    if forecast_data_function(prior_point_vector,fiducial_point_vector,error_vector) > samples_model_ML: 
                        samples_model_ML = forecast_data_function(prior_point_vector,fiducial_point_vector,error_vector)
                    # If one of the prior points has a larger maximum likelihood then replace the previous best

                    running_total+=1 # Also add to the running total                         
                    if running_total >= number_of_prior_points: break # Finish once reached specified number of prior points 

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

        with open(self.path_to_foxi_directory + '/' + self.chains_directory + self.current_data_chains) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the chains
            for line in file:
                columns = line.split()
                columns = np.asarray(columns).astype(np.float)
                
                if self.column_types_are_set == True: 
                    fiducial_point_vector_for_integral = self.column_functions(np.asarray(columns[chains_column_numbers]))
                else:
                    fiducial_point_vector_for_integral = np.asarray(columns[chains_column_numbers]) # All columns are flat formats unless this is True
                
                logargument = float(number_of_points)*forecast_data_function(fiducial_point_vector_for_integral,fiducial_point_vector,error_vector)/forecast_data_function_normalisation
                dkl = (forecast_data_function(fiducial_point_vector_for_integral,fiducial_point_vector,error_vector)/forecast_data_function_normalisation)*np.log(logargument)
                if np.isnan(dkl) == False and logargument > 0.0: DKL += dkl # Conditions to avoid problems in the logarithm

                running_total+=1 # Also add to the running total                         
                if running_total >= number_of_points: break # Finish once reached specified number points

        return [abslnB,decisivity,DKL,np.log(E),valid_ML] 
        # Output utilities, raw log evidences and binary indicator variables in 'valid_ML' to either validate (1.0) or invalidate (0.0) the fiducial 
        # point in the case of each model pair - this is used in the Maximum Likelihood average procedure in rerun_foxi

   
    def gaussian_forecast(self,prior_point_vector,fiducial_point_vector,error_vector,samples=None):
    # Posterior probability at some point - prior_point_vector - in arbirary dimensions using a Gaussian 
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


    def run_foxi(self,chains_column_numbers,prior_column_numbers,number_of_points,number_of_prior_points,error_vector,mix_models=False): 
    # This is the main algorithm to compute the utilites and compute other quantities then output them to a file
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
            plot_data_file.write(str(len(chains_column_numbers)) + "\t" + str(len(self.model_name_list)) + "\t" + str(0) + "\n")
            # Initialize output file

        if mix_models == True:
            possible_model_pairs = len(self.model_name_list) + ((len(self.model_name_list))*(len(self.model_name_list)-1)/2)
            # The number of possible distinct model pairs, making use of Binomial Theorem
            deci = np.zeros(possible_model_pairs)
            # Initialize an array of values of the decisivity utility function for the possible number of distinct model pairs 
            abslnB = np.zeros(possible_model_pairs)
            # Initialize an array of values of the absolute log Bayes factor for the possible number of distinct model pairs 

            plot_data_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_data_mix_models.txt",'w')
            plot_data_file.write(str(len(chains_column_numbers)) + "\t" + str(possible_model_pairs) + "\t" + str(1) + "\n")
            # Initialize output file in the case where all possible model combinations are tried within each utility

        self.density_functions = []
        # Initialise an empty list of density functions that may or may not be used

        for i in range(0,len(self.model_name_list)):  
         
            running_total = 0 # Initialize a running total of points read in from each prior      
 
            prior_data = [] # Initialize empty list for the new prior data

            with open(self.path_to_foxi_directory + '/' + self.priors_directory + self.model_name_list[i]) as file:
            # Compute quantities in loop dynamically as with open(..) reads off the prior values
                for line in file:
                    columns = line.split()
                    columns = np.asarray(columns).astype(np.float)

                    if self.prior_column_types_are_set == True: 
                        prior_point_vector = self.column_functions(np.asarray(columns[prior_column_numbers[i]]),prior=True)
                    else: 
                        prior_point_vector = np.asarray(columns[prior_column_numbers[i]]) # All columns are as input unless this is True
                    
                    prior_data.append(np.asarray(prior_point_vector)) # Store the data for the KDE
                                      
                    # Calculate the forecast probability and therefore the contribution to the Bayesian evidence for each model
                    running_total+=1 # Also add to the running total                         
                    if running_total >= number_of_prior_points: break # Finish once reached specified number of prior points
            
            c_string = ''
            for j in range(0,len(prior_column_numbers[i])): c_string += 'c'
            # Silly notational detail to get the kde to work for continuous variables in all dimensions    

            self.density_functions.append(kde(prior_data,var_type=c_string)) # Append a new Density Estimation function to the list for use later
            # Kernel Density Estimation uses statsmodels.nonparametric toolkit to estimate the 
            # density locally at the fiducial point itself, ensuring that there are no lost points
            # within each prior volume 'hull'

        
        # When using KDE method we quote the bandwidths used for each dimension for smoothing
        print('Using the statsmodels module: http://www.statsmodels.org/stable/index.html')
        print('The Kernel Density Bandwidth for each model listed in each dimension:' + '\n')
        for i in range(0,len(self.model_name_list)):
             print('Model ' + str(i) + ': ' + str(self.density_functions[i].bw))

        running_total = 0 # Re-initialize a running total of points read in from the chains

        with open(self.path_to_foxi_directory + '/' + self.chains_directory + self.current_data_chains) as file:
        # Compute quantities in loop dynamically as with open(..) reads off the chains
            for line in file:
                columns = line.split()
                columns = np.asarray(columns).astype(np.float)              

                if self.column_types_are_set == True: 
                    fiducial_point_vector = self.column_functions(np.asarray(columns[chains_column_numbers]))
                else: 
                    fiducial_point_vector = np.asarray(columns[chains_column_numbers]) # All columns are flat formats unless this is True

                [abslnB,deci,DKL,lnE,valid_ML] = self.utility_functions(fiducial_point_vector,chains_column_numbers,prior_column_numbers,forecast_data_function,number_of_points,number_of_prior_points,error_vector,mix_models=mix_models)
                
                plot_data_file.write("\t".join(map(str, np.asarray(fiducial_point_vector))) +  "\t"  +  str(running_total)  +  "\t"  +  "\t".join(map(str, np.asarray(abslnB))) +  "\t"  +  str(running_total)  +  "\t"  +  "\t".join(map(str, np.asarray(lnE))) +  "\t"  +  str(running_total)  +  "\t"  +  "\t".join(map(str, np.asarray(valid_ML))) +  "\t"  +  str(running_total)  +  "\t"  +  str(DKL) + "\n") 
                # Output fiducial point values etc... and other data to file if requested 


                running_total+=1 # Also add to the running total
                if running_total >= number_of_points: break # Finish once reached specified number of data points 
       
        plot_data_file.close() 

        points_category_file = open(self.path_to_foxi_directory + "/" + self.output_directory + "foxiplots_data_number_points_categories.txt",'w')
        # Initialize output file containing the counts of each category of point (defined in arxiv:XXXXXX)   

        points_category_file.write('Number of points in each category (defined in arxiv:XXXXXX) for each model:' + '\n')
        for i in range(0,len(self.model_name_list)):
            points_category_file.write('Model ' + str(i) + ": " + "n_A = " + str(self.number_of_category_A_points[i]) + " " + "n_B = " + str(self.number_of_category_B_points[i]) + " " + "n_C = " + str(self.number_of_category_C_points[i]) + " " + "n_D = " + str(self.number_of_category_D_points[i]) + '\n')
            # Write output

        points_category_file.close()


    def rerun_foxi(self,foxiplot_samples,number_of_foxiplot_samples,predictive_prior_types,TeX_output=False,model_name_TeX_input=[]): 
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
 


        '''


        self.flashy_foxi() # Display propaganda      

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
        
        [number_of_fiducial_point_dimensions,number_of_models] = self.analyse_foxiplot_line(line_for_analysis_1)
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
                    for j in range(0,len(columns)):
                        if j < number_of_fiducial_point_dimensions: 
                            fiducial_point_vector.append(float(columns[j]))
                        if j > number_of_fiducial_point_dimensions and j < number_of_fiducial_point_dimensions + number_of_models: 
                            if np.isnan(float(columns[j])) == False and np.isfinite(float(columns[j])) == True: 
                                abslnB[j - number_of_fiducial_point_dimensions - 1] = float(columns[j])
                            else:
                                abslnB[j - number_of_fiducial_point_dimensions - 1] = 1000.0 # Maximal permitted value for an individual sample
                            # Overall if else to avoid NANs and infinite values
                        if j > number_of_fiducial_point_dimensions + (2*number_of_models) + 2 and j < number_of_fiducial_point_dimensions + (3*number_of_models) + 3:  
                            valid_ML.append(float(columns[j]))
                            if float(columns[j]) == 0.0: invalid_ML.append(1.0)
                            if float(columns[j]) == 1.0: invalid_ML.append(0.0)
                    # Read in fiducial points, Maximum Likelihood points, invalid Maximum Likelihood points and utilities from foxiplot data file                          

                    predictive_prior_weight = 1.0 # Initialize the prior weight unit value
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
                    for j in range(0,len(columns)):
                        if j < number_of_fiducial_point_dimensions: 
                            fiducial_point_vector.append(float(columns[j]))
                        if j > number_of_fiducial_point_dimensions and j < number_of_fiducial_point_dimensions + number_of_models: 
                            if np.isnan(float(columns[j])) == False and np.isfinite(float(columns[j])) == True: 
                                abslnB[j - number_of_fiducial_point_dimensions - 1] = float(columns[j])
                            else:
                                abslnB[j - number_of_fiducial_point_dimensions - 1] = 1000.0 # Maximal permitted value for an individual sample
                            # Overall if else to avoid NANs and infinite values
                        if j > number_of_fiducial_point_dimensions + (2*number_of_models) + 2 and j < number_of_fiducial_point_dimensions + (3*number_of_models) + 3:  
                            valid_ML.append(float(columns[j]))
                    # Read in fiducial points and utilities from foxiplot data file                      

                    predictive_prior_weight = 1.0 # Initialize the prior weight unit value
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

        # Normalise outputs and avoid NAN values + infinite values
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
    # Plot data from files using either corner or 2D histogram and output to the directory foxioutput/
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



