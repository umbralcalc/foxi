import sys
sys.path.append('INSERT PATH TO foxisource/')
from foxi import foxi

foxiplots_instance = foxi('INSERT PATH TO foxi/') # Fire up a new instance of foxi for plotting
filename_choice = 'foxiplots_data.txt' # Choose file to be plotted
column_numbers = [0,1] # Choose the columns from the file to plot 
axes_labels = [r'$\mathrm{Anything 1}$',r'$\mathrm{Anything 2}$']
fontsize = 15 # Axes font sizes
ranges = [(0.0,1.0),(0.0,1.0)] # Lower and upper bounds set in a tuple for each variable
number_of_bins = 100 # Set the number of bins for the plot
number_of_samples = 100 # Set the number of samples to plot
label_values = [None,None] # Label important values within a list for each subplot, otherwise None for that element 
corner_plot = True # Decide between a corner plot or series of 2D histograms

foxiplots_instance.add_axes_labels(axes_labels,fontsize) # Add the LaTeX axes labels and fontsize
foxiplots_instance.plot_foxiplots(filename_choice,column_numbers,ranges,number_of_bins,number_of_samples,label_values,corner_plot=corner_plot) # Making plots

