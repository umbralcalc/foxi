# foxi

### Futuristic Observations and their eXpected Information

Using information theory and Bayesian inference this python package computes a suite of expected utilities given futuristic observations in a flexible and user-friendly way. In principle, all one needs to make use of foxi is a set of n-dim prior samples for each model and one set of n-dim samples from the current data. 

05/08/2018 - New features including support for forecast Fisher matrices!!! Example script to follow in due course.  

### The expected utilities used include...

1. The expected ln-Bayes factor between models and its Maximum-Likelihood averaged equivalent (see arxiv:1803.09491).

2. The decisiveness between models (see arxiv:1803.09491) and its Maximum-likelihood averaged equivalent, the decisivity.

3. The expected Kullback-Leibler divergence (or information gain) of the futuristic dataset.

### Main features

Flexible inputs – usable essentially for any forecasting problem in science with suitable samples. foxi is designed for all-in-one script calculation or an initial cluster run then local machine post-processing, which should make large jobs quite manageable subject to resources. We have added features such LaTeX tables and plot making for post-data analysis visuals and convenience of presentation. In addition, we have designed some user-friendly scripts with plenty of comments to get familiar with foxi.

## Getting started

To fork, simply type into the terminal:

> git clone https://github.com/umbralcalc/foxi.git 

...then enter the scripts directory

> cd foxi/foxiscripts

and take a look!

