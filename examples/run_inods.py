import sys
import os
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import INoDS_model as inods
import numpy as np
##################################################

edge_filename = "Edge_connections_poisson.csv"
health_filename = "Health_data_nolag.csv"
output_filename = "complete_data_SI_beta0.045"
null_networks = 100 ##the number of null networks required
priors = [(0,1), (0,1)] #order = beta, alpha
truth = [0.045, 0, 0.01]
#can be np.inf(=no recovery), or bounds
recovery_prob = np.inf
verbose=True
iteration = 1000
burnin = 750
normalize_edge_weight= False
nodelist = [str(num) for num in xrange(100)]

inods.run_nbda_analysis(edge_filename, health_filename, output_filename, nodelist, recovery_prob, truth, null_networks, priors, iteration, burnin, diagnosis_lag=False, null_comparison=True)
