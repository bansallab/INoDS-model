import sys
import os
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..')), ] + sys.path
import INoDS_model as inods
import numpy as np
import time
##################################################
## NOTE: INoDS requires the network and health data to be formatted in
## a specific manner. Check example files.
###############################################

#############################################
### Please edit these based on your input files
#############################################
#Provide the network hypothesis stored as an edgelist
edge_filename = "Edge_connections_poisson_n100_d4.csv"
# Prvide the health data
health_filename = "Health_data_nolag_beta_0.05_iter_0.csv"
# provide filename for output files
output_filename = "complete_data_SI_beta0.05"
###########################################
### Model parameters
###########################################
##do you know the true values of beta and epsilon? 
truth = [0.05, 0]

infection_type = "SI"
#####################################
#### run INoDS 
######################################
start = time.time()
inods.run_inods_sampler(edge_filename, health_filename, output_filename, infection_type, truth = truth, verbose=True, diagnosis_lag=False, burnin = 100,max_iteration = 300, null_networks=50, null_comparison=True, normalize_edge_weight=False, is_network_dynamic=True, parameter_estimate = True)

end = time.time()
print ("total run time (in minds)="), (end-start)/60.
