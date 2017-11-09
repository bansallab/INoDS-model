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
edge_filename = "Edge_connections_poisson.csv"
# Prvide the health data
health_filename = "Health_data_nolag.csv"
# provide filename for output files
output_filename = "complete_data_SI_beta0.045"

###########################################
### Model parameters
###########################################

##the number of null networks to create null ditribution of predictive power
##NOTE: edge connections of networks are completely randomized,i.e., Jaccard index=0
##If complete randomization is not possible, then the model will throw an error
null_networks = 100 

##do you know the true values? If not set it to [0,0,0]
truth = [0.045, 0, 0.01]

infection_type = "SI"
##specify chain length and burn-in
burnin = 10
#number of iterations after burnin
iteration = 50

#####################################
#### run INoDS 
######################################
start = time.time()
inods.run_inods_sampler(edge_filename, health_filename, output_filename, infection_type, truth, null_networks,  iteration, burnin, verbose=True, diagnosis_lag=False, null_comparison=True, normalize_edge_weight=False, is_network_dynamic=True, parameter_estimate = True)

end = time.time()
print ("total run time="), end-start
