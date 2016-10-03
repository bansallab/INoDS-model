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
## provide master list of node names
nodelist = [str(num) for num in xrange(100)]
# provide filename for output files
output_filename = "complete_data_SI_beta0.045"

###########################################
### Model parameters
###########################################

##the number of null networks to create null ditribution of predictive power
##NOTE: edge connections of networks are completely randomized,i.e., Jaccard index=0
##If complete randomization is not possible, then the model will throw an error
null_networks = 100 
#prior for beta, alpha
#beta = transmission parameter captured by network hypothesis
#"alpha" quantifies missingness of network hypothesis
priors = [(0,1), (0,1)] 

##do you know the true values? If not set it to [0,0,0]
truth = [0.045, 0, 0.01]
#Can nodes recover? If no, set recovery prob to np.inf. If yes, specify prior 
recovery_prob = np.inf

infection_type = "SI"
##specify chain length and burn-in
burnin = 200
#number of iterations after burnin
iteration = 500


## minor specfications
verbose=True
normalize_edge_weight= False

#####################################
#### run INoDS 
######################################
start = time.time()
inods.run_nbda_analysis(edge_filename, health_filename, output_filename, infection_type, nodelist, recovery_prob, truth, null_networks, priors, iteration, burnin, diagnosis_lag=False, null_comparison=True)

end = time.time()
print ("total run time="), end-start
