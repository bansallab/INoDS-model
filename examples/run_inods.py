import sys
sys.path.insert(0, r'/home/prathasah/Dropbox (Bansal Lab)/Git-files/INoDS-model')
import INoDS_model as inods
import numpy as np
import scipy.stats as ss
import time
##################################################

null_networks = 10 ##the number of null networks required
priors = [(0,1), (0,1)] #order = beta, alpha
truth = [0.045, 0, ss.randint.cdf(0,0, null_networks+1)]


#recovery_prob required only when there is diagnosis_lag
recovery_prob = [0, 1]
verbose=True
iteration = 1000
burnin = 1000
normalize_edge_weight= False
nodelist = [str(num) for num in xrange(100)]

infection_type = "SI"
edge_filename = "Edge_connections_poisson.csv"
health_filename = "Health_data_nolag.csv"
output_filename = "complete_data_infection_type"+infection_type+"_beta0.045_burnin_"+str(burnin)+"_iter_"+str(iteration)

start = time.time()
inods.run_nbda_analysis(edge_filename, health_filename, output_filename, infection_type, nodelist, recovery_prob, truth, null_networks, priors, iteration, burnin, diagnosis_lag=False, null_comparison=True)

end = time.time()
print ("total run time="), end-start
