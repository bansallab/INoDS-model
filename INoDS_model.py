import numpy as np
import matplotlib
matplotlib.use('Agg')
import networkx as nx
import csv
from numpy import ma
import dynesty
import corner
import copy
import matplotlib.pyplot as plt
import random as rnd
from multiprocessing import Pool, cpu_count
import INoDS_convenience_functions as nf
import warnings
import scipy.stats as ss
import time
import itertools
import pandas as pd
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
from dynesty import utils as dyfunc
from dynesty.dynamicsampler import stopping_function, weight_function, _kld_error
np.seterr(invalid='ignore')
np.seterr(divide='ignore')
warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)




#########################################################################
def diagnosis_adjustment(G, network, p, nodelist,contact_daylist,  recovery_prob, max_recovery_time, node_health_new, health_data_new, seed_date, network_min_date):


	###ensure that the proposal do not include 0 and are <1 
	diag_list = [min(max(num,0.000001),1) for num in p['diag_lag'][0]]
	
	##compute lagged time for each infection time
	lag_dict = [(node, time1, time2, int(ss.randint.ppf(diag_lag, 0,  len(contact_daylist[network][(node, time1, time2)])))) for (node, time1, time2), diag_lag in zip(sorted(contact_daylist[network]), diag_list)]
		
	## pick out corresponding date from contact_daylist
	new_infection_time= [(node, time1, time2, contact_daylist[network][(node, time1, time2)][lag_pos]) for (node, time1, time2, lag_pos) in lag_dict]
	##order = node, old infection time, old recovery time, new infection time and new recovery time
	new_infect_recovery_time =  [(node, time1, time2, new_time1, time2) for (node, time1, time2, new_time1) in new_infection_time]
		
	#########################################################
	# imputing recovery date##
	##########################################################	
	if recovery_prob:
	
		###ensure that the proposal recovery times do not include 0 and are <1 
		recovery_list = [min(max(num,0.000001),1) for num in p['gamma'][0]]
			
		## pick out corresponding recovery date (+1 to include period after time2  and time including max_recovery_time)
		new_infect_recovery_time = [(node, time1, time2, new_time1, int(ss.randint.ppf(recovery_param, time2,  max_recovery_time[(node, time1, time2)]+1))) for (node, time1, time2, new_time1, new_time2), recovery_param in zip(sorted(new_infect_recovery_time), recovery_list)]	
	##########################################################

	for (node, time1, time2, new_time1, new_time2) in new_infect_recovery_time:
		node_health_new[node][1].remove((time1, time2))
		node_health_new[node][1].append((new_time1, new_time2))	
		
		health_data_new[node] = {day: 1 for day in range(new_time1, new_time2+1)}
			
	infected_strength={}
	infected_strength[network] = {node:{time: calculate_infected_strength(node, time, health_data_new, G) for time in G.keys()} for node in nodelist}

	healthy_nodelist = return_healthy_nodelist(node_health_new ,seed_date, network_min_date)

	#create infection date list
	infection_date = [(node, new_time1) for (node, time1, time2, new_time1, new_time2) in new_infect_recovery_time if new_time1!= seed_date and new_time1 > network_min_date]
	infection_date = sorted(infection_date)	
	
	return infected_strength, healthy_nodelist, infection_date

#######################################################################
def log_likelihood(parameters, data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, max_recovery_time, network_min_date, parameter_estimate):
	r"""Computes the log-likelihood of network given infection data """
	
	if null_comparison:
		G_raw, health_data, node_health, nodelist, truth, time_min, time_max, seed_date,parameter_estimate = data
		health_data_new = copy.deepcopy(health_data)
		node_health_new = copy.deepcopy(node_health)		
		p = to_params(parameters, null_comparison, diagnosis_lag, nsick_param, recovery_prob, parameter_estimate)
		network =round(p['model'][0],2)
		G = G_raw[network]
		
	else:
		G_raw, health_data, node_health, nodelist, truth, time_min, time_max, seed_date  = data
		health_data_new = copy.deepcopy(health_data)
		node_health_new = copy.deepcopy(node_health)
		p = to_params(parameters, null_comparison, diagnosis_lag, nsick_param, recovery_prob, parameter_estimate)
		network = 0
		G= G_raw[network]

	###############################################################################################
	##diagnosis lag==
	##impute true infection date and recovery date (if SIR/SIS...)
	## infection_date = date picked as a day between last healthy report and first sick report
	## and when the degree of node was >0 the previous day
	##recovery_date = date picked as day with uniform probability between first reported sick day and first 
	##healthy date after sick report
	##################################################################################################
	
	if diagnosis_lag:
		infected_strength, healthy_nodelist, infection_date = diagnosis_adjustment(G, network, p, nodelist, contact_daylist, recovery_prob, max_recovery_time, node_health_new, health_data_new, seed_date, network_min_date)
		
	
	################################################################
	##Calculate rate of learning for all sick nodes at all sick    #
	## dates, but not when sick day is the seed date (i.e., the    #
	## first  report of the infection in the network               #
	################################################################	
	overall_learn_raw = np.array([calculate_lambda1(p['beta'][0], p['epsilon'][0], infected_strength[network], focal_node, sick_day) for (focal_node, sick_day) in infection_date])
	overall_learn = np.log(np.maximum(overall_learn_raw, 0.000001)) 
	################################################################
	##Calculate rate of NOT learning for all the days the node was #
	## (either reported or inferred) healthy                       #
	################################################################
	overall_not_learn_raw = not_learned_rate(healthy_nodelist,  p['beta'][0],p['epsilon'][0], infected_strength[network], seed_date, network_min_date)
	overall_not_learn_raw =  np.maximum(overall_not_learn_raw, 0.000001)
	overall_not_learn = np.log(overall_not_learn_raw)
	
	###########################################################
	## Calculate overall log likelihood                       #
	###########################################################
	loglike = overall_learn.sum() + overall_not_learn.sum()
	#print (p['beta'][0], p['epsilon'][0], network, loglike),
	if np.isinf(loglike) or np.isnan(loglike) or (loglike==0):return -np.inf
	else: return loglike

#############################################################################
def not_learned_rate(healthy_nodelist, beta, epsilon, infected_strength_network, seed_date, network_min_date):
	r""" Calculate 1- lambda for all uninfected days and returns 
	sum of log(1-lambdas)"""

	return np.array([1-calculate_lambda1(beta, epsilon, infected_strength_network, focal_node, date) for (focal_node, date) in healthy_nodelist])


##############################################################################
def return_healthy_nodelist(node_health1, seed_date, network_min_date):
	r""" healthy_nodelist is a list. Format = [(node1, day1, day2),...]
	where node1 is a node reported health and day1-day2 are the days
	when the node is uninfected"""
	
	healthy_nodelist = [(node, date1) for node in node_health1 if 0 in node_health1[node] for date1 in [date for (hd1, hd2) in node_health1[node][0] for date in range(hd1, hd2+1)] if date1!=seed_date and date1>network_min_date]
	
	return healthy_nodelist	
	
###############################################################################
def calculate_lambda1(beta1, epsilon1, infected_strength_network, focal_node, date):
	r""" This function calculates the infection potential of the 
	focal_node based on (a) its infected_strength at the previous time step (date-1),
	and (b) tranmission potential unexplained by the individual's network connections."""
	
	try:
		return 1-(np.exp(-(beta1*infected_strength_network[focal_node][date-1] + epsilon1)))
	except KeyError:
		print ("Could not calculate lambda for node and date", focal_node, date-1)
		


################################################################################
def calculate_infected_strength(node, time1, health_data_new, G):
	r""" This function calculates the infected strength of focal node = node 
	as the sum of the weighted edge connections of the node at time=time1. Only
	those nodes are considered that are reported as sick (= 1) at time1."""
	
	## infected strength is sum of all edge weights of focal nodes connecting to infected nodes
	## NOTE: health_data_new[node_i].get(time1) checks if time1 is present in health_data[node_i] AND if the value is 1
		

	if time1 in G and node in G[time1].nodes(): 
		strength = [G[time1][node][node_i]["weight"] for node_i in G[time1].neighbors(node) if (node_i in health_data_new and health_data_new[node_i].get(time1))]
		
		
	else: strength=[]
	return sum(strength)

################################################################################
def to_params(arr, null_comparison, diagnosis_lag, nsick_param, recovery_prob, parameter_estimate):
	r""" Converts a numpy array into a array with named fields"""
	
	# Note gamma is estimated only when there is a diagnosis lag
	if diagnosis_lag and recovery_prob: 
		if null_comparison: 
			arr2 = np.array(parameter_estimate+ list(arr))
			return arr2.view(np.dtype([('beta', np.float),
			('epsilon', np.float),
			('gamma', np.float, nsick_param),
			('diag_lag', np.float, nsick_param),
			('model', np.float)]))

		return arr.view(np.dtype([('beta', np.float),
			('epsilon', np.float),
			('gamma', np.float, nsick_param),
			('diag_lag', np.float, nsick_param)]))

	elif diagnosis_lag:
		if null_comparison: 
			arr2 = np.array(parameter_estimate+ list(arr))
			return arr2.view(np.dtype([('beta', np.float),
			('epsilon', np.float),
			('diag_lag', np.float, nsick_param),
			('model', np.float)])) 
	
		return arr.view(np.dtype([('beta', np.float),
			('epsilon', np.float),
			('diag_lag', np.float, nsick_param)])) 
	

	if null_comparison: 
			arr2 = np.array(parameter_estimate+ list(arr))
			return arr2.view(np.dtype([('beta', np.float), 
			('epsilon', np.float),
			('model', np.float)]))
			
	return arr.view(np.dtype([('beta', np.float), 
			('epsilon', np.float)]))
	
#############################################################################
def prior_transform(parameters):
    """Transforms our unit cube samples `u` to a flat prior between in each variable."""
    
    #min and max for beta and epsilon
    ##although beta and epsilon does not have an upper bound, specify an large upper bound to prevent runaway samplers
    aprime = np.array(parameters[0:2])
    amin = 0
    amax = 10
    
    ##min max for other param estimates
    bprime = np.array(parameters[2:])
    bmin = 0
    bmax = 1
    
    a = aprime*(amax-amin) + amin  # convert back to a
    b = bprime*(bmax-bmin) + bmin  # convert back to a
    
    return tuple(list(a)+list(b))

#############################################################################
def prior_transform_null(parameter):
    """Transforms our unit cube samples `u` to a flat prior between in each variable."""
    
    #min and max for beta and epsilon
    ##although beta and epsilon does not have an upper bound, specify an large upper bound to prevent runaway samplers
    aprime = parameter
    amin = 0.1
    amax = 1
    
    a = aprime*(amax-amin) + amin  # convert back to a
    
    return tuple(a)

#############################################################################
def prior_transform_alternate(parameter):
    """Transforms our unit cube samples `u` to a flat prior between in each variable."""
    
    #min and max for beta and epsilon
    ##although beta and epsilon does not have an upper bound, specify an large upper bound to prevent runaway samplers
    aprime = parameter
    amin = 0.0
    amax = 0.01
    
    
    a = aprime*(amax-amin) + amin  # convert back to a
    #print (parameter, a)
    return tuple(a)

#######################################################################
def start_sampler(data, recovery_prob, verbose,  contact_daylist, max_recovery_time, nsick_param, output_filename, diagnosis_lag=False, null_comparison=False,  **kwargs3):
	r"""Sampling performed using emcee """

	parameter_estimate=None
	##############################################################################
	
	G_raw, health_data, node_health, nodelist, true_value,  time_min, time_max, seed_date =data		
	######################################
	### Set number of parameters to estimate
	######################################
	ndim_base = 2
	if recovery_prob: ndim_base += nsick_param
	ndim = ndim_base+nsick_param
	
	################################################################################
	##calculating infection date and infection strength outside loglik to speed up #
	##computations
	################################################################################
	network_min_date = min(G_raw.keys())
	if not diagnosis_lag:
		######################################################################
		infection_date = [(node, time1) for node in node_health if 1 in node_health[node] for (time1,time2) in node_health[node][1]]
		
		## remove days in infection_date if the day is either the seed_date or before network_min_date
		infection_date = [(node, time1) for (node, time1) in infection_date if time1!=seed_date and time1 > network_min_date]
		infection_date = sorted(infection_date)
		
		######################################################################
		
		##for parameter estimate step we need data for the empirical network only
		infected_strength = {0:{node:{time: calculate_infected_strength(node, time, health_data, G_raw[0]) for time in range(time_min, time_max+1)} for node in nodelist}}
		
		
	else: 
		infection_date = None
		infected_strength=None	
		

	healthy_nodelist = return_healthy_nodelist(node_health, seed_date, network_min_date)
	################################################################################
	pool = Pool()
	if ndim<3:
		sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim=ndim,  pool=pool, queue_size=cpu_count()-1, use_pool={'propose_point': False}, logl_args =[data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, max_recovery_time, network_min_date, parameter_estimate] )
		sampler.run_nested(print_progress = verbose)
		
	
	else:
		thresh = 0.01
		maxc = 10000
		sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim=ndim, pool=pool, queue_size=cpu_count()-1, use_pool={'update_bound': False}, dlogz=thresh,logl_args =[data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, max_recovery_time, network_min_date, parameter_estimate])  
		ncall = sampler.ncall
		niter = sampler.it - 1
		for results in sampler.sample_initial(maxcall=maxc):
			ncall += results[9]
			niter += 1
			delta_logz = results[-1]
			#print('dlogz ' + str(delta_logz), 'thresh ' + str(thresh), 'nc ' + str(ncall), 'niter ' + str(niter), "log", results[3])
			pass
		
		stop, stop_vals = stopping_function(sampler.results, args = { 'post_thresh': 0.05}, return_vals=True)
		while True:
			stop, stop_vals = stopping_function(sampler.results, return_vals=True)  # evaluate stop
			if not stop:
			    logl_bounds = weight_function(sampler.results)  # derive bounds
			    for results in sampler.sample_batch(nlive_new = 50,logl_bounds=logl_bounds, maxcall=15000):
			        ncall += results[4]  # worst, ustar, vstar, loglstar, nc...
			        niter += 1
			        pass
			    sampler.combine_runs()  # add new samples to previous results
			 
			    break
			
	pool.close()
	pool.join()		    
	
	return sampler, ndim
#######################################################################
def perform_null_comparison(output_filename, data, recovery_prob,  verbose,  contact_daylist, max_recovery_time, nsick_param, diagnosis_lag=False, null_comparison=True, **kwargs3):
	
	G_raw, health_data, node_health, nodelist, true_value,  time_min, time_max, seed_date,parameter_estimate = data
	
	################################################################################
	##calculating infection date and infection strength outside loglik to speed up #
	##computations
	################################################################################
	network_min_date = min(G_raw.keys())
	if not diagnosis_lag:		
		infection_date = [(node, time1) for node in node_health if 1 in node_health[node] for (time1,time2) in node_health[node][1]]
		infection_date = [(node, time1) for (node, time1) in infection_date if time1!=seed_date and time1 > network_min_date]
		infection_date = sorted(infection_date)	
		infected_strength = {network:{node:{time: calculate_infected_strength(node, time, health_data, G_raw[network]) for time in range(time_min, time_max+1)} for node in nodelist} for network in G_raw}	
		
	else: 
		infection_date = None
		infected_strength=None	
		
		
	
	healthy_nodelist = return_healthy_nodelist(node_health, seed_date, network_min_date)
	##############################################################################
	
	sampler = dynesty.NestedSampler(log_likelihood, prior_transform_alternate, ndim=1,   logl_args =[data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, max_recovery_time, network_min_date, parameter_estimate], bound="none", walks=500)
	sampler.run_nested(print_progress=False, dlogz=5,maxcall=5000)
	
	dres = sampler.results
	samples = dres.samples #samples
	weights = np.exp(dres['logwt'] - dres['logz'][-1])  # normalized weight
	log_evidence_alternate = dres.logz[-1]        # value of logZ
	dlogZerrdynesty = dres.logzerr[-1]  # estimate of the statistcal uncertainty on logZ
	print ("evidence of alternate", log_evidence_alternate)
	
	##############################################################################
	sampler = dynesty.NestedSampler(log_likelihood, prior_transform_null, ndim=1,  logl_args =[data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, max_recovery_time, network_min_date, parameter_estimate], bound="none", walks=500)
	sampler.run_nested(print_progress=False,dlogz=5, maxcall=5000)
	
	dres = sampler.results
	samples = dres.samples #samples
	weights = np.exp(dres['logwt'] - dres['logz'][-1])  # normalized weight
	log_evidence_null = dres.logz[-1]        # value of logZ
	dlogZerrdynesty = dres.logzerr[-1]  # estimate of the statistcal uncertainty on logZ
	
	print ("log_evidence_null ", log_evidence_null )
	log_BF_alternate = log_evidence_alternate - log_evidence_null
	
	print ("Log Bayes factor of alternate vs null = ", log_BF_alternate)
	
	file1 = open(output_filename+ "_null_hypothesis_testing.txt", "w+")
	file1.write("Log marginal evidence for null hypothesis is " + str(round(log_evidence_null,2))+ "\n")
	file1.write("Log marginal evidence for alternate hypothesis is " + str(round(log_evidence_alternate,2))+ "\n")
	file1.write("Log Bayes Factor of alternate vs null = " + str(round(log_BF_alternate,2))+ "\n")

##############################################################################################
def summarize_sampler(sampler, G_raw, true_value, output_filename, nparam = None, corner_plot=True):
	r""" Summarize the results of the sampler"""

	
	summary_type = "parameter_estimate"	
	dres = sampler.results
	tf = open(output_filename+ "_parameter_summary.txt", "w+")

	samples = dres.samples #samples
	weights = np.exp(dres['logwt'] - dres['logz'][-1])  # normalized weights
	parameter_estimate = []
	for num in range(nparam):  # for each parameter
	    CI = dyfunc.quantile(dres['samples'][:, num], [0.025, 0.5, 0.975], weights=weights)
	    parameter_estimate.append(CI[1])
	    if num ==0:
	        print ("The median estimate and 95% credible interval for beta is " + str(round(CI[1],5))+" ["+ str(round(CI[0],5))+ "," + str(round(CI[2],5))+ "]")
	        tf.write("The median estimate and 95% credible interval for beta is " + str(round(CI[1],5))+" ["+ str(round(CI[0],5))+ "," + str(round(CI[2],5))+ "]\n")
	    elif num ==1:
	        print ("The median estimate and 95% credible interval for epsilon is " + str(round(CI[1],5))+" ["+ str(round(CI[0],5))+ "," + str(round(CI[2],5))+ "]")
	        tf.write("The median estimate and 95% credible interval for epsilon is " + str(round(CI[1],3))+" ["+ str(round(CI[0],5))+ "," + str(round(CI[2],5))+ "]\n")
	    else:
	        print ("Printing median and 95% credible interval for the rest of the unknown parameters")
	        print (str(round(CI[1],3))+" ["+ str(round(CI[0],3))+ "," + str(round(CI[2],3))+ "]")
	        tf.write("median and 95% credible interval for the rest of the unknown parameter #" +str(num)+"\n")
	        tf.write(str(round(CI[1],3))+" ["+ str(round(CI[0],3))+ "," + str(round(CI[2],3))+ "]\n")
	
	
	dlogZdynesty = dres.logz[-1]        # value of logZ
	dlogZerrdynesty = dres.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

	# output log marginal likelihood
	tf.write("Log marginalized evidence of the network hypothesis is = " + str(round(dlogZdynesty,3))+ "+/- "+ str(round(dlogZerrdynesty,3))+"\n")
	print('Log marginalised evidence (using dynamic sampler) is {} +/- {}'.format(round(dlogZdynesty,3), round(dlogZerrdynesty,3)))

	tf.close()
	
	if corner_plot:
		dpostsamples = resample_equal(samples, weights)
	
		fig = corner.corner(dpostsamples, labels=[r"$beta$", r"$epsilon$"], quantiles=[0.16, 0.5, 0.84],  truths= true_value, truth_color ="red" ,hist_kwargs={'density': True})
	
		fig.savefig(output_filename + "_" + summary_type +"_posterior.png")
	
	return parameter_estimate
	
######################################################################33
def run_inods_sampler(edge_filename, health_filename, output_filename, infection_type, truth = None, verbose=True, complete_nodelist = None, null_comparison=True,  edge_weights_to_binary=False, normalize_edge_weight=False, diagnosis_lag=False, is_network_dynamic=True, parameter_estimate=True, draw_corner_plot=True):
	r"""Main function for INoDS """
	
	
	###########################################################################
	##health_data is the raw dictionary. The structure of dictionary:         # 
	###health_data[node][timestep] = diagnosis                                #   
	## Node health is dictionary with primary key - node id, 		  # 
	## secondary key = infection status (0=healthy, 1=infected)               # 
	## node_health[node id][infection status] = tuple of (min, max) time      #
	## period when the node is in the infection status                        #
	###########################################################################
	#Can nodes recover?
	recovery_prob = nf.can_nodes_recover(infection_type)
	time_min = 0
	time_max = nf.extract_maxtime(edge_filename, health_filename)

	G_raw = {}
	## read in the dynamic network hypthosis (HA)
	nodelist = nf.extract_nodelist(edge_filename, health_filename)
	if complete_nodelist is None: complete_nodelist = nodelist
	G_raw[0] = nf.create_dynamic_network(edge_filename, complete_nodelist, edge_weights_to_binary, normalize_edge_weight, is_network_dynamic, time_max)
	
	
	
	health_data, node_health = nf.extract_health_data(health_filename, infection_type, nodelist, time_max, diagnosis_lag)
	#find the first time-period when an infection was reported 
	seed_date = nf.find_seed_date(node_health)

	
	contact_daylist = None
	max_recovery_time = None	
	nsick_param = 0
	
	##########################################################################
	if parameter_estimate:
	##Step 1: Estimate unknown parameters of network hypothesis HA.
		if diagnosis_lag:
			#Format: contact_daylist[network_type][(node, time1, time2)] =       
			## potential time-points when the node could have contract infection 
			contact_daylist = nf.return_contact_days_sick_nodes(node_health, seed_date, G_raw)
			nsick_param = len(contact_daylist[0])
		
		if recovery_prob: max_recovery_time = nf.return_potention_recovery_date(node_health, time_max)	

		true_value = truth
		data1 = [G_raw, health_data, node_health, nodelist, true_value,  time_min, time_max, seed_date]

		if verbose: print ("estimating model parameters.........................")
		start = time.time()
		sampler, nparameters = start_sampler(data1,  recovery_prob,  verbose,  contact_daylist, max_recovery_time, nsick_param, output_filename, diagnosis_lag = diagnosis_lag)
		
		
		parameter_summary = summarize_sampler(sampler, G_raw, true_value, output_filename, nparam = nparameters, corner_plot =  draw_corner_plot)
		if verbose: print ("time taken for parameter estimation (mins)===", (time.time() - start)/60.)
	#############################################################################
	if not parameter_estimate and sum(truth)==0:
		raise ValueError("Parameter estimate is set to False and no truth is supplied!")

	########################################################################
	##Step 2: Perform hypothesis testing by comparing HA against null networks
	if null_comparison:
		if not parameter_estimate: parameter_summary = truth
	
			
		#generate few for testing H_a
		for num in [round(a,2) for a in np.arange(0,0.02, 0.01)]:
			G_raw[num] = nf.permute_network(G_raw[0], num, complete_nodelist,network_dynamic = is_network_dynamic)
	
		for num in [round(a,2) for a in np.arange(0.1,1.01, 0.01)]:
			
			
			if verbose: print ("generating null network =", num)
			
			G_raw[num] = nf.permute_network(G_raw[0], num, complete_nodelist,network_dynamic = is_network_dynamic)
		
	
			
		true_value = truth
		data1 = [G_raw, health_data, node_health, nodelist, true_value, time_min, time_max, seed_date, parameter_summary]

		if diagnosis_lag:
			#Format: contact_daylist[network_type][(node, time1, time2)] =       
			## potential time-points when the node could have contract infection 
			contact_daylist = nf.return_contact_days_sick_nodes(node_health, seed_date, G_raw)
			nsick_param = len(contact_daylist[0])

		if recovery_prob: max_recovery_time = nf.return_potention_recovery_date(node_health, time_max)	
		if verbose: print ("comparing network hypothesis with null..........................")


		perform_null_comparison(output_filename, data1, recovery_prob,  verbose, contact_daylist, max_recovery_time, nsick_param, diagnosis_lag = diagnosis_lag)
		summary_type = "null_comparison"
		
	##############################################################################
	
######################################################################33
if __name__ == "__main__":
	
	print ("run the run_inods.py file")	
	
	
		
