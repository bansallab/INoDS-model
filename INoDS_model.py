import numpy as np
import matplotlib
matplotlib.use('Agg')
import networkx as nx
import csv
import emcee
from numpy import ma
from emcee import autocorr
from emcee import EnsembleSampler
import corner
import copy
import matplotlib.pyplot as plt
import random as rnd
from multiprocess import Pool
import INoDS_convenience_functions as nf
import warnings
import scipy.stats as ss
import cPickle
import time
import itertools
import pandas as pd
np.seterr(invalid='ignore')
np.seterr(divide='ignore')
warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
#########################################################################
def diagnosis_adjustment(G, network, p, nodelist,contact_daylist,  recovery_prob, max_recovery_time, node_health_new, health_data_new):


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

	healthy_nodelist = return_healthy_nodelist(node_health_new)

	#create infection date list
	infection_date = [(node, new_time1) for (node, time1, time2, new_time1, new_time2) in new_infect_recovery_time]
	infection_date = sorted(infection_date)	
	
	return infected_strength, healthy_nodelist, infection_date

#######################################################################
def log_likelihood(parameters, data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, max_recovery_time, parameter_estimate):
	r"""Computes the log-likelihood of network given infection data """
	
	if null_comparison:
		G_raw, health_data, node_health, nodelist, truth, time_min, time_max, seed_date,parameter_estimate = data
		health_data_new = copy.deepcopy(health_data)
		node_health_new = copy.deepcopy(node_health)		
		p = to_params(parameters, null_comparison, diagnosis_lag, nsick_param, recovery_prob, parameter_estimate)
		network = int(p['model'][0])
		G = G_raw[network]
	else:
		G_raw, health_data, node_health, nodelist, truth, time_min, time_max, seed_date  = data
		health_data_new = copy.deepcopy(health_data)
		node_health_new = copy.deepcopy(node_health)
		p = to_params(parameters, null_comparison, diagnosis_lag, nsick_param, recovery_prob, parameter_estimate)
		network=0 
		G= G_raw[0]

	network_min_date = min(G.keys())
	###############################################################################################
	##diagnosis lag==
	##impute true infection date and recovery date (if SIR/SIS...)
	## infection_date = date picked as a day between last healthy report and first sick report
	## and when the degree of node was >0 the previous day
	##recovery_date = date picked as day with uniform probability between first reported sick day and first 
	##healthy date after sick report
	##################################################################################################
	
	if diagnosis_lag:
		infected_strength, healthy_nodelist, infection_date = diagnosis_adjustment(G, network, p, nodelist, contact_daylist, recovery_prob, max_recovery_time, node_health_new, health_data_new)
		
	######################################################################	

	################################################################
	##Calculate rate of learning for all sick nodes at all sick    #
	## dates, but not when sick day is the seed date (i.e., the    #
	## first  report of the infection in the network               #
	################################################################
	overall_learn = [np.log(calculate_lambda1(p['beta'][0], p['epsilon'][0], infected_strength[network], focal_node, sick_day)) for (focal_node, sick_day) in infection_date if sick_day!=seed_date and sick_day > network_min_date]
	################################################################
	##Calculate rate of NOT learning for all the days the node was #
	## (either reported or inferred) healthy                       #
	################################################################
	overall_not_learn = [not_learned_rate(focal_node,healthy_day1, healthy_day2, p['beta'][0],p['epsilon'][0], infected_strength[network], seed_date, network_min_date) for (focal_node,healthy_day1, healthy_day2) in healthy_nodelist]
	###########################################################
	## Calculate overall log likelihood                       #
	########################################################### 
	loglike = sum(overall_learn) + sum(overall_not_learn)
	
	if loglike == -np.inf or np.isnan(loglike) or (sum(overall_learn) + sum(overall_not_learn)==0):return -np.inf
	else: return loglike

#############################################################################
def not_learned_rate(focal_node, healthy_day1, healthy_day2, beta, epsilon, infected_strength_network, seed_date, network_min_date):
	r""" Calculate 1- lambda for all uninfected days and returns 
	sum of log(1-lambdas)"""

	lambda_list = [1-calculate_lambda1(beta, epsilon, infected_strength_network, focal_node, date) for date in [date1 for date1 in range(healthy_day1, healthy_day2+1) if date1!=seed_date and date1>network_min_date]]
	return sum([np.log(num) for num in list(lambda_list)])

##############################################################################
def return_healthy_nodelist(node_health1):
	r""" healthy_nodelist is a list. Format = [(node1, day1, day2),...]
	where node1 is a node reported health and day1-day2 are the days
	when the node is uninfected"""
	
	healthy_nodelist = [(node, healthy_day1, healthy_day2) for node in node_health1 if node_health1[node].has_key(0) for healthy_day1, healthy_day2 in node_health1[node][0]]
	healthy_nodelist = sorted(list(healthy_nodelist))
	
	return healthy_nodelist	
	
###############################################################################
def calculate_lambda1(beta1, epsilon1, infected_strength_network, focal_node, date):
	r""" This function calculates the infection potential of the 
	focal_node based on (a) its infected_strength at the previous time step (date-1),
	and (b) tranmission potential unexplained by the individual's network connections."""
	
	prob_not_infected = np.exp(-(beta1*infected_strength_network[focal_node][date-1] + epsilon1))
	#avoid returning 1 which will lead lnlike to be -np.inf
	return min(1-prob_not_infected, 0.99999999)


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
			arr2 = np.array(list(parameter_estimate) + list(arr))
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
			arr2 = np.array(list(parameter_estimate) + list(arr))
			return arr2.view(np.dtype([('beta', np.float),
			('epsilon', np.float),
			('diag_lag', np.float, nsick_param),
			('model', np.float)])) 
	
		return arr.view(np.dtype([('beta', np.float),
			('epsilon', np.float),
			('diag_lag', np.float, nsick_param)])) 
	

	if null_comparison: 
			arr2 = np.array(list(parameter_estimate) + list(arr))
			return arr2.view(np.dtype([('beta', np.float), 
			('epsilon', np.float),
			('model', np.float)]))
			
	return arr.view(np.dtype([('beta', np.float), 
			('epsilon', np.float)]))
	

#####################################################################
def autocor_checks(sampler, output_filename):
	r""" Perform autocorrelation checks"""

	print('Chains contain samples after thinning (= 5) across all walkers ='), sampler.chain.shape[-2]*sampler.chain.shape[1]
	f = [autocorr.function(sampler.chain[i, :])  for i in range(3)]

	ax = plt.figure().add_subplot(111)
	for i in range(3):
		ax.plot(f[i], "k")
		ax.axhline(0, color="k")
		ax.set_xlabel(r"Steps (after thinning)")
		ax.set_title(r"Autocorrelation of three walkers")
		ax.set_ylabel(r"Autocorrelation")
		plt.savefig(output_filename+'_autocorrelation.png')

#####################################################################
def log_evidence(sampler):
	r""" Calculate log evidence and error"""

	logls = sampler.lnlikelihood[:, :, :]
	logls = ma.masked_array(logls, mask=logls == -np.inf)
	mean_logls = logls.mean(axis=-1).mean(axis=-1)
	logZ = -np.trapz(mean_logls, sampler.betas)
	logZ2 = -np.trapz(mean_logls[::2], sampler.betas[::2])
	logZerr = abs(logZ2 - logZ)
	return logZ, logZerr

#######################################################################
def summary(sampler):
    r"""Calculate mean and standard deviation of the sampler chains. """
  
  
    ndim = sampler.chain.shape[-1]
    CI = np.empty([ndim, 3])
  
    post_samples = sampler.chain[:,:,:].reshape((-1, ndim))
    
    for num in range(ndim):
        CI[num] = np.percentile(post_samples[:,num], [2.5, 50, 97.5])

    
    return CI

##############################################################################
def log_prior(parameters,  diagnosis_lag, nsick_param, recovery_prob, parameter_estimate):
    
    null_comparison = False
    p = to_params(parameters, null_comparison, diagnosis_lag, nsick_param, recovery_prob, parameter_estimate)
    ##although beta does not have an upper bound, specify an large upper bound to prevent runaway walkers
    if p['beta'][0] <  0 or  p['beta'][0] >  1000: return -np.inf
    if p['epsilon'][0] < 0 or  p['epsilon'][0] > 1000: return -np.inf
    
    if diagnosis_lag:
        if (p['diag_lag'][0] < 0.000001).any() or (p['diag_lag'][0] > 1).any():return -np.inf
        if recovery_prob:
           if (p['gamma'][0] < 0.000001).any() or (p['gamma'][0] > 1).any():return -np.inf 
    return 0

##############################################################################
def log_posterior(parameters, data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, max_recovery_time, parameter_estimate):

	lnprior =  log_prior(parameters,  diagnosis_lag, nsick_param, recovery_prob, parameter_estimate)
	lnlik = log_likelihood(parameters, data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, max_recovery_time, parameter_estimate)
	
	return lnprior + lnlik

#######################################################################
def start_sampler(data, recovery_prob,  burnin, niter, verbose,  contact_daylist, max_recovery_time, nsick_param, diagnosis_lag=False, null_comparison=False,  **kwargs3):
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
	
	########################################### 
	###set starting positions for the walker
	#############################################
	nwalkers = max(50, 2*ndim) # number of MCMC walkers
	starting_guess = np.zeros((nwalkers, ndim))
	##starting guess for beta  
	starting_guess[ :, 0] = np.random.uniform(low = 0, high = 10, size=nwalkers)
	##start epsilon close to zero
	epsilons = np.random.power(4, size = nwalkers)
	starting_guess[:, 1] = 1-epsilons
	if diagnosis_lag:
		starting_guess[:, 2: ] = np.random.uniform(low = 0.001, high = 1,size=(nwalkers, ndim-2))
		
		
	################################################################################
	##calculating infection date and infection strength outside loglik to speed up #
	##computations
	################################################################################
	if not diagnosis_lag:		
		infection_date = [(node, time1) for node in node_health if node_health[node].has_key(1) for (time1,time2) in node_health[node][1]]
		infection_date = sorted(infection_date)
		##for parameter estimate step we need data for the empirical network only
		infected_strength = {0:{node:{time: calculate_infected_strength(node, time, health_data, G_raw[0]) for time in range(time_min, time_max+1)} for node in nodelist}}
		pool = None
		threads = 1
		
		
	else: 
		infection_date = None
		infected_strength=None	
		threads = 8
		

	healthy_nodelist = return_healthy_nodelist(node_health)
	################################################################################
	if threads>1:
		
		sampler = EnsembleSampler(nwalkers=nwalkers, dim=ndim, lnpostfn = log_posterior, a=2.0, args = [data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, max_recovery_time, parameter_estimate], threads = threads) 

	if threads<=1:
		sampler = EnsembleSampler(nwalkers=nwalkers, dim=ndim, lnpostfn = log_posterior, a=2.0, args = [data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, max_recovery_time, parameter_estimate])


	#Run user-specified burnin
	print ("burn in......")
	for i, (p, lnprob, lnlike) in enumerate(sampler.sample(starting_guess, iterations = burnin)): 
		if verbose:
			print("burnin progress..."), (100 * float(i) / burnin)

			
		else: pass

	sampler.reset()
	#################################
	print ("sampling........")
	nthin = 2
	for i, (p, lnprob, lnlike) in enumerate(sampler.sample(p, lnprob0 = lnprob,  iterations= niter, thin= nthin)):
		if verbose:print("sampling progress"), (100 * float(i) / niter)
		else: pass


	try:
		print ("autocorrelation time"), np.mean(sampler.get_autocorr_time())
	except:
		print ("could not estimate autocorrelation")
	##############################
	#The resulting samples are stored as the sampler.chain property:
	assert sampler.chain.shape == (nwalkers, niter/nthin, ndim)
	return sampler, ndim
#######################################################################
def perform_null_comparison(data, recovery_prob,  burnin, niter, verbose,  contact_daylist, max_recovery_time, nsick_param, diagnosis_lag=False, null_comparison=True, **kwargs3):
	r"""Sampling performed using emcee """

	G_raw, health_data, node_health, nodelist, true_value,  time_min, time_max, seed_date,parameter_estimate = data

	################################################################################
	##calculating infection date and infection strength outside loglik to speed up #
	##computations
	################################################################################
	if not diagnosis_lag:		
		infection_date = [(node, time1) for node in node_health if node_health[node].has_key(1) for (time1,time2) in node_health[node][1]]
		infection_date = sorted(infection_date)
		infected_strength = {network:{node:{time: calculate_infected_strength(node, time, health_data, G_raw[network]) for time in range(time_min, time_max+1)} for node in nodelist} for network in G_raw}
		pool = None
		threads = 1
		
		
	else: 
		infection_date = None
		infected_strength=None	
		threads = 8
		

	healthy_nodelist = return_healthy_nodelist(node_health)	
	##############################################################################
	
	logl_list = []
	for network in G_raw:
		logl = log_likelihood(np.array([network]), data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, max_recovery_time, parameter_estimate)	
		
		logl_list.append(logl)
	
	return logl_list

##############################################3
def getstate(sampler):
        self_dict = sampler.__dict__.copy()
        del self_dict['pool']
        return self_dict

######################################################
def calculate_BIC(sampler, G_raw, network, nparam):
	
	best_lglike= max(sampler.flatlnprobability)
	#print ("best log likelihood"), best_lglike
	highest_prob = np.argmax(sampler.lnprobability)
	hp_loc = np.unravel_index(highest_prob, sampler.lnprobability.shape)
	#print ("best param values"), sampler.chain[hp_loc[0], hp_loc[1]]
	N = 0
	for time in G_raw[network]:
		N+= len(G_raw[network][time].nodes)
	BIC = np.log(N)*nparam - 2*(best_lglike)
	return BIC

##############################################################################################
def summarize_sampler(sampler, G_raw, true_value, output_filename, summary_type, nparam = None):
	r""" Summarize the results of the sampler"""

	if summary_type =="parameter_estimate":
	
		CI = summary(sampler)
		tf = open(output_filename+ "_parameter_summary.txt", "w+")
		for num in xrange(CI.shape[0]):
			if num ==0:
				print ("The median estimate and 95% credible interval for beta is " + str(round(CI[0,1],3))+" ["+ str(round(CI[0,0],3))+ "," + str(round(CI[0,2],3))+ "]")
				tf.write("The median estimate and 95% credible interval for beta is " + str(round(CI[0,1],3))+" ["+ str(round(CI[0,0],3))+ "," + str(round(CI[0,2],3))+ "]\n")
			elif num ==1:
				print ("The median estimate and 95% credible interval for epsilon is " + str(round(CI[1,1],3))+" ["+ str(round(CI[1,0],3))+ "," + str(round(CI[1,2],3))+ "]")
				tf.write("The median estimate and 95% credible interval for epsilon is " + str(round(CI[1,1],3))+" ["+ str(round(CI[1,0],3))+ "," + str(round(CI[1,2],3))+ "]\n")
			else:
				print ("Printing median and 95% credible interval for the rest of the unknown parameters")
				print (str(round(CI[num,1],3))+" ["+ str(round(CI[num,0],3))+ "," + str(round(CI[num,2],3))+ "]")
				tf.write("median and 95% credible interval for the rest of the unknown parameter #" +str(num)+"\n")
				tf.write(str(round(CI[num,1],3))+" ["+ str(round(CI[num,0],3))+ "," + str(round(CI[num,2],3))+ "]\n")
		tf.close()
				
			
		fig = corner.corner(sampler.flatchain[:, 0:2], quantiles=[0.16, 0.5, 0.84], labels=["$beta$", "$epsilon$"], truths= true_value, truth_color ="red")
			
		fig.savefig(output_filename + "_" + summary_type +"_posterior.png")
		nf.plot_beta_results(sampler, filename = output_filename + "_" + summary_type +"_beta_walkers.png" )
		bic  = calculate_BIC(sampler, G_raw, 0, nparam)
		print ("BIC ===="), bic
		autocor_checks(sampler, output_filename)
		#cPickle.dump(getstate(sampler), open( output_filename + "_" + summary_type +  ".p", "wb" ), protocol=2)
		return CI

	#################################
	if summary_type =="null_comparison":
		best_par = None
		N_networks = len(G_raw)
		sampler_null = sampler[1:]
		df = pd.DataFrame(sampler)
		file_name = output_filename + "_" + summary_type +  ".csv"
		df.to_csv(file_name)
		if N_networks>1:
			ha = sampler[0]
			nulls = sampler[1:]
			ext_val = [int(num>=ha) for num in nulls]
			print ("p-value of network hypothesis"), sum(ext_val)/(1.*len(ext_val))
			ind = [num for num in xrange(N_networks)]
			
			########pretty matplotlib figure format
			axis_font = {'fontname':'Arial', 'size':'16'}
			plt.clf()
			plt.figure(figsize=(8, 10))    
			plt.gca().spines["top"].set_visible(False)    
			plt.gca().spines["right"].set_visible(False)    
			plt.gca().get_xaxis().tick_bottom()    
			plt.gca().get_yaxis().tick_left()   
			##############################
			hist = np.histogram(sampler, 10)[0]
			plt.hist(sampler[1:], bins=10, normed=False, color="#969696")
			plt.axvline(x=sampler[0], ymin=0, ymax=max(hist), linewidth=2, color='#e41a1c', label ="Network hypothesis")
			plt.xlabel("Log-likelihood", **axis_font)
			plt.ylabel("Frequency", **axis_font)
			plt.legend()
			plt.legend(frameon=False)
			plt.savefig(output_filename + "_" + summary_type +"_posterior.png")
		
	
######################################################################33
def run_inods_sampler(edge_filename, health_filename, output_filename, infection_type,  null_networks = 500, burnin =1000, iteration=2000, truth = None, verbose=True, complete_nodelist = None, null_comparison=False,  edge_weights_to_binary=False, normalize_edge_weight=False, diagnosis_lag=False, is_network_dynamic=True, parameter_estimate=True):
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
	G_raw[0] = nf.create_dynamic_network(edge_filename, complete_nodelist, edge_weights_to_binary, normalize_edge_weight, is_network_dynamic, time_max)
	nodelist = nf.extract_nodelist(G_raw[0])
	
	
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

		print ("estimating model parameters.........................")
		start = time.time()
		sampler, nparameters = start_sampler(data1,  recovery_prob,  burnin, iteration, verbose,  contact_daylist, max_recovery_time, nsick_param, diagnosis_lag = diagnosis_lag)
		summary_type = "parameter_estimate"
		CI = summarize_sampler(sampler, G_raw, true_value, output_filename, summary_type, nparam = nparameters)
		best_par = np.array([CI[num,1] for num in xrange(CI.shape[0])])
		print ("time==="), time.time() - start
	#############################################################################
	if not parameter_estimate and sum(truth)==0:
		raise ValueError("Parameter estimate is set to False and no truth is supplied!")

	########################################################################
	##Step 2: Perform hypothesis testing by comparing HA against null networks
	if null_comparison:
		if parameter_estimate:
			
			CI =  summary(sampler)
			parameter_estimate = [CI[0][1], CI[1][1]]
		else:
			parameter_estimate = truth

		if isinstance(null_networks, dict):
			for (num,val) in enumerate(null_networks):
				G_raw[num+1] = null_networks[val]	
			
			
		if isinstance(null_networks, int):
			print ("generating null graphs.......")
			jaccard_list =[]
			for num in xrange(null_networks): 
				if verbose: print ("generating null network ="), num
				G_raw[num+1], jaccard = nf.randomize_network(G_raw[0], complete_nodelist,network_dynamic = is_network_dynamic)
				jaccard_list.append(jaccard)
			if np.mean(jaccard_list)>0.4: 
				print ("Warning!! Randomized network resembles empircal network. May lead to inconsistent evidence")
		
		true_value = truth
		data1 = [G_raw, health_data, node_health, nodelist, true_value, time_min, time_max, seed_date, parameter_estimate]

		if diagnosis_lag:
			#Format: contact_daylist[network_type][(node, time1, time2)] =       
			## potential time-points when the node could have contract infection 
			contact_daylist = nf.return_contact_days_sick_nodes(node_health, seed_date, G_raw)
			nsick_param = len(contact_daylist[0])

		if recovery_prob: max_recovery_time = nf.return_potention_recovery_date(node_health, time_max)	
		print ("comparing network hypothesis with null..........................")


	
		logl_list = perform_null_comparison(data1, recovery_prob, burnin,  iteration,  verbose, contact_daylist, max_recovery_time, nsick_param, diagnosis_lag = diagnosis_lag, null_networks=null_networks)
		summary_type = "null_comparison"
		summarize_sampler(logl_list, G_raw, true_value, output_filename, summary_type)
	##############################################################################


######################################################################33
if __name__ == "__main__":

	print ("run the run_inods.py file")

	
