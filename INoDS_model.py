import matplotlib
matplotlib.use('Agg')
import networkx as nx
import csv
import numpy as np
from numpy import ma
from emcee import PTSampler, autocorr
import corner
import copy
import matplotlib.pyplot as plt
import random as rnd
from multiprocess import Pool
import INoDS_convenience_functions as nf
import warnings
from scipy import signal
import scipy.stats as ss
import cPickle
import time
import pandas as pd
np.seterr(invalid='ignore')
np.seterr(divide='ignore')
warnings.simplefilter("ignore")
warnings.warn("deprecated", DeprecationWarning)
#######################################################################
def log_likelihood(parameters, data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, recovery_daylist, null_comparison_data):
	r"""Computes the log-likelihood of network given infection data """
	
	if null_comparison:
		G_raw, health_data, node_health, nodelist, truth, time_min, time_max, seed_date,parameter_estimate = data
		health_data_new = copy.deepcopy(health_data)
		node_health_new = copy.deepcopy(node_health)		
		p = to_params(parameters, null_comparison, diagnosis_lag, nsick_param, recovery_prob, null_comparison_data)
		network = int(ss.randint.ppf(p['model'][0], 0,  len(G_raw)))
		G = G_raw[network]
	else:
		G_raw, health_data, node_health, nodelist, truth, time_min, time_max, seed_date  = data
		health_data_new = copy.deepcopy(health_data)
		node_health_new = copy.deepcopy(node_health)
		p = to_params(parameters, null_comparison, diagnosis_lag, nsick_param, recovery_prob, null_comparison_data)
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
	
	######################################################################
	if diagnosis_lag:
		infection_date = []
		diag_list = p['diag_lag'][0]
		#ensure that all numbers in diag_lsit are between 0.000001 and 1
		diag_list = [min(max(num,0.000001),1) for num in diag_list]
		## iterate through focal node, infection time-period and diagnosis lag 
		for (node, time1, time2), diag_lag in zip(sorted(contact_daylist[network]), diag_list):
			## compute lagged time for each infection time based on diag_list
			lag = int(ss.randint.ppf(diag_lag, 0,  len(contact_daylist[network][(node, time1, time2)])))
			## pick out corresponding date from contact_daylist
			new_infection_time  = contact_daylist[network][(node, time1, time2)][lag]
			new_recovery_time = time2
			infection_date.append((node, new_infection_time))
			
			if recovery_prob!=np.inf:
				max_recovery_date = recovery_daylist[(node, time1, time2)]
				##uniform ppf is defined as (prob, lower_limite, upper_limit - lower_limit)
				new_recovery_time = int(ss.uniform.ppf(p['gamma'][0], new_infection_time, max_recovery_date-new_infection_time))
				## if the proposed recovery date coincides with a date when the indiv was reported sick then return invalid
				
				if new_recovery_time<= time2: return -np.inf
			node_health_new[node][1].remove((time1, time2))
			node_health_new[node][1].append((new_infection_time, new_recovery_time))	
			
			for day in range(new_infection_time, new_recovery_time+1): health_data_new[node][day]=1
	
			infected_strength={}
			infected_strength[network] = {node:{time: calculate_infected_strength(node, time, health_data_new, G) for time in G.keys()} for node in nodelist}

			healthy_nodelist = return_healthy_nodelist(node_health_new)
			infection_date = sorted(infection_date)
	######################################################################	

	################################################################
	##Calculate rate of learning for all sick nodes at all sick    #
	## dates, but not when sick day is the seed date (i.e., the    #
	## first  report of the infection in the network               #
	################################################################
	overall_learn = [np.log(calculate_lambda1(p['beta'][0], p['alpha'][0], infected_strength[network], focal_node, sick_day)) for (focal_node, sick_day) in infection_date if sick_day!=seed_date]
	#print ("overall learn"), [infected_strength[network][focal_node][sick_day-1] for (focal_node, sick_day) in infection_date if sick_day!=seed_date]
	################################################################
	##Calculate rate of NOT learning for all the days the node was #
	## (either reported or inferred) healthy                       #
	################################################################
	overall_not_learn = [not_learned_rate(focal_node,healthy_day1, healthy_day2, p['beta'][0],p['alpha'][0], infected_strength[network], seed_date, network_min_date) for (focal_node,healthy_day1, healthy_day2) in healthy_nodelist]	

	###########################################################
	## Calculate overall log likelihood                       #
	########################################################### 
	loglike = sum(overall_learn) + sum(overall_not_learn)
	if loglike == -np.inf or np.isnan(loglike):return -np.inf
	else: return loglike

#############################################################################
def not_learned_rate(focal_node, healthy_day1, healthy_day2, beta, alpha, infected_strength_network, seed_date, network_min_date):
	r""" Calculate 1- lambda for all uninfected days and returns 
	sum of log(1-lambdas)"""

	lambda_list = [1-calculate_lambda1(beta, alpha, infected_strength_network, focal_node, date) for date in [date1 for date1 in range(healthy_day1, healthy_day2+1) if date1!=seed_date and date1>network_min_date]]
	return sum([np.log(num) for num in lambda_list])

##############################################################################
def return_healthy_nodelist(node_health1):
	r""" healthy_nodelist is a list. Format = [(node1, day1, day2),...]
	where node1 is a node reported health and day1-day2 are the days
	when the node is uninfected"""
	
	healthy_nodelist = [(node, healthy_day1, healthy_day2) for node in node_health1 if node_health1[node].has_key(0) for healthy_day1, healthy_day2 in node_health1[node][0]]
	healthy_nodelist = sorted(healthy_nodelist)
	
	return healthy_nodelist	
	
###############################################################################
def calculate_lambda1(beta1, alpha1, infected_strength_network, focal_node, date):
	r""" This function calculates the infection potential of the 
	focal_node based on (a) its infected_strength at the previous time step (date-1),
	and (b) tranmission potential unexplained by the individual's network connections."""
	
	prob_not_infected = np.exp(-(beta1*infected_strength_network[focal_node][date-1] + alpha1))
	#avoid returning 1 which will lead lnlike to be -np.inf
	return min(1-prob_not_infected, 0.99999999)

################################################################################
def calculate_infected_strength(node, time1, health_data_new, G):
	r""" This function calculates the infected strength of focal node = node 
	as the sum of the weighted edge connections of the node at time=time1. Only
	those nodes are considered that are reported as sick (= 1) at time1."""
	
	## infected strength is sum of all edge weights of focal nodes connecting to infected nodes
	## NOTE: health_data_new[node_i].get(time1) checks if time1 is present in health_data[node_i] AND if the value is 1
	if node in G[time1].nodes(): strength = [G[time1][node][node_i]["weight"] for node_i in G[time1].neighbors(node) if health_data_new[node_i].get(time1)]
	else: strength=[]
	return sum(strength)

################################################################################
def to_params(arr, null_comparison, diagnosis_lag, nsick_param, recovery_prob, null_comparison_data):
	r""" Converts a numpy array into a array with named fields"""
		
	# Note gamma is estimated only when there is a diagnosis lag
	if diagnosis_lag and recovery_prob!=np.inf: 
		if null_comparison: 
			arr = np.array(null_comparison_data + list(arr))
			return arr.view(np.dtype([('beta', np.float),
			('alpha', np.float),
			('gamma', np.float),
			('diag_lag', np.float, nsick_param),
			('model', np.float)]))
		return arr.view(np.dtype([('beta', np.float),
			('alpha', np.float),
			('gamma', np.float),
			('diag_lag', np.float, nsick_param)]))

	elif diagnosis_lag:
		if null_comparison: 
			arr = np.array(null_comparison_data + list(arr))
			return arr.view(np.dtype([('beta', np.float),
			('alpha', np.float),
			('diag_lag', np.float, nsick_param),
			('model', np.float)])) 
	
		return arr.view(np.dtype([('beta', np.float),
			('alpha', np.float),
			('diag_lag', np.float, nsick_param)])) 
	

	if null_comparison: 
			arr = np.array(null_comparison_data + list(arr))
			return arr.view(np.dtype([('beta', np.float), 
			('alpha', np.float),
			('model', np.float)]))
			
	return arr.view(np.dtype([('beta', np.float), 
			('alpha', np.float)]))
	

#####################################################################
def autocor_checks(sampler, itemp=0, outfile=None):
	r""" Perform autocorrelation checks"""

	## sampler shape is (ntemp, nwalkers, niter, dim). Chain containing samples is niter*nwalkers
	#a_exp = sampler.acor[0]
	#a_exp = max(a_exp)
	#print('Additional burn-in required'), int(10 * a_exp)
	
	print('Chains contain samples after thinning (across all walkers)='), sampler.chain.shape[-2]*sampler.chain.shape[1]
	#a_int = np.max([autocorr.integrated_time(sampler.chain[itemp, i, :], c=5) for i in range(sampler.chain.shape[1])], 0)
	#a_int = max(a_int)
	#print('Each chain produces one independent sample per steps ='), int(a_int)
	print ('Gelman-Rubin covergence test passed?'), gelman_rubin(sampler.chain[0,:,:,:])
	

##############################################################
def gelman_rubin(chain_ensemble):
    """
    returns: Gelman-Rubin scale reduction factor for set of chains
    Parameters:
    chain_ensemble: the ensemble of chains to use; this paramater expects that the first nsteps/2 of each chain have already been discarded;
    chain_ensemble should be an array of the format nchains x nsteps x ndim
    code adapted from function https://github.com/catherinezucker/dustcurve/blob/7b018b47fd782878a2778b44756e32eafeae235f/dustcurve/diagnostics.py available under GNU open source license
    """

    nruns,nsteps,ndim=chain_ensemble.shape

    #calculate the mean of each chain
    mean=np.mean(chain_ensemble,axis=1)

    #calculate the variance of each chain
    var=np.var(chain_ensemble,axis=1, ddof=1)

    #calculate the mean of the variances of each chain
    # W is the average within-chain variance
    W=np.mean(var, axis=0)

    #calculate the variance of the chain means multiplied by n
    # B is between chain variance
    B=nsteps*np.var(mean, axis=0, ddof=1)
    #calculate estimated variance:
    sigmasq=(1-1/nsteps) * W + (1/nsteps)*B

    #sampling variability
    Vhat=sigmasq+B/(nruns*nsteps)

    #degree of freedo
    d = 2*Vhat**2/np.var(Vhat)

    #calculate the potential scale reduction factor
    Rhat = np.sqrt((d+3)*Vhat/((d+1)*W))

    return Rhat, all(num<1.1 for num in Rhat)
#################################################
def check_convergence_corr(sampler,  pval_threshold=0.05):
    
    # Only do the check if we've got enough steps
    if sampler.lnprobability is None:
        print "sampler.lnprobability is None."
        return

    lnpost = sampler.lnprobability[:, :, :]
    ntemps = lnpost.shape[0]
    nwalkers = lnpost.shape[1]
    nsteps = lnpost.shape[2]

    step_indices = np.repeat(np.arange(0, nsteps), nwalkers)
    pass_arr = np.zeros(ntemps)
    for temp_ix in range(0, ntemps):
        pts = lnpost[temp_ix].flatten(order='F')
        (rval, pval) = ss.pearsonr(step_indices, pts)
        if rval < 0 or pval > pval_threshold:
            pass_arr[temp_ix] = True
    passes = np.all(pass_arr)
    return np.all(pass_arr)

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

#####################################################################
def flatten_chain(sampler):
    r"""Flatten zero temperature chain"""

    c = sampler.chain
    c = c[0,:, :]
    ct = c.reshape((np.product(c.shape[:-1]), c.shape[-1]))
    return ct

#######################################################################
def summary(sampler):
    r"""Calculate mean and standard deviation of the sampler chains. Best parameter are chosed
	based on the maximum log probability. """
  
    #mean = samples.mean(0)
    #mean = [round(num,5) for num in mean]
    #sd = samples.std(0)
    #sd = [round(num,5) for num in sd]
  
    best_lnprob = np.unravel_index(sampler.lnlikelihood[0,:,:].argmax(), sampler.lnlikelihood[0,:,:].shape)
    best_pars = sampler.chain[0,best_lnprob[0],best_lnprob[1]]
    return best_pars

##############################################################################
def log_prior(parameters, priors, null_comparison, diagnosis_lag, nsick_param, recovery_prob, null_comparison_data):
    
    p = to_params(parameters, null_comparison, diagnosis_lag, nsick_param, recovery_prob, null_comparison_data)
    if null_comparison:
	if p['model'][0] < 0.000001  or  p['model'][0] >  1: return -np.inf 
	return 0

    else:
	if p['beta'][0] <  priors[0][0] or p['beta'][0] > priors[0][1]: return -np.inf
	if p['alpha'][0] < priors[1][0]  or  p['alpha'][0] >  priors[1][1]: return -np.inf 
	 
	if diagnosis_lag: 
		if (p['diag_lag'][0] < 0.000001).any() or (p['diag_lag'][0] > 1).any():return -np.inf
		if recovery_prob != np.inf:
			if p['gamma'][0] < recovery_prob[0]  or  p['gamma'][0] >  recovery_prob[1]: return -np.inf 
	#print ss.powerlaw.logpdf((1-p['alpha'][0]), 4)
    	return  ss.powerlaw.logpdf((1-p['alpha'][0]), 4)
#######################################################################
def start_sampler(data, recovery_prob, priors, niter, min_burnin, max_burnin, verbose,  contact_daylist, recovery_daylist, nsick_param, diagnosis_lag=False, null_comparison=False, **kwargs3):
	r"""Sampling performed using emcee """

	null_comparison_data=None
	##############################################################################
	if null_comparison: 
		G_raw, health_data, node_health, nodelist, true_value,  time_min, time_max, seed_date,parameter_estimate = data
		ndim = 1 
		betas = np.linspace(0, -0.04, 10)
		betas = 10**(np.sort(betas)[::-1])
		ntemps = 10
		nwalkers = max(10, 2*ndim) # number of MCMC walkers
		starting_guess = np.zeros((ntemps, nwalkers, ndim))
		starting_guess[:, :, 0] = np.random.uniform(low = 0.0001, high =1, size=(ntemps, nwalkers))
		null_comparison_data = parameter_estimate
		
	else: 
		G_raw, health_data, node_health, nodelist, true_value,  time_min, time_max, seed_date =data		
		######################################
		### Set number of parameters to estimate
		######################################
		ndim_base = 2
		if recovery_prob != np.inf: ndim_base += 1
		ndim = ndim_base+nsick_param
		
		####################### 
		##Adjust temperature ladder
		#######################
		betas = np.linspace(0, -2, 15)
		betas = 10**(np.sort(betas)[::-1])
		ntemps = 15
		
		########################################### 
		###set starting positions for the walker
		#############################################
		nwalkers = max(50, 2*ndim) # number of MCMC walkers
		starting_guess = np.zeros((ntemps, nwalkers, ndim))
		##starting guess for beta  
		starting_guess[:, :, 0] = np.random.uniform(low = priors[0][0], high = priors[0][1], size=(ntemps, nwalkers))
		##start alpha close to zero
		alphas = np.random.power(4, size = (ntemps, nwalkers))
		starting_guess[:, :, 1] = 1-alphas
		if diagnosis_lag:
			if recovery_prob != np.inf: starting_guess[:, :, 2] = np.random.uniform(low = recovery_prob[0], high = recovery_prob[1], size=(ntemps, nwalkers))
			starting_guess[:, :, ndim_base: ndim_base+nsick_param] = np.random.uniform(low = 0.001, high = 1,size=(ntemps, nwalkers, nsick_param))
		
		
	################################################################################
	##calculating infection date and infection strength outside loglik to speed up #
	##computations
	################################################################################
	if not diagnosis_lag:		
		infection_date = [(node, time1) for node in node_health if node_health[node].has_key(1) for (time1,time2) in node_health[node][1]]
		infection_date = sorted(infection_date)
		healthy_nodelist = return_healthy_nodelist(node_health)	
		infected_strength = {network:{node:{time: calculate_infected_strength(node, time, health_data, G_raw[network]) for time in G_raw[network].keys()} for node in nodelist} for network in G_raw}
			
		
	else: 
		infection_date = None
		infected_strength=None
		healthy_nodelist = None
	
	##############################################################################
	
	sampler = PTSampler(ntemps=ntemps, nwalkers=nwalkers, dim=ndim, betas=betas, logl=log_likelihood, logp=log_prior, a = 1.5,  loglargs=(data, infection_date, infected_strength, healthy_nodelist, null_comparison, diagnosis_lag,  recovery_prob, nsick_param, contact_daylist, recovery_daylist, null_comparison_data), logpargs=(priors, null_comparison, diagnosis_lag, nsick_param, recovery_prob, null_comparison_data)) 
	
	#Run user-specified burnin
	print ("burn in......")
	abs_tol = 3.0 # The maximum allowed difference for convergence
        rel_tol = 0.1 # The fraction of the error allowed for convergence

        # Run the sampler at incremental value of burnin period; at the end
        # of each round, check for TI convergence. If there is no substantial improvement in TI, go on to main
        # sampling. If not, reinitialize sampler and run again. 
	nburn = min_burnin
	last_ti = None
	done = False
	total_nburn = min_burnin
	while not done:
		for i, (p, lnprob, lnlike) in enumerate(sampler.sample(starting_guess, iterations = nburn)): 
			if verbose:print("burnin progress and time"), (100 * float(i) / nburn), end-start
			else: pass
	
		if last_ti is None:
                	(last_ti, last_ti_err) = sampler.thermodynamic_integration_log_evidence()
			nburn=10
			total_nburn+=nburn
		else: 
			(curr_ti, curr_ti_err) = sampler.thermodynamic_integration_log_evidence()
			#compute difference in evidence
			diff = np.abs(last_ti - curr_ti)
			
			#check for convergence
			if diff<abs_tol and curr_ti_err < abs_tol and last_ti_err < abs_tol and diff < (last_ti_err * rel_tol) and check_convergence_corr(sampler, pval_threshold=0.001): 
				print ("total burnin steps = "), total_nburn
				done = True
			else:
				last_ti = curr_ti
				last_ti_err = curr_ti_err
				nburn=10
				total_nburn+=nburn

			if total_nburn >= max_burnin:
				print ("Exceeded maximum iterations. Convergence not acheived")
				return sampler, False
			
		cur_start_position = p
		sampler.reset()

	
	#################################
	print ("sampling........")
	nthin = 5
	for i, (p, lnprob, lnlike) in enumerate(sampler.sample(p, lnprob0 = lnprob,  lnlike0= lnlike, iterations= niter, thin= nthin)):  
		if verbose:print("sampling progress"), (100 * float(i) / niter)
		else: pass
	
	#######################################
	#checks for model evidence
	#mean_logls = np.mean(sampler.lnlikelihood.reshape((ntemps, -1)), axis=1)
	#betas = sampler.betas
	#betas = [np.log(beta) for beta in betas]
	#plt.plot(betas, mean_logls, "-o") 
 	#plt.xlabel(r'$\beta$')
	#plt.ylabel(r'$\beta \left\langle \ln L \right\rangle_\beta$')
 	#plt.title('Thermodynamic Integration Integrand')
	#plt.savefig("plot.png")
	
	#logls = sampler.lnlikelihood
	#logls = logls[:, :, 0:]  # Drop the burn-in steps 

	#mean_logls2 = np.mean(np.mean(logls, axis=1)[:, 0:], axis=1)
	#lnZ_RRS = -np.dot(mean_logls2, np.diff(np.concatenate((betas, np.array([0])))))

	
	#logls_ave = np.mean(logls, axis=1)  # Average over the steps 
	#logls_ave = np.mean(logls_ave, axis=1) # Average over the walkers
	#lnZ_trap = -np.trapz(logls_ave, betas)
	#print ("RRS lnZ = "),lnZ_RRS
	#print ("trap lnZ="), lnZ_trap
	#########################################################
		
	##############################
	#The resulting samples are stored as the sampler.chain property:
	assert sampler.chain.shape == (ntemps, nwalkers, niter/nthin, ndim)

	return sampler, True
##############################################3
def getstate(sampler):
        self_dict = sampler.__dict__.copy()
        del self_dict['pool']
	return self_dict

##############################################################################################
def summarize_sampler(sampler, G_raw, true_value, output_filename, summary_type, recovery_prob):
	r""" Summarize the results of the sampler"""

	if summary_type =="parameter_estimate":
		samples = flatten_chain(sampler)
		best_par = summary(sampler)
		print ("parameter estimate of network hypothesis"), best_par
		cPickle.dump(getstate(sampler), open( output_filename + "_" + summary_type +  ".p", "wb" ), protocol=2)
			
		if recovery_prob!=np.inf:
			fig = corner.corner(sampler.flatchain[0, :, 0:3], quantiles=[0.16, 0.5, 0.84], labels=["$beta$", "$alpha$", "$rho$"], truths= true_value, truth_color ="red")
		else:
			fig = corner.corner(sampler.flatchain[0, :, 0:2], quantiles=[0.16, 0.5, 0.84], labels=["$beta$", "$alpha$"], truths= true_value, truth_color ="red")
			
		fig.savefig(output_filename + "_" + summary_type +"_posterior.png")
		nf.plot_beta_results(sampler, true_value[0], filename = output_filename + "_" + summary_type +"_beta_walkers.png" )
		logz, logzerr = log_evidence(sampler)
		evidence = np.exp(logz)
		error = evidence*logzerr
		print ("Log Bayes evidence and error"), logz, logzerr
		print ("Transformed evidence and error"), evidence, error


			
	autocor_checks(sampler)
	#except: print ("Warning!! Autocorrelation checks could not be performed. Sampler chains may be too short")        
	
	
	#################################
	if summary_type =="null_comparison":
		N_networks = len(G_raw)
		sampler1 = sampler.flatchain[0, :, 0]
		bins = [0]+[ss.randint.cdf(num, 0, N_networks) for num in xrange(N_networks)]
		hist = np.histogram(sampler1, bins)[0]
		df = pd.DataFrame(hist)
		file_name = output_filename + "_" + summary_type +  ".csv"
		df.to_csv(file_name)
		ha = hist[0]
		nulls = hist[1:]
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
		plt.hist(hist[1:], bins=10, normed=True, color="#969696")
		plt.axvline(x=hist[0], ymin=0, ymax=max(hist), linewidth=2, color='#e41a1c', label ="Network hypothesis")
		plt.xlabel("Predictive power", **axis_font)
		plt.ylabel("Frequency", **axis_font)
		plt.legend()
		plt.legend(frameon=False)
		plt.savefig(output_filename + "_" + summary_type +"_posterior.png")
		
	

#######################################################################
def find_aggregate_timestep(health_data):
	r"""Returns the timestep where all infection status are reported """

	timelist = []
	for node in health_data.keys():
		for time in health_data[node].keys(): timelist.append(time)

	if len(list(set(timelist)))!=1: print("Place error message here")
	return list(set(timelist))[0]
	
######################################################################33
def run_inods_sampler(edge_filename, health_filename, output_filename, infection_type,  recovery_prob, truth, null_networks, priors,  iteration, min_burnin=50, max_burnin=5000, verbose=True, null_comparison=False,  edge_weights_to_binary=False, normalize_edge_weight=False, diagnosis_lag=False, is_network_dynamic=True, parameter_estimate=True):
	r"""Main function for INoDS """
	
	###########################################################################
	##health_data is the raw dictionary. The structure of dictionary:         # 
	###health_data[node][timestep] = diagnosis                                #   
	## Node health is dictionary with primary key - node id, 		  # 
	## secondary key = infection status (0=healthy, 1=infected)               # 
	## node_health[node id][infection status] = tuple of (min, max) time      #
	## period when the node is in the infection status                        #
	###########################################################################
	nodelist = nf.extract_nodelist(edge_filename)
	time_min = 0
	time_max = nf.extract_maxtime(edge_filename, health_filename)
	health_data, node_health = nf.extract_health_data(health_filename, infection_type, nodelist, time_max, diagnosis_lag)
	
	#find the first time-period when an infection was reported 
	seed_date = nf.find_seed_date(node_health)

	G_raw = {}
	## read in the dynamic network hypthosis (HA)
	G_raw[0] = nf.create_dynamic_network(edge_filename,  edge_weights_to_binary, normalize_edge_weight, is_network_dynamic, time_max)
	contact_daylist = None
	recovery_daylist = None	
	nsick_param = 0
	
	###############################################################
	##if infection diagnosis is lagged, then true infection day is#
	## inferred using infectious contact history of focal node    # 
	###############################################################
	if diagnosis_lag:
		########################################################################
		## contact daylist is a dictionary                                     #  
		##Format: contact_daylist[network_type][(node, time1, time2)] =        #
		## potential time-points when the node could have contract infection   #  
		########################################################################
		contact_daylist = nf.return_contact_days_sick_nodes(node_health, seed_date, G_raw)
		nsick_param = len(contact_daylist[0])
		########################################################################
		## recovery daylist is a dictionary.                                   #  
		##Format: recovery_daylist[(node, time1, time2)] = recovery_date       #  
		## where node=focal node, time1:time2 = time-period of being infected. # 
		## recovery date = maximum time-point of node recovery                 #
		########################################################################
		if recovery_prob!=np.inf: recovery_daylist = nf.return_potention_recovery_date(node_health, time_max, G_raw)	
	
	##########################################################################
	if parameter_estimate:
	##Step 1: Estimate unknown parameters of network hypothesis HA.
		true_value = truth[:-1]
		data1 = [G_raw, health_data, node_health, nodelist, true_value,  time_min, time_max, seed_date]
		print ("estimating model parameters.........................")
		sampler, is_converged  = start_sampler(data1,  recovery_prob, priors,  iteration, min_burnin, max_burnin, verbose,  contact_daylist, recovery_daylist, nsick_param, diagnosis_lag = diagnosis_lag,null_comparison=False)
		summary_type = "parameter_estimate"
		if is_converged: summarize_sampler(sampler, G_raw, true_value, output_filename, summary_type, recovery_prob)
	#############################################################################
	if not parameter_estimate and sum(truth)==0:
		raise ValueError("Parameter estimate is set to False and no truth is supplied!")

	########################################################################
	##Step 2: Perform hypothesis testing by comparing HA against null networks
	if null_comparison:
		if parameter_estimate:
			samples = flatten_chain(sampler)
			parameter_estimate =  summary(sampler)
		else:
			parameter_estimate = truth
		print ("generating null graphs.......")
		for num in xrange(null_networks): 
			if verbose: print ("generating null network="), num
			G_raw[num+1] = nf.randomize_network(G_raw[0])
		true_value = truth
		data1 = [G_raw, health_data, node_health, nodelist, true_value, time_min, time_max, seed_date, parameter_estimate]
		print ("comparing network hypothesis with null..........................")
		#reset sampler
		sampler, is_converged = start_sampler(data1, recovery_prob, priors,  iteration, min_burnin, max_burnin,  verbose, contact_daylist, recovery_daylist, nsick_param, diagnosis_lag = diagnosis_lag, null_comparison=True, null_networks=null_networks)
		summary_type = "null_comparison"
		summarize_sampler(sampler, G_raw, true_value, output_filename, summary_type, recovery_prob)
	##############################################################################

	
	
		
######################################################################33
if __name__ == "__main__":

	print ("run the run_inods.py file")

	
