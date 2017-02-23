import csv
import networkx as nx
import numpy as np
import scipy.stats as ss
from random import shuffle
import matplotlib.pyplot as plt
import pandas as pd
#################################################################################
def create_dynamic_network(edge_filename, normalize_edge_weight, is_network_dynamic, time_max):

	
	df = pd.read_csv(edge_filename)
	df.columns = df.columns.str.lower()
	header = list(df)
	
	if [str1 for str1 in header[:2]] != ["node1", "node2"]: 
		raise ValueError("The first two columns in network file should be arranged as named as 'node1', 'node2'")
	if "weight" not in header[2]:
		raise ValueError("The third column in network file should contain string = weight")
	if is_network_dynamic and (len(header)<4 or header[3] != "timestep"):
		raise ValueError("Time-stamps are either missing or the column is not labelled as 'timestep'!")

	##rename the weight column as weight
	df.rename(columns={header[2]: 'weight'}, inplace=True)

	if not is_network_dynamic:
		if "timestep" in header:
			raise ValueError("Network dynamic set as False but the infection data has timesteps!")
		n_edges = len(df.index)
		timelist = [[num]*n_edges for num in xrange(time_max+1)]
		timelist = [val for sublist in timelist for val in sublist]
		df=pd.concat([df]*(time_max+1), ignore_index=True)
		df['timestep']=timelist

	
	if normalize_edge_weight:
		## If the user asks for edge weight normalization, then calculate total edge weights
		## at each time step
		df_sum = df.groupby('timestep')['weight'].sum()
		total_weight= df_sum.to_dict()
		for time1 in total_weight:
			#replace  raw weights with normalized weights
			df.ix[df.timestep==time1, "weight"] = df["weight"]/total_weight[time1] 
			

	
	G = {}
	for time1 in df["timestep"].unique():  G[time1] = nx.Graph()
	for time1 in G:
		df_sub= df.loc[df['timestep'] == time1]
		df_sub['node1'] = df_sub['node1'].astype(str)
		df_sub['node2'] = df_sub['node2'].astype(str)
		edge_list = list(zip(df_sub.node1, df_sub.node2))
		G[time1].add_edges_from(edge_list)
		edge_attr = dict(zip(zip(df_sub.node1, df_sub.node2), df_sub.weight))
		nx.set_edge_attributes(G[time1], 'weight', edge_attr)


	return G

##########################################################################
def extract_nodelist(edge_filename):

	df = pd.read_csv(edge_filename)
	node1_list = list(df["node1"].unique())
	node2_list = list(df["node2"].unique())
	nodelist = list(set(node1_list + node2_list))
	##convert to string
	nodelist = [str(node) for node in nodelist]
	return nodelist
#######################################################################
def check_edge_weights(G):
	""" Code convergence is better if the edge weights are normalized
	at each time step"""
	
	total_edge_wt = []
	for time1 in G.keys():

		total_wt = sum(nx.get_edge_attributes(G[time1],'weight').values())
		total_wt = int(round(total_wt,1))
		total_edge_wt.append(total_wt)
	
	if list(set(total_edge_wt))!= [1]:
		print ("Warning: Code convergence is better if the edge weights are normalized at each time step" )

#################################################################
def permute_network(G1, permutation):
	
	G2 = {}
	jaccard_list = []
	for time in G1.keys():
		G2[time] = nx.Graph()
		G2[time].add_nodes_from(G1[time].nodes())
		num_swaps = int(permutation*len(G1[time].edges()))
		num_orig = len(G1[time].edges()) - num_swaps
		orig_edges = G1[time].edges()
		shuffle(orig_edges)
		track_wt = []
		for num in xrange(num_orig):
			node1, node2 = orig_edges.pop()	
			G2[time].add_edge(node1, node2)
			wt= G1[time][node1][node2]["weight"]
			G2[time][node1][node2]["weight"] = wt 
			track_wt.append(wt)
		wtlist = [G1[time][node1][node2]["weight"] for node1, node2 in G1[time].edges()]
		#remove edge weights that have been assigned
		for wt in track_wt: wtlist.remove(wt)
		

		connections = 0 # skip over node pairs that already have an edge
		while connections <num_swaps:
			node1, node2 = np.random.choice(G2[time].nodes(), 2, replace=False)
			if not G2[time].has_edge(node1, node2): 
					G2[time].add_edge(node1, node2)
					if len(wtlist)==0:print ("check!!!"), permutation, num_swaps, num_orig, len(G2[time].edges())
					G2[time][node1][node2]["weight"] = wtlist.pop()
					connections+=1
		
		jaccard_list.append(calculate_jaccard(G1[time], G2[time]))
	#print ("random graph check"), permutation, np.mean(jaccard_list)
			
	return G2 


#################################################################
def randomize_network(G1):
	
	G2 = {}
	for time in G1.keys():
		G2[time] = nx.Graph()
		G2[time].add_nodes_from(G1[time].nodes())
		wtlist = [G1[time][node1][node2]["weight"] for node1, node2 in G1[time].edges()]
		shuffle(wtlist)
		for num in xrange(len(G1[time].edges())): #for each edge in G1[time]
			#select two random nodes from G2[time]
			condition_met = False # skip over node pairs that already have an edge
			while not condition_met:
				node1, node2 = np.random.choice(G2[time].nodes(), 2, replace=False)
				if not (G2[time].has_edge(node1, node2) or G1[time].has_edge(node1, node2)): 
					condition_met = True
					G2[time].add_edge(node1, node2)
					G2[time][node1][node2]["weight"] = wtlist.pop()
		

	jaccard = calculate_mean_temporal_jaccard(G1, G2)
	if jaccard > 0.4: print ("Warning!! Randomized network resembles empircal network. May lead to inconsistent evidence")
	#print ("random graph check"), jaccard
			
	return G2 
#######################################################################		
def stitch_health_data(health_data):
	""" Fill in time steps with same infection status"""
	for node in health_data.keys():
		timelist = health_data[node].keys()
		timelist = sorted(timelist)
		for num in range(1, len(timelist)):
			time2 = timelist[num]
			time1 = timelist[num-1]
			if health_data[node][time1]==health_data[node][time2]:
				for step in range(time1+1, time2): health_data[node][step] = health_data[node][time2]
	
	
	return health_data	
#######################################################################
def extract_health_data(health_filename, infection_type, nodelist, diagnosis_lag=False):

	r"""node_health is a dictionary of dictionary. Primary key = node id.
	Secondary key = 0/1. 0 (1) key stores chunk of days when the node is **known** to be healthy (infected).
	 Dates stored as tuple of (start date, end date)"""

	health_data = {}
	for node in nodelist: health_data[str(node)]={}

	with open (health_filename, 'r') as csvfile:
		fileread = csv.reader(csvfile, delimiter = ',')
		next(fileread, None) #skip header
		for row in fileread: 
			node = row[0]
			timestep = int(row[1])
			diagnosis = int(row[2])
			if node in nodelist:health_data[node][timestep] = diagnosis
			

	if diagnosis_lag: health_data = stitch_health_data(health_data)

	node_health = {}
	for node in health_data.keys():
		sick_list_node=[]
		healthy_list_node=[]
		node_health[node] = {}
		## sort infection reported into infected (sick_list) and healthy (healthy_list) lists
		for time1 in health_data[node].keys():
			if health_data[node][time1]==1: sick_list_node.append(time1)
			if health_data[node][time1]==0: healthy_list_node.append(time1)
		
		if len(healthy_list_node)>0:
			healthy_list_node = sorted(healthy_list_node)
			node_health[node][0] = select_healthy_time(healthy_list_node, node, health_data, infection_type)
			
				
		
		if len(sick_list_node)>0:
			sick_list_node = sorted(sick_list_node)
			node_health[node][1]= select_sick_times(sick_list_node, node, health_data) 

		if node_health[node].has_key(1):		
			for time1, time2 in node_health[node][1]:
				##impute the missing report of sick in health data dictionary
				for day in range(time1, time2+1): health_data[node][day]=1
			
	
	return health_data, node_health
##############################################################################
def select_healthy_time(healthy_list_node, node, health_data, infection_type):

	r""" Select chunks of time-periods (from healthy_list_node) for which the node
	     is reported uninfected"""

	healthy_times = []
	
	#if SIR type of infection
	if infection_type[-1].lower()=='r':
		#min date = if there was no report before the day
		min_date = [time for time in healthy_list_node if len([val for key, val in health_data[node].items() if key<time])==0]
		
	else: 
		#min date = if there is (any report before focal date AND the last report is sick) OR there is no report before the day
		min_date = [time for time in healthy_list_node if (len([val for key, val in health_data[node].items() if key<time])>0 and health_data[node][max([key for key in health_data[node] if key < time])]==1) or len([val for key, val in health_data[node].items() if key<time])==0]
		
	#max_date = if there is (any report after focal date AND the report is sick) OR there is no report after the day
	max_date = [time for time in healthy_list_node if (len([val for key, val in health_data[node].items() if key> time])>0 and health_data[node][min([key for key in health_data[node] if key > time])]==1) or len([val for key, val in health_data[node].items() if key>time])==0]
	
	min_date = sorted(min_date)
	max_date = sorted(max_date)
	
	for day1, day2 in zip(min_date, max_date): healthy_times.append((day1, day2))
	return healthy_times
	
############################################################################
def select_sick_times(sick_list_node, node, health_data):
	r""" Select chunks of time-periods (from sick_list_node) for which the node
	     is reported sick"""


	sick_times = []
	#min date = if there is (any report before the focal date AND the last report therin is healthy) OR there is no report before the day
	min_date = [time for time in sick_list_node if (len([val for key, val in health_data[node].items() if key<time])>0 and health_data[node][max([key for key in health_data[node] if key < time])]==0) or len([val for key, val in health_data[node].items() if key<time])==0]
	#max date = if there is (any report after the focal date AND the first report theirin is sick) OR there is no report after the focal date
	max_date = [time for time in sick_list_node if (len([val for key, val in health_data[node].items() if key> time])>0 and health_data[node][min([key for key in health_data[node] if key > time])]==0) or len([val for key, val in health_data[node].items() if key>time])==0]
	
	min_date = sorted(min_date)
	max_date = sorted(max_date)
	
	for day1, day2 in zip(min_date, max_date): sick_times.append((day1, day2))
	return sick_times

#########################################################################
def return_contact_days_sick_nodes(node_health, seed_date, G_raw):
	r"""If infection diagnosis is lagged, then true infection day is
	inferred using infectious contact history of the focal node """

	contact_daylist={key:{} for key in G_raw}
	## select all nodes that were reported infected and sort
	for node in sorted([node1 for node1 in node_health.keys() if node_health[node1].has_key(1)]):
		## removing seed nodes
		sick_days = [(time1, time2) for (time1, time2) in sorted(node_health[node][1]) if time1!= seed_date]
		##for all time periods when the node was reported sick
		for time1, time2 in sick_days:
			#default day start
			day_start =0
			if node_health[node].has_key(0):
				##choose all uninfected time-periods before the focal sick period
				healthy_dates = [(healthy_day1, healthy_day2) for healthy_day1, healthy_day2 in node_health[node][0] if healthy_day2 < time1]
				if len(healthy_dates)>0:
					##choose the latest uninfected period
					lower_limit, upper_limit = max(healthy_dates, key=lambda x:x[1])
					##choose the last ever reported time-point of being uninfected
					day_start = upper_limit

			##contact_daylist is dictionary. Primary key = network type. Could be network hypothesis or null network
			## secondary key (node1, time1, time2) indicated focal node and its infected time period
			##values are the time-points when the node could have potentially contracted infection
			for network in G_raw:
				#choose only those days where nodes has contact the previous day
				contact_daylist[network][(node, time1, time2)] =[day for day in range(day_start+1, time1+1) if G_raw[network][day-1].degree(node)>0]

	return contact_daylist

#########################################################################
def return_potention_recovery_date(node_health, time_max,  G_raw):
	r""" For SIR model. Returns the potential time-points of recovery for each 
	infected focal node"""
	
	recovery_daylist = {}
	## select all nodes that were reported infected and sort
	for node in sorted([node1 for node1 in node_health.keys() if node_health[node1].has_key(1)]):
		## sort sick days for the focal node
		sick_days = sorted(node_health[node][1])
		for time1, time2 in sick_days:
			if node_health[node].has_key(0):
				##choose all uninfected time-periods after the focal sick period
				healthy_dates = [(healthy_day1, healthy_day2) for healthy_day1, healthy_day2 in node_health[node][0] if healthy_day2 > time1]
				if len(healthy_dates)>0:
					##choose the first report of uninfection
					lower_limit, upper_limit = min(healthy_dates, key=lambda x:x[1])
					recovery_date = lower_limit
			else: recovery_date = time_max
				
			##recovery date can be any time-point between the last report of node "infection" state to the the first 
			## report of uninfection afterwards (or time_max of the study)
			recovery_daylist[(node, time1, time2)] = recovery_date

	return recovery_daylist
####################################################################################
def find_seed_date(node_health):
	r"""Find the first time-period where an infection was reported"""

	sick_dates = [val for node in node_health.keys() for key,val in node_health[node].items() if key==1]
	#flatten list
	sick_dates = [item for sublist in sick_dates for item in sublist]
	
	#pick the first date
	sick_dates=[num[0] for num in sick_dates]
	
	#sort tuple according to increasing infection dates
	sick_dates = sorted(sick_dates)
	# pick out the first reported infection date
	seed_date = sick_dates[0]
	
	return seed_date

########################################################################
def calculate_jaccard(g1, g2):

	edges1 = [tuple(sorted(num)) for num in g1.edges()]
	edges2 = [tuple(sorted(num)) for num in g2.edges()]
	w11 = len(list(set(edges1) & set(edges2)))
	w10 = len(list(set(edges1) - set(edges2)))
	w01 =  len(list(set(edges2) - set(edges1)))
	ratio = w11/ (1.*(w11+w10+w01))
	return ratio
########################################################################
def calculate_mean_temporal_jaccard(g1, g2):

	jlist = []
	for time1 in g1:
		edges1 = [tuple(sorted(num)) for num in g1[time1].edges()]
		edges2 = [tuple(sorted(num)) for num in g2[time1].edges()]
		w11 = len(list(set(edges1) & set(edges2)))
		w10 = len(list(set(edges1) - set(edges2)))
		w01 =  len(list(set(edges2) - set(edges1)))
		ratio = w11/ (1.*(w11+w10+w01))
		jlist.append(ratio)
	return np.mean(jlist)
########################################################################
def check_init_pars(logl, logp, p0, data):
    
    ok = True
    p0 = p0.reshape((1, -1, p0.shape[-1]))
    if logp(p0) == -np.inf: 
		print ("warning log prior false")
		ok = False
       
    p1 = p0[0::][0]
    loglist=[]
    for num in xrange(p1.shape[0]): 
		loglist.append(logl(p1[num], data))
   
    if -np.inf in loglist:ok = False
       

    return ok
########################################################
def compute_diagnosis_lag_truth(graph, contact_datelist, filename):

	diag_date = {}
	infection_date={}
	lag_truths = []
	with open (filename, 'r') as csvfile:
		fileread = csv.reader(csvfile, delimiter = ',')
		next(fileread, None) #skip header
		for row in fileread:
			node = row[0]
			timestep = int(row[1])
			diagnosis = int(row[2])
			if diagnosis==1: infection_date[node] = timestep
		

	for node, time1, time2 in sorted(contact_datelist):
		daylist = [day for day in contact_datelist[(node, time1, time2)] if graph[day-1].degree(node)>0]
		pos = [pos for pos, date in enumerate(daylist) if date==infection_date[node]][0]
		lag_truths.append(ss.randint.cdf(pos,  0,  len(daylist)))

	return lag_truths

######################################################333
def plot_beta_results(sampler, beta_truth, filename):

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 6))
        ax1.plot(sampler.chain[0, :, :, 0].T, color="k", lw=0.1)
        ax1.set_ylabel("Walker positions for $beta$")
        ax1.set_xlabel("Simulation step")
        
    
        samples = sampler.chain[0, :, :, 0].reshape((-1, 1))

        ax2.hist(samples, bins=50, histtype="step", normed=True, label="posterior", color="k", linewidth=2)
        ax2.axvline(beta_truth, label="data point", color="r")
        ax2.legend(frameon=False, loc="best")
        ax2.set_xlabel("$beta$ posterior")
        
 	plt.tight_layout()
        plt.savefig(filename)

########################################################################
def delete_edge_connections(g, percent_remove):

	edge_list = []
	for time in g:	
		for edge1 in g[time].edges(): edge_list.append((time, edge1))

	total_edges = len(edge_list)
	del_edges = int(percent_remove*total_edges)
	del_edges_per_time = int(del_edges/float(len(g)))
	print total_edges, del_edges, del_edges_per_time, len(g)
	G={}

	for time in g:
		G[time]=nx.Graph()
		edgelist = g[time].edges()
		shuffle(edgelist)
		mod_edgelist = edgelist[del_edges_per_time:]
		G[time].add_edges_from(mod_edgelist)
		for (node1, node2) in mod_edgelist:
			G[time][node1][node2]["weight"] = g[time][node1][node2]["weight"] 
	
	print ("graph check for del edges"), calculate_mean_temporal_jaccard(g, G)
	return G
		
	
	
