import INoDS_model as inods
##################################################

edge_filename = "Edge_connections.csv"
health_filename = "Health_data.csv"
null_networks = 10 ##the number of null networks required
priors = [(0,1), (0,1)] #order = beta, alpha
verbose=True
iteration = 2
burnin = 1
normalize_edge_weight= True
nodelist = [str(num) for num in xrange(100)]

nods.run_nbda_analysis(edge_filename, health_filename, nodelist, null_networks, priors, iteration, burnin, diagnosis_lag=False, null_comparison=False)
