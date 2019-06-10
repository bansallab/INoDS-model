INoDS (Inferring Network of infectious Disease Spread) 
================================================

INoDs is a tool to assess whether an empirical contact network is likely to generate an observed pattern of infectious disease spread in a host population. INoDS also provides epidemiological insights into the spreading pathogen by estimating the per-contact rate of pathogen transmission.

The details of the tool is described in

> Sah, P and Bansal, S. [Identifying the dynamic contact network of infectious disease spread](https://www.biorxiv.org/content/early/2017/07/28/169573). 
> bioRxiv (2017): 169573.


Requirements for directly running the source code
================================================
* [Python 2.7](http://python.org/)
* [Emcee 3.0rc2](http://dfm.io/emcee/current/)
* [Networkx 2.2](https://networkx.github.io/)
* [Corner 2.0.1](https://pypi.python.org/pypi/corner/)
* [h5py 2.9.0]
* [tqdm-4.32.1]


Environment
================================
A working Python environment for this project can be generated using the [conda](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) "environment.yml":  
```conda env create -f environment.yml```  
To activate the Python environment titled "inods":  
```source activate inods```


Usage
================================

A quick demo of the code is included in the examples folder. The code can be run using the command

$ python run_inods.py


Input files
================================
edge_filename: Filename of the network edgelist. See *Edge_connections_poisson.csv* in the examples folder for the accepted file format. 
* Note 1: Both dynamic and static networks are accepated. For static networks, remove the "timestep" column in the edgelist file
* Note 2: The networks can be unweighted or weighted. For unweighted networks (as shown in *Edge_connections_poisson.csv*), set all values in the *weight* column as one.


health_filename: Filename of the infection data. See *Health_data_nolag.csv* in the examples folder for the accepted file format. Infection states are coded as: 0 - diagnosed to be uninfected and 1 - diagnosed to be infected. Node ids in the infection data should correspond to the network edgelist, but infection data on all nodes (or all timesteps) is not required.

Parameters
===================================
output_filename: Desired filename for the output files.


infection_type: Can be either "SI", "SIR" or "SIS".


truth (optional, default = None): True values of parameters, if known, are entered as a list. If unknown set the truth as a list of zeroes.


null_networks (optional, default = 500): Total number of null network. Default number of null networks is 500.


burnin (optional, default = 1000): Total burn-in perior for *emcee* sampler.


max_iteration (optional, default = 50,000): Maximum number of iterations after burn-in for *emcee* sampler. The sampler terminates at maximum iteration even if convergence is not achieved.  


diagnosis_lag: (optional, default = False). Set to True when actual infection timing is unknown and the infection file reports *diagnosis times* instead of *infection times*.  


verbose: (optional, default = True) Set to False to supress printing of detailed status messages. 


null_comparison: (optional, default = True) Set to False to skip comparing the predictive power of empirical contact network to null networks.  


edge_weights_to_binary: (optional, default = False) Set to True to remove edge-weights from the empircal network, and assign all edges with edge-weight of one.


normalize_edge_weight: (optional, default = False) Set to True to normalize edge-weights of the empircal network by dividing all edge-weights with the maximum edge-weight.


is_network_dynamic: (optional, default = True) Set to False if the empirical network is static.


parameter_estimate: (optional, default = True) Set to False to skip the the estimation of unknown parameters.


compare_asocial_social_force: (optional, default = True) Set to False to skip comparisons of "social" vs. "asocial" force of infection given the empircal contact network.


Output
================================

The tools outputs the following files

* Convergence diagnostics: Average autocorrelation time (in blue) is plotted as a function of chain length. Convergence is assumed to be achieved when the chain length (N) is longer than 100 times the estimated autocorrelation time. The N>100xautocorrelation time threshold is plotted as a dashed line.
* Parameter estimation: Three files are generated for this step. (i) Summary of parameter estimates and BIC of the contact network, (ii) Posterior plot of &beta; and epsilon parameter, (iii) A plot of walker positions for &beta; parameter and &beta; posterior.
* Null comparison: At this step two files are generated - a .csv file with the likelihood of disease data under the assumption that the disease  spreads through the contact network (first row) and likelihoods for null networks (second to last row). A figure summarizing the results is also generated.


License
================================

Copyright 2017 Pratha Sah and Shweta Bansal.

INoDS is free software made available under the MIT License. For details see the LICENSE file.
