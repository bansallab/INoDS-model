INoDS (Inferring Network of infectious Disease Spread) 
================================================

INoDs is a tool to assess whether an empirical contact network is likely to generate an observed pattern of infectious disease spread in a host population. INoDS also provides epidemiological insights into the spreading pathogen by estimating the per-contact rate of pathogen transmission.

The details of the tool is described in

> Sah, P and Bansal, S. [Identifying the dynamic contact network of infectious disease spread](https://www.biorxiv.org/content/early/2017/07/28/169573). 
> bioRxiv (2017): 169573.


Requirements for directly running the source code
================================================
* [Python 2.7](http://python.org/)
* [Emcee 2.1.0](http://dfm.io/emcee/current/)
* [Networkx 1.11](https://networkx.github.io/)
* [Corner 2.0.1](https://pypi.python.org/pypi/corner/)


Usage
================================

A quick demo of the code is included in the examples folder. The code can be run using the command

$ python run_inods.py


Model parameters
================================


Output
================================

The tools outputs the following files

*Convergence diagnostic* 
Autocorrelation plot of three randomly selected walkers.
*Parameter estimation* 
Three files are generated for this step. (i) Output of emcee.PTsampler saved as an pickled object, (ii) Posterior plot of $\beta$ and error parameter, (iii) A plot of walker positions for $\beta$ parameter and $\beta$ posterior.
*Null comparison*
At this step two files are generated - a .csv file with predictive power of the empirical contact network (first row) and null network, and a figure summarizing the results.


License
================================

Copyright 2017 Pratha Sah and Shweta Bansal.

INoDS is free software made available under the MIT License. For details see
the LICENSE file
