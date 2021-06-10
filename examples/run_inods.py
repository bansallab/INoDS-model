if __name__ == '__main__': 
    import sys
    import os
    sys.path = [os.path.abspath(os.path.join(__file__, '..', '..')), ] + sys.path
    sys.path.append('/Users/prathasah/Dropbox (Bansal Lab)/Git-files/INoDS-model/')
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
    edge_filename = "data/Edge_connections_poisson_n100_d4.csv"
    # Prvide the health data
    health_filename = "data/Health_data_nolag_beta_0.1.csv"
    # provide filename for output files
    output_filename = "example_dataset"
    ###########################################
    ### Model parameters
    ###########################################
    ##do you know the true values of beta and epsilon? 
    truth = [0.1, 0]
    
    infection_type = "SI"
    #####################################
    #### run INoDS 
    ######################################

    inods.run_inods_sampler(edge_filename, health_filename, output_filename, infection_type, truth = truth, verbose=True, is_network_dynamic=True,  parameter_estimate=True)

