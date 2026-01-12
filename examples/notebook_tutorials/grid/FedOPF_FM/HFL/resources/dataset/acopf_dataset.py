from resources.dataset.acopf_prob import ACOPFProblem
# from resources.dataset.acopf_prob_advanced import ACOPFProblem
import os

def get_acopf_data(
        client_id: int
): 
    """
    Return the ACOPF dataset for a given client.
    :param client_id: the client id <== NOTE that this should be the # of bus
    """
    # Load acopf dataset
    dir = os.getcwd() + "/datasets/acopf/case"+str(client_id)+"/"
    data_dir = dir + "FeasiblePairs_case"+str(client_id)+"_perturb_5000_samples.mat"
    grid_dir = dir + "pglib_opf_case"+str(client_id)+".mat"

    acopf_dataset = ACOPFProblem(data_filename=data_dir, grid_filename=grid_dir)

    return acopf_dataset.train_dataset, acopf_dataset.valid_dataset, acopf_dataset.test_dataset, acopf_dataset
    
