import KSAT
import SimAnn
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

def worker(args): # define worker function 
    N, M, seed, mcmc_steps, anneal_steps, beta0, beta1 = args
    random_seed = np.random.randint(0,10000)
    # Create a random KSAT problem instance
    problem = KSAT.KSAT(N=N, M=M, K=3, seed = random_seed)
    # Run Simulated Annealing
    result = SimAnn.simann(problem, anneal_steps=anneal_steps, mcmc_steps=mcmc_steps, beta0=beta0, beta1=beta1, seed=seed)
    return 1 if result.cost() == 0 else 0 # 1 for solved 0 for not solved

def P(N, M, instances=30): # define P
    arg = (N, M, 31, 10*N, 500, 0.1, -np.log(1-(0.95)**(1/(10*N))))  #setting parameters
    args = [arg for _ in range(instances)] #setting parameters
    with Pool (processes= os.cpu_count()) as pool: # Multiprocessing
        results = pool.map(worker, args)
    
    return sum(results)/instances # return the ratio of solved instances
        

def compute_P_N_M(N, M_clauses): #compute P for multiple M values
    probabilities = []
    for M in M_clauses: #
        p = P(N, M) # compute probabilities for each value of M
        probabilities.append(p)
        
    return probabilities


def homogenize(array, sentinel): # homogenise the arrays
    max_length = max([len(row) for row in array])
    for row in array:
        current_lenght = len(row)
        difference = max_length - current_lenght
        if difference == 0:
            continue
        row.extend([sentinel for _ in range(difference)])
        
    return array



if __name__ == "__main__": # Necessary for multiprocessing
    P_N_M = compute_P_N_M(N=200, M_clauses=[400,500,600,700,800,900,1000]) #set the values and compute probabilities
    m = 100 # initialise m
    n_list = [200, 300, 400, 500, 600] # different values of N
    p_matrix = [] # initialise probabilities matrix
    m_matrix = [] 
    
    for n in n_list: # iterate through each N
        prob = 1.0
        probabilities = [] # initialise probabilities list
        M_numbers = []
        while prob > 0.5: # increase M while we are over the algorithmic treshold for 1/2
            m += 100 
            prob = P(n, m)
            probabilities.append(prob) # append to the list
            M_numbers.append(m)
            
            print(f"P(N={n}, M={m}) = {prob:.2f}")
            
            
        p_matrix.append(probabilities) # append the list into the matrix
        m_matrix.append(M_numbers)
        
    p_matrix = homogenize(p_matrix, sentinel=0) # homogenise with sentinel = 0
    m_matrix = homogenize(m_matrix, np.inf) # homogenise with sentinel = inf
    
    m_algs = [np.interp(0.5, P_values[::-1], M_values[::-1]) for P_values, M_values in zip(p_matrix, m_matrix)] # algorithmic threshold for each N
    plt.plot([200, 300, 400, 500, 600], m_algs, marker = 'o') # plot algoritmic threshold for each N
    plt.xlabel("N Values")
    plt.ylabel("M algs")
    plt.title("Algorithm Thresholds")
    plt.savefig("N's comparison")
    plt.clf()
    
    print(f"m_algs = {m_algs}")
    
    m_algs_array = np.array(m_algs)
    n_list_array = np.array(n_list)
    
    coefficients = m_algs_array/n_list_array # find coefficients, c, for each m^alg, N pair
    mean_coefficients = np.mean(coefficients) # mean coefficients, the final value of c
    print(f"mean_coefficients = {mean_coefficients}")
        
    
    # Plot results
    
    # Plot seperately for each M
    idx = 0
    for M_values, P_values in zip(m_matrix, p_matrix):
        idx += 1
        plt.plot(M_values, P_values, marker='o')
        plt.axhline(0.5, color='red', linestyle='--', label="P(N, M) = 0.5")
        plt.xlabel("Number of Clauses (M)")
        plt.ylabel("Empirical Probability P(N, M)")
        plt.title(f"Empirical Probability P(N, M) vs Number of Clauses")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plot{idx}")
        plt.clf()
    
    # Plot together for each M
    for M_values, P_values in zip(m_matrix, p_matrix):
        idx += 3
        plt.plot(M_values, P_values, marker='o')
    plt.axhline(0.5, color='red', linestyle='--', label="P(N, M) = 0.5")
    plt.xlabel("Number of Clauses (M)")
    plt.ylabel("Empirical Probability P(N, M)")
    plt.title(f"Empirical Probability P(N, M) vs Number of Clauses")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot_together{idx}")
    plt.clf()

    # Collapsed plots
    for i, (M_values, P_values) in enumerate(zip(m_matrix, p_matrix)):
        idx += 1 
        plt.plot(M_values/n_list_array[i], P_values, marker='o')
    plt.axhline(0.5, color='red', linestyle='--', label="P(N, M) = 0.5")
    plt.xlabel("Normalized M values")
    plt.ylabel("Empirical Probability P(N, M)")
    plt.title(f"Collapsed Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot{idx}")
    plt.clf()
    



    

