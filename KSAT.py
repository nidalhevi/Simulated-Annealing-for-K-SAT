import numpy as np
from copy import deepcopy

class KSAT:
    def __init__(self, N, M, K, seed = None):
        if not (isinstance(K, int) and K >= 2):
            raise Exception("k must be an int greater or equal than 2")
        self.K = K
        self.M = M
        self.N = N

        ## Optionally set up the random number generator state
        if seed is not None:
            np.random.seed(seed)
    
        # s is the sign matrix
        s = np.random.choice([-1,1], size=(M,K))
        
        # index is the matrix reporting the index of the K variables of the m-th clause 
        index = np.zeros((M,K), dtype = int)        
        for m in range(M):
            index[m] = np.random.choice(N, size=(K), replace=False)
            
        # Dictionary for keeping track of literals in clauses
        clauses = []   
        for n in range(N):
            clauses.append([i for i, row in enumerate(index) if n in row])
        
        self.s, self.index, self.clauses = s, index, clauses        
        
        ## Inizializza la configurazione
        x = np.ones(N, dtype=int)
        self.x = x
        self.init_config()

    ## Initialize (or reset) the current configuration
    def init_config(self):
        N = self.N 
        self.x[:] = np.random.choice([-1,1], size=(N))

    
    ## Definition of the cost function
    # Here you need to complete the function computing the cost using eq.(4) of pdf file
    def cost(self):        
        matrix = self.s * self.x[self.index] # setting the matrix of literals by multiplying the sign matrix by corresponding x values of the index matrix
        cost_row = np.prod((1 - matrix) / 2, axis=1) # cost of each row
        return np.sum(cost_row) # total cost
                 
    
    ## Propose a valid random move. 
    def propose_move(self):
        N = self.N
        move = np.random.choice(N)
        return move
    
    ## Modify the current configuration, accepting the proposed move
    def accept_move(self, move):
        self.x[move] *= -1


    
    ## Compute the extra cost of the move (new-old, negative means convenient)
    # Here you need complete the compute_delta_cost function as explained in the pdf file
    def compute_delta_cost(self, move):
        # Access the affected clauses for the variable being flipped(multiplied by -1)
        affected_clauses = self.clauses[move]

        x = self.x
        s = self.s
        index = self.index

        # Get the indices of the variables in the affected clauses
        clause_indices = index[affected_clauses] 
        
        # Get the signs and values of the variables in the affected clauses
        clause_signs = s[affected_clauses]       
        clause_values = x[clause_indices]         
        
        # Compute the old contributions for all affected clauses
        old_matrix = clause_signs * clause_values 
        old_cost_row = np.prod((1 - old_matrix) / 2, axis=1)  
        
        # Flip the variable in the affected clauses
        flipped_values = clause_values.copy()
        flipped_values[clause_indices == move] *= -1  # Flip the move variable in the clauses
        
        # Compute the new contributions for all affected clauses
        new_matrix = clause_signs * flipped_values 
        new_cost_row = np.prod((1 - new_matrix) / 2, axis=1)  
        
        # Compute the delta cost
        delta_cost = np.sum(new_cost_row) - np.sum(old_cost_row)

        return delta_cost

    ## Make an entirely independent duplicate of the current object.
    def copy(self):
        return deepcopy(self)
    
    ## The display function should not be implemented
    def display(self):    
        print("K-SAT Problem Configuration")
        print("=" * 30)
        print(f"Number of Variables (N): {self.N}")
        print(f"Number of Clauses (M): {self.M}")
        print(f"Number of Variables per Clause (K): {self.K}")
        print("\nSign Matrix (s):")
        print(self.s)
        print("\nIndex Matrix (index):")
        print(self.index)
        print("\nCurrent Configuration (x):")
        print(self.x)
        print("\nClauses:")
        for m in range(self.M):
            variables = self.index[m]
            signs = self.s[m]
            clause = " ∨ ".join([f"¬x{v+1}" if s == -1 else f"x{v+1}" for v, s in zip(variables, signs)])
            print(f"Clause {m+1}: {clause}")
        print("=" * 30)

        
        
# example = KSAT(M=3, K=3, N=4, seed=42)

# print(example.cost())
# print(example.compute_delta_cost(move = 2))
    
