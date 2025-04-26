import SimAnn
import KSAT


## Generate a problem to solve.
# This generate a K-SAT instance with N=100 variables and M=350 Clauses
ksat = KSAT.KSAT(N=200, M=200, K=3, seed=31)

## Optimize it.
best = SimAnn.simann(ksat,
                     mcmc_steps = 2000, anneal_steps = 150,
                     beta0 = 0.1, beta1 = 10.57,
                     seed = 31,
                     debug_delta_cost = False) # set to True to enable the check

KSAT.KSAT.display()




