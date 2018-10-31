"""
Solution configuration for HW02 Problem 3.
"""
from pymdp.mdp import MarkovDecisionProcess
from pymdp.solvers import value_iteration, policy_iteration

# state x = (i,j) if the trailer is at site i and the repairman at facility j
states = [(i, j) for i in range(1, 5) for j in range(1, 5)]

# action a = k to move the trailer to site k
actions = {s: [i for i in range(1, 5)] for s in states}

beta = 0.95

# matrix[i][j] = probability that repairman goes to facility j from facility i
repairman_matrix = [
        [0.1, 0.3, 0.3, 0.3],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.8, 0.2],
        [0.4, 0.0, 0,0, 0.6]
        ]

def transition(y, x, a):
    if y[0] != a:
        return 0.0
    else:
        return repairman_matrix[x[1]-1][y[1]-1]

cmat = [
        [0, 200, 200, 200],
        [0, 50, 100, 100],
        [0, 100, 50, 100],
        [0, 100, 100, 50]
        ]

dmat = [
        [0, 300, 300, 300],
        [300, 0, 300, 300],
        [300, 300, 0, 300],
        [300, 300, 300, 0]
        ]

def reward(x, a):
    return -cmat[x[0]-1][x[1]-1] - dmat[x[0]-1][a-1]

mdp = MarkovDecisionProcess(states, actions, reward, transition, beta)
policy = policy_iteration(mdp)
# vi_result = value_iteration(mdp, 1e-3)
# 
# for state in states:
#     print(' & '.join(map(str, [state, vi_result[0][state],
#         vi_result[1][state]])))
# 
# print("through value iteration")
# print(policy_iteration(mdp))
# print("through policy iteration")
