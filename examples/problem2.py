from mdp_solver import MarkovDecisionProcess, value_iteration

states = [1, 2, 3, 4, 5]
actions = {
        1: ['M', 'U'],
        2: ['M', 'U'],
        3: ['M', 'U'],
        4: ['M', 'U'],
        5: ['M', 'U']
        }

def transition(y, x, a): 
    if a not in actions[x]: 
        return 0.0
    elif a == 'M':
        if y == 1:
            return 1.
        else:
            return 0.
    else:
        probs = {1: 0.97, 2: 0.9, 3: 0.8, 4: 0.3, 5: 0.}
        if y == x + 1:
            return probs[x]
        elif y == 1:
            return 1. - probs[x]
        else:
            return 0.

def reward(y, a):
    if y == 1 and a == 'M':
        return -100.
    elif y == 1 and a == 'U':
        return -3000.
    elif a == 'U':
        return 500.
    else:
        return 0.


machine_operator = MarkovDecisionProcess(states, actions, reward, transition,
        discount=0.9)

print(value_iteration(machine_operator, 1e-3))
