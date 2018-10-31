from mdp_solver import MarkovDecisionProcess, policy_iteration
from collections import defaultdict
from csv import DictReader

states = [(i, j) for i in range(1, 4) for j in range(1, 5)]
actions = defaultdict(list)
rewards = defaultdict(float)

action_transitions = {
        'up': (1,0), 
        'down': (-1,0), 
        'left': (0,-1), 
        'right': (0,1), 
        'stop': (0,0)
        }

with open('grid.csv') as grid_csv:
    reader = DictReader(grid_csv)
    for row in reader:
        state = int(row['row']), int(row['col'])
        actions[state] = row['actions'].split()
        rewards[state] = float(row['reward'])

def reward(state, action):
    if action == 'stop':
        return 0.
    else:
        return rewards[state]

def transition(next_state, state, action):
    viable_transitions = get_transitions(state, action)
    if next_state not in viable_transitions:
        return 0.0
    else:
        return viable_transitions[next_state]

def _ib(state):
    return (max(1, min(3, state[0])), max(1, min(4, state[1])))

def get_transitions(state, action):
    viable_transitions = defaultdict(float)
    def _shift(action):
        shifts = []
        if action == 'up':
            shifts = [action_transitions[a] for a in ['left','up','right']]
        elif action == 'down':
            shifts = [action_transitions[a] for a in ['right','down','left']]
        elif action == 'left':
            shifts = [action_transitions[a] for a in ['down', 'left', 'up',]]
        elif action == 'right':
            shifts = [action_transitions[a] for a in ['up','right','down']]
        else:
            return [action_transitions['stop'], 1.]
        return list(zip(shifts, [0.1, 0.8, 0.1]))

    def _check_state(state, action):
        if state == (2,2):


    shifts = _shift(action)
    print(shifts)
    for (y, x), pr in shifts:
        next_state = _check_state((state[0]+y, state[1]+x), action)
        viable_transitions[next_state] += pr

    return viable_transitions





