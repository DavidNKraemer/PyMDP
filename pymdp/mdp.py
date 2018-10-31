from collections import namedtuple

MarkovDecisionProcess = namedtuple(
    "MarkovDecisionProcess",
    ["states", "actions", "reward", "transition", "discount"]
    )

class MDPFunction(dict):
    def __init__(self, dictionary):
        super.__init__(dictionary)

    def __call__(self, key):
        return self.__getitem__(key)


def default_value(mdp):
    """
    Returns the "zero" value function, which assigns to each state the value 0.0

    Value functions will be implemented as dictionaries. So if x is a state of
    the MDP, then value[x] yields the value function at x.
    """
    return MDPFunction({x: 0. for x in mdp.states})


def default_policy(mdp):
    """
    Returns an arbitrary policy, which assigns to each state some available
    action.

    Policy functions will be implemented as dictionaries. So if x is a state of
    the MDP, then policy[x] yields the action associated with the stationary
    policy for x.

    Note that the default policy requires there to be an available action at
    every state.
    """
    return MDPFunction({x: mdp.actions[x][0] for x in mdp.states})


