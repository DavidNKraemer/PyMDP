## Policy Iteration ##
from numpy import eye, array
from numpy.linalg import solve

from pymdp.mdp import MarkovDecisionProcess, default_value, default_policy
from pymdp.bellman import bellman_difference


def policy_iteration(mdp: MarkovDecisionProcess):
    """
    Performs policy iteration on a given infinite horizon MDP.
    """
    policy = default_policy(mdp)
    value = solve_for_value(mdp, policy)

    improvements = determine_improvements(mdp, value)

    while can_improve(improvements):
        #print(policy)
        #print(value)
        policy = improve_policy(mdp, policy, value, improvements)
        value = solve_for_value(mdp, policy)
        improvements = determine_improvements(mdp, value)

    #print(value)
    return policy


def policy_transition(mdp, policy):
    """
    Computes the transition matrix corresponding to a particular policy
    function, so that the (i,j)th entry in the matrix corresponds to the
    probability of arriving at state i from state j through policy[j].
    """
    result = array(
        [
            [mdp.transition(y, x, policy[x]) for y in mdp.states] \
                    for x in mdp.states])

    return result


def policy_reward(mdp, policy):
    """
    For each $x \in X$
    $$ r(\phi)_x = \sum_{y \in X} P(y \mid x, \phi(x)) r(y, \phi(x)) $$
    """
    return array([sum(mdp.reward(y, policy[x]) * mdp.transition(y, x, policy[x]) \
        for y in mdp.states) \
        for x in mdp.states])


def solve_for_value(mdp, policy):
    """
    Solves the system
    $$
    (I - \beta P(policy)) v = r(policy)
    $$
    for the value function v.
    """
    system_matrix = eye(len(mdp.states)) - mdp.discount * policy_transition(mdp, policy)
    expected_reward = policy_reward(mdp, policy)

    result = solve(system_matrix, expected_reward)

    return {x: result[k] for (k, x) in enumerate(mdp.states)}



def determine_improvements(mdp, value):
    """
    Computes, for each state, the set of all associated actions for which the
    value function would increase if the policy were to adopt them.

    The state gets skipped if no improvements can be made in it. Thus the
    returned improvements are only those which *strictly* improve the policy's
    associated value function.
    """
    improvements = {x: [a for a in mdp.actions[x] \
        if bellman_difference(mdp, value, x, a) > 0.] for x in mdp.states}
    return {x: improvements[x] for x in improvements if len(improvements[x]) > 0}


def can_improve(improvements):
    """
    A boolean check on whether any improvement is actually possible from the
    present policy.
    """
    return len(improvements) > 0


def improve_policy(mdp, policy, value, improvements):
    """
    Given a policy, value function, and set of state-action combinations that
    can modify the policy so that the value function increases strictly, do just
    that.
    """
    def select_improvement(state):
        """
        For a given state, select the maximum improving action to be the new
        poliy. Otherwise, leave alone the original policy.
        """
        if state in improvements:
            return max(
                ((a, bellman_difference(mdp, value, state, a)) \
                    for a in improvements[state]),
                key=lambda pair: pair[1]
                )[0]
        else:
            return policy[state]

    return {x: select_improvement(x) for x in mdp.states}
