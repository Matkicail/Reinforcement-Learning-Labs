###
# Group Members
# Name:Student Number
# Name:Student Number
# Name:Student Number
# Name:Student Number
###

import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt


def valueOfState(s, action, rewards):
    """
    Given some state see what the neighbours are.
    """
    if action == 0:
        if s < 5:
            return rewards[s]
        else:
            return rewards[s-5]
    if action == 2:
        if s > 19:
            return rewards[s]
        else:
            return rewards[s+5]
    if action == 1:
        if (s) % 4 == 0:
            return rewards[s]
        else:
            return rewards[s+1]
    if action == 3:
        if s % 5 == 0:
            return rewards[s]
        else:
            return rewards[s-1]

    
def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:

        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """
    values = []
    # initialise values
    for i in range(25):
        values.append(0)
    # set the error to 10 initally just so that we can get the loop to execute the first time as 0 < theta always true for non-zero positive theta 
    delta = 10
    pi = 0.25
    actions = [0, 1, 2, 3]
    while(delta > theta):
        delta = 0
        for s in range(24):
            v = values[s]
            temp = 0
            for a in actions:
                temp += policy[s][a] * ( -1 + valueOfState(s, a, values) )
            values[s] = temp
            delta = max(delta, abs(v - values[s]))
            
        print(delta)
    return values 
     
def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        raise NotImplementedError

    raise NotImplementedError


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        raise NotImplementedError

    raise NotImplementedError

def createRandomPolicy(rows, cols, numActions):
    space = np.zeros((rows*cols, numActions))
    for i in range(cols * rows):
        for j in range(numActions):
            space[i,j] = 1/numActions
    return space

def printWorldMoves(world):
    for i in range(5):
        temp = ""
        for j in range(5):
            temp += world[i*5 + j]
            if j < 4:
                temp += " "
        print(temp)
    return

def printRandomWorldMoves(env):
    world = []
    for i in range(24):
        world.append('o')
    world.append('T')
    obs = env.reset()
    firstObs = obs
    if obs != 24:
        world[obs] = "X"
    for i in range(20):
        # four cardinal dir
        action = int(np.random.randint(low=0,high=4))
        # not the first X and not the terminal state
        obs = env.step(action)
        if obs[0] != 24:
            if action == 0:
                world[obs[0]] = "U"
            elif action == 1:
                world[obs[0]] = "R"
            elif action == 2:
                world[obs[0]] = "D"
            elif action == 3:
                world[obs[0]] = "L"
        print("-------------")
        printWorldMoves(world)
    world[obs[0]] = "X"
    return 
def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    
    state = env.reset()
    print("")
    env.render()
    print("")

    # TODO: generate random policy
    randomPolicy = createRandomPolicy(rows=5, cols=5, numActions=4)
    printRandomWorldMoves(env)
    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    # TODO: evaluate random policy
    policy = np.ones((25,4)) / 4
    v = policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001)

    # TODO: print state value for each state, as grid shape

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    # TODO: use  policy improvement to compute optimal policy and state values
    policy, v = [], []  # call policy_iteration

    # TODO Print out best action for each state in grid shape

    # TODO: print state value for each state, as grid shape

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    # TODO: use  value iteration to compute optimal policy and state values
    policy, v = [], []  # call value_iteration

    # TODO Print out best action for each state in grid shape

    # TODO: print state value for each state, as grid shape

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)


if __name__ == "__main__":
    main()
