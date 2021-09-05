###
# Group Members
# Name:Student Number
# Name:Student Number
# Name:Student Number
# Name:Student Number
###

import numpy as np
from numpy.core.fromnumeric import argmax
from environments.gridworld import GridworldEnv
import time
import matplotlib.pyplot as plt


def valueOfState(s, action, rewards):
    """
    Given some state see what the neighbours are.
    """
    # go up
    if action == 0:
        # falls off top
        if s < 5:
            return rewards[s]
        # state above
        else:
            return rewards[s-5]
    # go right
    if action == 1:
        # falls off rightside
        if s % 5 == 4:
            return rewards[s]
        # state to right
        else:
            return rewards[s+1]
    # go down
    if action == 2:
        # falls off bottom
        if s > 19:
            return rewards[s]
        # state below
        else:
            return rewards[s+5]
    # go left
    if action == 3:
        # falls off left side
        if s % 5 == 0:
            return rewards[s]
        # state to the left
        else:
            return rewards[s-1]

def createRandomPolicy(rows, cols, numActions):
    space = np.zeros((rows*cols, numActions))
    for i in range(cols * rows):
        for j in range(numActions):
            space[i,j] = 1/numActions
    return space
    
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
    actions = [0, 1, 2, 3]
    while(delta > theta):
        delta = 0
        for s in range(25):
            v = values[s]
            temp = 0
            if s != 24:
                for a in actions:
                    temp += policy[s][a] * ( -1 + valueOfState(s, a, values) )
            values[s] = temp
            delta = max(delta, abs(v - values[s]))
            
        # print(delta)
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
    # randomly creating first set of values
    values = np.random.randint(low = - 11, high = 0, size = 25)
    policy = createRandomPolicy(5, 5, 4)
    theta = 0.00001
    cont = True
    while cont:

        # set the error to 10 initally just so that we can get the loop to execute the first time as 0 < theta always true for non-zero positive theta 
        # Policy Evaluation
        delta = 10
        actions = [0, 1, 2, 3]
        while(delta > theta):
            delta = 0
            for s in range(25):
                v = values[s]
                temp = 0
                if s != 24:
                    for a in actions:
                        temp += policy[s][a] * ( -1 + valueOfState(s, a, values) )
                values[s] = temp
                delta = max(delta, abs(v - values[s]))
        
        # Policy Improvement 
        stable = True
        for s in range(25):
            # find the old action that had the greatest probability
            oldAction = argmax(policy[s,:])
            tempVals = []
            for a in range(4):
                tempVals.append( -1 + valueOfState(s, a, values))
            # get the best action
            bestAct = argmax(tempVals)
            policy[s,:] = [0,0,0,0]
            policy[s,bestAct] = 1
            if oldAction != bestAct:
                stable = False
        if stable == True:
            cont = False

    return policy, values

    # did not use this function as I made my own
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
    values = np.random.randint(low = - 11, high = 0, size = 25)
    policy = createRandomPolicy(5, 5, 4)
    actions = []
    for a in range(env.action_space.n):
        actions.append(a)
    cont = True
    while cont:
        delta = 0
        for s in range(25):
            # terminal state must stay 0
            if s == 24:
                values[s] = 0
            # update the non-terminal states
            else:
                v = values[s]
                tempArr = []
                for a in actions:
                    tempArr.append(-1 + discount_factor * valueOfState(s, a, values))
                values[s] = max(tempArr)
                env.P
                policy[s,:] = [0, 0, 0, 0]
                policy[s][argmax(tempArr)] = 1
                delta = max(delta, abs(v - values[s]))
        if delta < theta:
            cont = False
    return policy, values
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

def printStateValues(values):
    for i in range(5):
        temp = ""
        for j in range(5):
            temp += str(values[i*5 + j])
            if j < 4:
                temp += " "
        print(temp)
    return

def printDirs(policy):
    for i in range(5):
        temp = ""
        for j in range(5):
            if i*5 + j < 24:
                dir = argmax(policy[i*5+j,:])
                if dir == 0:
                    temp += "U"
                elif dir == 1:
                    temp += "R"
                elif dir == 2:
                    temp += "D"
                else:
                    temp += "L"
            else:
                temp += "T"
            if j < 4:
                temp += " "
        print(temp)
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
    # the below matrix would be a # [5x5, 4] shaped matrix
    randomPolicy = createRandomPolicy(rows=5, cols=5, numActions=4)
    printRandomWorldMoves(env)
    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    # TODO: evaluate random policy
    policy = np.ones((25,4)) / 4
    v = policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001)

    # TODO: print state value for each state, as grid shape
    printStateValues(v)


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
    policy, v = policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0)
    
    # TODO Print out best action for each state in grid shape
    print("POLICY DIRECTIONS ARE AS FOLLOWS \n ############################")
    printDirs(policy)
    print("##########################")
    # TODO: print state value for each state, as grid shape
    print("VALUES ARE AS FOLLOWS \n ############################")
    printStateValues(v)
    print("##########################")
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
    policy, v = value_iteration(env, theta=0.0001, discount_factor=1.0)

    print("POLICY DIRECTIONS ARE AS FOLLOWS \n ############################")
    printDirs(policy)
    print("##########################")
    # TODO: print state value for each state, as grid shape
    print("VALUES ARE AS FOLLOWS \n ############################")
    printStateValues(v)
    print("##########################")

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    # Mikayla
    # TODO
    # The function value iteration is the part that was meant to be done, but I finished off the entire lab just to do it all
    # The final requirement is just to plot the average running time for policy iteration and value iteration 
    # by varying the discount rate.
    # So these functions just need to take in a specific discount rate and then record the runtime in seconds/milliseconds.
    
    discounts = np.logspace(-0.2, 0, num=30)
    policyTimes = np.array(())
    for discount in discounts:
        policy = createRandomPolicy(5,5,4)
        timeIter = 0
        
        for i in range(10):
            start = time.time()
            policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=discount)
            timeIter += time.time() - start
        timeIter /= 10
        policyTimes = np.append(policyTimes, timeIter)
        print("Policy Iteration: \t Discount factor: {0}, time taken: {1}".format(discount, timeIter))
    
    print("\n ################################################################ \n")

    valueTimes = np.array(())
    for discount in discounts:
        timeIter = 0
        
        for i in range(10):
            start = time.time()
            value_iteration(env, theta=0.0001, discount_factor=discount)
            timeIter += time.time() - start
        timeIter /= 10
        valueTimes = np.append(valueTimes, timeIter)
        print("Value Iteration: \t Discount factor: {0}, time taken: {1}".format(discount, timeIter))

    discountInfo = []
    for discount in discounts:
        discountInfo.append( str(round(discount,2)))

    plt.bar(discountInfo, policyTimes, label="Policy Iteration")
    plt.bar(discountInfo, valueTimes, label="Value Iteration")
    plt.title("Comparison of Average Running Times Iteration Approaches")
    plt.ylabel("Run time (seconds)")
    plt.xlabel("Discount Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
