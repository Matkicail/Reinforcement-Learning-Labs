import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
from torch import optim
import torch.nn as nn

import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimplePolicy(nn.Module):

    # https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
    # https://towardsdatascience.com/policy-gradient-methods-104c783251e0
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super().__init__()
        # actual network
        self.model = torch.nn.Sequential(
            torch.nn.Linear(s_size, h_size),
            torch.nn.ReLU(), # ask michael what he thinks we could replace this with
            torch.nn.Linear(h_size, a_size),
            torch.nn.Softmax(dim=0)
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        

    def forward(self, x): # need x as a tensor
        return self.model(x.float()) 


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def compute_returns(rewards, gamma):
    # returns the full sum rather than just an array of reward ones which we want to get using our batch method - since better
    exponents = np.arange(start = 0, stop = len(rewards))
    gammas = gamma ** exponents
    returns = np.array(gammas * rewards)
    return np.sum(returns)

def compute_returns_batch(rewards, gamma):
    exponents = np.arange(start = 0, stop = len(rewards))
    gammas = gamma ** exponents
    returns = np.array(gammas * rewards)
    return returns

def reinforce(env, policy_model, seed, learning_rate,
              number_episodes,
              max_episode_length,
              gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(type(seed))
    try:
        seed = int(seed[0])
    except:
        pass
    optimizer = optim.Adam(policy_model.model.parameters(), lr=learning_rate)
    scores = []
    firstStateSeen = False
    resetState = []
    # for num of episodes
    for trajectory in range(number_episodes):
        if trajectory % 100 == 0:
            # env.render()
            print("\t Current run is {0}".format(trajectory))
        env.seed(int(seed))  
        currState = env.reset()
        if firstStateSeen == False:
            resetState = currState
        else:
            for st in range(len(resetState)):
                if resetState[st] != currState:
                    raise EnvironmentError

        states = []
        rewards = []
        actions = []
        length = 0
        for t in range(max_episode_length):
            # if trajectory % 100 == 0:
            #     print("Current step: {0} \r".format(t), end="\r")
            #     env.render()
            actProb = policy_model(torch.from_numpy(currState)) # inside here is the creation of a float, this automatically calls forward pass
            action = np.random.choice(np.array([0,1]), p=actProb.data.numpy()) # choose an action at random
            actions.append(action)
            prevState = currState
            states.append(prevState)
            currState, reward, done, info = env.step(action)
            rewards.append(reward) # every time it successfully held up the pole give it a reward
            # baseline could be instead to give it t + 1 for every step it survives
            length = t + 1
            if done:
                break
        scores.append(length) # how long did it keep the pole up
        
        # preventing issues for torch
        rewards = np.array(rewards)
        actions = np.array(actions)
        states = np.array(states)
        rewardBt = torch.Tensor(rewards).flip(dims=(0,))
        vals = compute_returns_batch(rewardBt.numpy(), gamma)
        batchGvals = torch.FloatTensor(compute_returns_batch(rewardBt.numpy(),gamma))
        expectedReturnsBatch = torch.FloatTensor(batchGvals)
        expectedReturnsBatch /= expectedReturnsBatch.max()

        stateBt = torch.Tensor(states)
        actionBt = torch.Tensor(actions)
        predBatch = policy_model(stateBt) # will push through the network
        probBatch = predBatch.gather(dim=1, index=actionBt.long().unsqueeze(1)).squeeze()
        loss = -torch.sum(torch.log(probBatch) * expectedReturnsBatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return policy_model, scores


def compute_returns_naive_baseline(rewards, gamma):
    exponents = np.arange(start = 0, stop = len(rewards))
    baseLine = np.arange(start = len(rewards), stop = 0, step = -1) # 
    rewards -= baseLine
    gammas = gamma ** exponents
    returns = np.array(gammas * rewards)
    return np.sum(returns)

def compute_returns_naive_baseline_batch(rewards, gamma):
    exponents = np.arange(start = 0, stop = len(rewards))
    baseLine = np.arange(start = len(rewards), stop = 0, step = -1) # 
    rewards -= baseLine
    gammas = gamma ** exponents
    returns = np.array(gammas * rewards)
    return returns

def reinforce_naive_baseline(env, policy_model, seed, learning_rate,
                             number_episodes,
                             max_episode_length,
                             gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(type(seed))
    try:
        seed = int(seed[0])
    except:
        pass
    optimizer = optim.Adam(policy_model.model.parameters(), lr=learning_rate)
    scores = []
    firstStateSeen = False
    resetState = []
    # for num of episodes
    for trajectory in range(number_episodes):
        if trajectory % 100 == 0:
            # env.render()
            print("\t Current run is {0}".format(trajectory))
        env.seed(int(seed))  
        currState = env.reset()
        if firstStateSeen == False:
            resetState = currState
        else:
            for st in range(len(resetState)):
                if resetState[st] != currState:
                    raise EnvironmentError

        states = []
        rewards = []
        actions = []
        length = 0
        for t in range(max_episode_length):
            # if trajectory % 100 == 0:
            #     print("Current step: {0} \r".format(t), end="\r")
            #     env.render()
            actProb = policy_model(torch.from_numpy(currState)) # inside here is the creation of a float, this automatically calls forward pass
            action = np.random.choice(np.array([0,1]), p=actProb.data.numpy()) # choose an action at random
            actions.append(action)
            prevState = currState
            states.append(prevState)
            currState, reward, done, info = env.step(action)
            rewards.append(reward) # every time it successfully held up the pole give it a reward
            # baseline could be instead to give it t + 1 for every step it survives
            length = t + 1
            if done:
                break
        scores.append(length) # how long did it keep the pole up
        
        # preventing issues for torch
        rewards = np.array(rewards)
        actions = np.array(actions)
        states = np.array(states)
        rewardBt = torch.Tensor(rewards).flip(dims=(0,))
        vals = compute_returns_naive_baseline_batch(rewardBt.numpy(), gamma)
        batchGvals = torch.FloatTensor(compute_returns_naive_baseline_batch(rewardBt.numpy(),gamma))
        expectedReturnsBatch = torch.FloatTensor(batchGvals)
        expectedReturnsBatch /= expectedReturnsBatch.max()

        stateBt = torch.Tensor(states)
        actionBt = torch.Tensor(actions)
        predBatch = policy_model(stateBt) # will push through the network
        probBatch = predBatch.gather(dim=1, index=actionBt.long().unsqueeze(1)).squeeze()
        loss = -torch.sum(torch.log(probBatch) * expectedReturnsBatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return policy_model, scores

def run_reinforce(seed=42):
    env = gym.make('CartPole-v1')
    policy_model = SimplePolicy(s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)
    policy, scores = reinforce(env=env, policy_model=policy_model, seed=seed, learning_rate=1e-2,
                               number_episodes=1500,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=True)
    
    moveAve = moving_average(scores, 50)
    plt.plot(scores, label="score")
    plt.plot(moveAve, label="moving-average-50")
    plt.xlabel("# Episodes")
    plt.ylabel(" Score ")
    plt.legend()
    plt.title("Reinforce Learning")
    plt.show()

def investigate_variance_in_reinforce():
    env = gym.make('CartPole-v1')
    print("Investigating reinforce")
    seeds = np.random.randint(1000, size=5)
    numEpisode = 1500
    scores = []
    for seed in seeds:
        policy_model = SimplePolicy(s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)
        policy_model, score = reinforce(env=env, policy_model=policy_model, seed=seed.copy(), learning_rate=1e-3,
                               number_episodes=numEpisode,
                               max_episode_length=1000,
                               gamma=1.0,
                               verbose=True)
        scores.append(score)

    scores = np.array(scores)
    print("Saving scores: shape is - {0}".format(scores.shape))
    np.savetxt("reinforce_standard_scores.txt", scores)
    mean = np.sum(scores,axis=0)/numEpisode
    print(len(scores))
    var = np.std(scores, axis=0)
    movAvgRet = moving_average(np.mean(scores,axis=0), 50)
    steps = np.arange(start = 0 , stop = numEpisode)
    y1 = mean - np.sqrt(var)
    y2 = mean + np.sqrt(var)
    fig, ax = plt.subplots()
    ax.plot(steps, movAvgRet)
    ax.fill_between(steps, y1,y2, alpha=0.2)
    plt.title("Averaged returns")
    plt.show()
    return mean, np.sqrt(var)


def run_reinforce_with_naive_baseline(mean, std): # not actually using the mean and std as I found a better baseline
    env = gym.make('CartPole-v1')
    scores = []
    seeds = np.random.randint(1000, size=1)
    for seed in seeds:
        policy_model = SimplePolicy(s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)
        policy_model, score= reinforce_naive_baseline(env, policy_model, seed, 1e-2,
                                1500,
                                1000,
                                1, verbose=True)
        scores.append(score)
    scores = np.array(scores)
    print("Saving scores: shape is - {0}".format(scores.shape))
    np.savetxt("reinforce_baseline_standard_scores_one_run.txt", scores)

def investigate_variance_in_reinforce_baseline(mean, std): # not actually using the mean and std as I found a better baseline
    env = gym.make('CartPole-v1')
    scores = []
    seeds = np.random.randint(1000, size=5)
    for seed in seeds:
        policy_model = SimplePolicy(s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)
        policy_model, score= reinforce_naive_baseline(env, policy_model, seed, 1e-2,
                                1500,
                                1000,
                                1, verbose=True)
        scores.append(score)
    scores = np.array(scores)
    print("Saving scores: shape is - {0}".format(scores.shape))
    np.savetxt("reinforce_baseline_standard_scores.txt", scores)

if __name__ == '__main__':
    model = SimplePolicy()
    print(model.model)
    mean = 0 # place holder values sin case want to just run baseline directly
    std = 0 # place holder values sin case want to just run baseline directly
    # run_reinforce()
    mean, std = investigate_variance_in_reinforce()
    # print(mean)
    # print(std)
    # investigate_variance_in_reinforce_baseline(mean, std)
