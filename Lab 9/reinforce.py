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
        nn.Module.__init__(self)
        # actual network
        self.model = torch.nn.Sequential(
            torch.nn.Linear(s_size, h_size),
            torch.nn.ReLU(),
            torch.nn.Linear(h_size, a_size),
            torch.nn.Softmax(dim=0)
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = self.softmax(x)
        return x    


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def compute_returns(rewards, gamma):
    # based on summation given in the pseudocode
    # the reward would be G = gamma^(k-t-1) * rewards_k
    exponents = np.arange(start = 0, stop = len(rewards))
    gammas = gamma ** exponents
    returns = np.array(gammas * rewards)
    return np.sum(returns)


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
            print("\t Current run is {0}".format(trajectory))
        env.seed(int(seed))  
        currState = env.reset()
        if firstStateSeen == False:
            resetState = currState
        else:
            for st in range(len(resetState)):
                if resetState[st] != currState:
                    raise EnvironmentError
        done = False
        transitions = []
        for t in range(max_episode_length):
            actProb = policy_model.model(torch.from_numpy(currState).float())
            action = np.random.choice(np.array([0,1]), p=actProb.data.numpy())
            prevState = currState
            currState, rewards, done, info = env.step(action)
            transitions.append((prevState, action, t+1))
            if done:
                break
        scores.append(len(transitions))
        rewardBatch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))
        batchGvals = []
        for i in range(len(transitions)):
            newGVal = 0
            power = 0
            for j in range(i, len(transitions)):
                newGval = newGVal + ((gamma**power)*rewardBatch[j]).numpy()
                power += 1
            batchGvals.append(newGval)
        expectedReturnsBatch = torch.FloatTensor(batchGvals)
        expectedReturnsBatch /= expectedReturnsBatch.max()

        stateBatch = torch.Tensor([s for (s,a,r) in transitions])
        actionBatch = torch.Tensor([a for (s,a,r) in transitions])
        predBatch = policy_model.model(stateBatch)
        probBatch = predBatch.gather(dim=1, index=actionBatch.long().view(-1,1)).squeeze()
        loss = -torch.sum(torch.log(probBatch) * expectedReturnsBatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return policy_model, scores


def compute_returns_naive_baseline(rewards, gamma):
    raise NotImplementedError
    return returns


def reinforce_naive_baseline(env, policy_model, seed, learning_rate,
                             number_episodes,
                             max_episode_length,
                             gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)

    raise NotImplementedError
    return policy, scores


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
    seeds = np.random.randint(1000, size=5)
    scores = []
    seeds = list(seeds)
    for seed in seeds:
        _, score =run_reinforce(seed = seed)
        scores.append(score)
    
    scores = np.array(scores)
    mean = np.mean(scores)
    var = np.var(scores)

    steps = 1500
    y1 = mean - std
    y2 = mean + std
    plt.fill_between(steps, y1, y2, where=y1>=y2, facecolor='blue')
    plt.fill_between(steps, y1, y2, where=y2>=y1, facecolor='blue')
    return mean, std


def run_reinforce_with_naive_baseline(mean, std):
    env = gym.make('CartPole-v1')

    np.random.seed(53)
    seeds = np.random.randint(1000, size=5)
    raise NotImplementedError


if __name__ == '__main__':
    model = SimplePolicy()
    print(model.model)
    # run_reinforce()
    mean, std = investigate_variance_in_reinforce()
    print(mean)
    print(std)
    run_reinforce_with_naive_baseline(mean, std)
