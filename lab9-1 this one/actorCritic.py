import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, firstLayer, secondLayer, thirdLayer, learning_rate=4e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.criticNetwork = nn.Sequential(
            nn.Linear(num_inputs, firstLayer), # take in the state space
            nn.Linear(firstLayer, secondLayer),
            nn.Linear(secondLayer, thirdLayer),
            nn.Linear(thirdLayer, 1) # critic says how good this was
        )
        self.actorNetwork = nn.Sequential(
            nn.Linear(num_inputs, firstLayer), # take in the observation
            nn.Linear(firstLayer, secondLayer),
            nn.Linear(secondLayer, thirdLayer),
            nn.Linear(thirdLayer, num_actions), # actor's final action layer
            nn.Softmax()
            
        )
    def forward(self, state): # pass state in as a tensor
        value  = self.criticNetwork(state.float())
        policy_dist = self.actorNetwork(state.float())
        return value, policy_dist

def setEnv():
    env = gym.make("LunarLander-v2")
    env.seed(5)
    env.reset()
    return env

def getNeuralNetLayers(env, hiddenLayers):
    obs = env.reset()
    layers = []
    layers.append(len(obs))
    for hiddenLayer in hiddenLayers:
        layers.append(hiddenLayer)
    layers.append(env.action_space.n)

def runActorCritic(env, hiddenLayers, learningRate, maxEpisodeLength, maxEpisodes, GAMMA=1):
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    
    actor_critic = ActorCritic(num_inputs, num_outputs, hiddenLayers[0], hiddenLayers[1], hiddenLayers[2])
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learningRate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(maxEpisodes):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        for steps in range(maxEpisodeLength):

            # ensure all tensors are on device 
            # then when pulling from device (the tensor) and trying to convert to numpy, make sure to convert the tensort to a CPU tensor
            value, policy_dist = actor_critic.forward(torch.Tensor(state).to(device))
            value = value.detach().numpy()
            dist = policy_dist.detach().numpy() 

            action = np.random.choice(num_outputs, p=np.squeeze(dist)) 
            log_prob = torch.log(policy_dist[action]) #squeeze(0) converting it from a 2D tensort with each row with 1 elem it is converting it back into a 1D tensor, 
            # then [action] is to select the probability of taking that action - so it is getting the log prob of that action
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            new_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state
            
            if done or steps == maxEpisodeLength-1:
                Qval, _ = actor_critic.forward(torch.Tensor(new_state))
                Qval = Qval.detach().numpy()
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-100:]))
                if episode % 100 == 0:                    
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), steps, average_lengths[-1]))
                break
        
        # compute Q values
        Qvals = np.zeros_like(values) # copies the same shape as values and fills all the elements with zeros
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
  
        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

        
    
    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 100).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()

env = gym.make("LunarLander-v2")
env.seed(0)
# structure the layers to have the first layer be input, the final layer be the output
hiddenLayers = [32,64,32]
layers = getNeuralNetLayers(env, hiddenLayers)

# max length and that we will determine here (not the same as in our alg for project)
maxEpisodes = 5000
maxSteps = 500
gamma = 1
learningRate = 4e-4
runActorCritic(env, hiddenLayers, learningRate, maxSteps, maxEpisodes, GAMMA=1)


