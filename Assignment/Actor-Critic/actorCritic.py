import torch  
import numpy as np  
import torch.nn as nn
import torch.optim as optim
from wrapper import BasicWrapper, customGym
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, firstLayer, secondLayer, thirdLayer, fourthLayer, fifthLayer, learning_rate=4e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.criticNetwork = nn.Sequential(
            nn.Linear(num_inputs, firstLayer), # take in the state space
            nn.LeakyReLU(),
            nn.Linear(firstLayer, secondLayer), 
            nn.LeakyReLU(),
            nn.Linear(secondLayer, thirdLayer),
            nn.LeakyReLU(),
            nn.Linear(thirdLayer, 1),
        )
        self.actorNetwork = nn.Sequential(
            nn.Linear(num_inputs, firstLayer), # take in the state space
            nn.LeakyReLU(),
            nn.Linear(firstLayer, secondLayer), 
            nn.LeakyReLU(),
            nn.Linear(secondLayer, thirdLayer),
            nn.LeakyReLU(),
            nn.Linear(thirdLayer, num_actions),
            nn.Softmax() # ensure the agent takes an action with probabilities that sum to one
            
        )
    def forward(self, state): # pass state in as a tensor
        value  = self.criticNetwork(state.float())
        policy_dist = self.actorNetwork(state.float())
        return value, policy_dist

def getNeuralNetLayers(env, hiddenLayers):
    obs = env.reset()
    layers = []
    layers.append(len(obs))
    for hiddenLayer in hiddenLayers:
        layers.append(hiddenLayer)
    layers.append(env.action_space.n)


def runActorCritic(env, hiddenLayers, learningRate, maxEpisodeLength, maxEpisodes, num_inputs, num_outputs, GAMMA=0.9, stepSize=5, eps=1e-5):
    
    actor_critic = ActorCritic(num_inputs, num_outputs, hiddenLayers[0], hiddenLayers[1], hiddenLayers[2], hiddenLayers[3], hiddenLayers[4])
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learningRate)
    global maxSteps
    global seed
    lengths = []
    all_rewards = []
    entropy_term = 0
    for episode in range(maxEpisodes):
        env.fullReset()
        log_probs = []
        values = []
        rewards = []
        env.seed(0)
        state = env.reset()
        # if episode > maxEpisodes-5:
        #     input("Time to view it")
        for steps in range(maxEpisodeLength):
            
            # ensure all tensors are on device 
            # then when pulling from device (the tensor) and trying to convert to numpy, make sure to convert the tensort to a CPU tensor
            value, policy_dist = actor_critic.forward(torch.Tensor(state).to(device))
            value = value.detach().numpy()
            values.append(value)
            dist = policy_dist.detach().numpy() 

            if episode % 5 == 0:
                if steps % 45 == 0:
                    env.render() 
                
            action = np.random.choice(num_outputs, p=np.squeeze(dist)) 
            log_prob = torch.log(policy_dist[action]) #squeeze(0) converting it from a 2D tensort with each row with 1 elem it is converting it back into a 1D tensor, 
            log_probs.append(log_prob)
            # then [action] is to select the probability of taking that action - so it is getting the log prob of that action
            entropy_term -= np.sum(np.mean(dist) * (np.log(dist + eps)))
            new_state, reward, done, info = env.step(action, maxLength= maxSteps)
            rewards.append(reward)
            state = new_state

            
            if done or steps == maxEpisodeLength-1:
                Qval, val = actor_critic.forward(torch.Tensor(new_state).to(device))
                Qval = Qval.detach().numpy()
                all_rewards.append(np.sum(rewards))
                lengths.append(steps)
                if episode % stepSize == 0:                    
                    print("episode: {0}, avg reward: {1}, avg length: {2}".format(episode, np.mean(all_rewards[-stepSize:]), np.mean(lengths[-stepSize:])))
                break
        
        # compute Q values
        Qvals = np.zeros_like(values) # copies the same shape as values and fills all the elements with zeros
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval
  
        #update actor critic
        values = np.array(values)
        values = torch.FloatTensor(values).to(device)
        Qvals = np.array(Qvals)
        Qvals = torch.FloatTensor(Qvals).to(device)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()
    global net

    torch.save(actor_critic.actorNetwork.state_dict(), "./actorNetwork/modelNegativeRewards-maxEpisodes-{0}-maxSteps-{1}-{2}".format(maxEpisodes, maxSteps,net))
    torch.save(actor_critic.criticNetwork.state_dict(), "./criticNetwork/modelNegativeRewards-maxEpisodes-{0}-maxSteps-{1}-{2}".format(maxEpisodes, maxSteps,net))
    
    return rewards, lengths

maxSteps = 100
seed=(0)
env = BasicWrapper(customGym(maxLength=maxSteps, seed=seed), maxSteps)
num_inputs = len(env.reset())
num_outputs = env.action_space.n
net = 1
hiddenLayers = [2,2,2,2,2] * net
layers = getNeuralNetLayers(env, hiddenLayers)
# max length and that we will determine here (not the same as in our alg for project)
maxEpisodes = 5
learningRate = 5e-7
rewards, length = runActorCritic(env, hiddenLayers, learningRate, maxSteps, maxEpisodes, num_inputs, num_outputs)
rewards = np.array(rewards)
length = np.array(length)
np.savetxt("rewardsActorCritic{0}smallnet{1}episodes-{2}-net.txt".format(maxSteps,maxEpisodes, net),rewards)
np.savetxt("lengthsActorCritic{0}smallnet{1}episodes-{2}-net.txt".format(maxSteps,maxEpisodes, net),length)


