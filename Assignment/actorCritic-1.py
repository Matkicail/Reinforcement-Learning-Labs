# action-critic part
import os #file moving
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

# Agent stuff
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import collections
# from ____ import ActorCriticNetwork 

# environment / gym
# !pip install utils
import gym
import minihack
import numpy as np
# from actor_critic import Agent

import matplotlib.pyplot as plt

# BOTTOM USELESS
def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

class ActorCriticNetwork(keras.Model):
  def __init__(self, n_actions, fc1_dims= SUBJECT_TO_ENVIRONMENT_OBSERVATION_SPACE, fc2_dims=512, name='actor_critic', chkpt_dir = 'tmp/actor_critic'):
    super(ActorCriticNetwork, self).__init__()

	# FIRST SET OF DIMS WILL BE THE INPUT FROM ENVIRONMENT
	# FOLLOWING FROM THIS - NO REQUIREMENTS ARE SPECIFIC - I.E WHAT YOU FEEL IS CORRECT (should be done)
	# Output for actor is an action so [0,num_actions) and softmax to determine which action
	# Output from critic is just a value (value of reward / or critic's view)
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.n_actions = n_actions
	# keeping track - book-keeping (I.E SAVING NEURAL NETWORK
    self.model_name = name
    self.checkpoint_dir = chkpt_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_ac")

    self.fc1 = Dense(self.fc1_dims, activation='relu') # number of input dims are specificed
    self.fc2 = Dense(self.fc2_dims, activation='relu')
    self.value = Dense(1, activation= None) # value of what did
    self.policy = Dense(n_actions, activation='softmax') # to get action

    # feed forward
  def call(self, state):
    value = self.fc1(state)
    value = self.fc2(value)
    v  = self.value(value)
    pi = self.policy(value)
	# v is how the critic views you
	# pi is the policy of the actor
    return v, pi

class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=78):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))


	# what is the world at current
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state) # underscore represented useless

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        self.action = action

        return action.numpy()[0]

    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)
        
    def learn(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32) #St
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32) #St-1
        reward = tf.convert_to_tensor(reward, dtype=tf.float32) # not fed to NN
        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta # ACTOR MADE SOME ERROR
            critic_loss = delta**2 # CRITIC MADE SOME ERROR AND THIS IS RELEVANT
            total_loss = actor_loss + critic_loss # NEED SOME WAY OF GETTING TOTAL NETWORK'S LOSS FOR BOTH PEOPLE

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))

# 
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return items

# code to run env - MATTHEW WILL DEAL WITH THE ENVIRONMENT TO GET IT SATISFACTORY
# make a gym wrapper - need to keep base functions.
env = gym.make("MiniHack-Quest-Hard-v0")
agent = Agent(alpha = 1e-5, n_actions = env.action_space.n)
n_games = 1800
filename= "cartpole.png"
figure_file = "plots/" + filename
best_score = env.reward_range[0]
score_history = []
load_checkpoint = False 

if load_checkpoint:
    agent.load_models()

# ACTUAL CODE TO RUN GAME 
for i in range(n_games):

	# HERE IS MATTHEW GETTING THE INPUT VECTOR
    observation = env.reset() # we always start at the same the same state
    observation = flatten(observation)
    observation = np.array(observation)[:,1]
    observation = np.array([i.astype(int) for i in observation])
    temp = []
    for obs in observation:
        obs = list(obs.flatten())
        for item in obs:
            temp.append(item)
    observation = np.array(temp).flatten()
    


    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)

	# HERE IS MATTHEW GETTING THE INPUT VECTOR
        observation_, reward, done, info = env.step(action)
        score += reward
        if not load_checkpoint:
            agent.learn(observation, reward, observation_, done)
        observation = observation_

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint:
            agent.save_models()

    print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

if not load_checkpoint:
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)

  


# %%



