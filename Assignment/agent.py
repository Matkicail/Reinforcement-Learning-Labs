from gym import spaces
import random
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

device = "cuda"

# Memory representation of states
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNAgent:
	def __init__(
		self,
		observation_space: spaces.Box,
		action_space: spaces.Discrete,
		replay_buffer: ReplayBuffer,
		use_double_dqn,
		lr,
		batch_size,
		gamma,
	):
		"""
		Initialise the DQN algorithm using the Adam optimiser
		:param action_space: the action space of the environment
		:param observation_space: the state space of the environment
		:param replay_buffer: storage for experience replay
		:param lr: the learning rate for Adam
		:param batch_size: the batch size
		:param gamma: the discount factor
		"""

		# TODO: Initialise agent's networks, optimiser and replay buffer
		self.observation_space = observation_space
		self.action_space = action_space
		self.policy_net = DQN(self.observation_space,  self.action_space).to(device)
		self.target_net = DQN(self.observation_space,  self.action_space).to(device)
		self.optimizer = optim.Adam(self.policy_net.parameters(), lr)
		self.memory = replay_buffer
		self.BATCH_SIZE = batch_size
		self.GAMMA = gamma
		# raise NotImplementedError

	def optimise_td_loss(self):
		"""
		Optimise the TD-error over a single minibatch of transitions
		:return: the loss
		"""
		# TODO
		#   Optimise the TD-error over a single minibatch of transitions
		#   Sample the minibatch from the replay-memory
		#   using done (as a float) instead of if statement
		#   return loss

		if len(self.memory) < self.BATCH_SIZE:
			return

		# Sample from our memory
		states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)
		
		actions = actions.astype(np.uint8)
		
		non_final_mask = 1 - dones
		non_final_mask = torch.from_numpy(non_final_mask).float().to(device)

		state_batch = torch.from_numpy(states).float().to(device)
		action_batch = torch.from_numpy(actions).long().to(device)
		reward_batch = torch.from_numpy(rewards).float().to(device)
		next_states_batch = torch.from_numpy(next_states).float().to(device)

		action_batch = action_batch.unsqueeze(1)
		print(action_batch.size())

		state_action_values = self.policy_net(state_batch)
		print(state_action_values.size())
		state_action_values = state_action_values.gather(1, action_batch)
		state_action_values = state_action_values.squeeze(1)

		next_state_values = self.target_net(next_states_batch)
		next_state_values = next_state_values.max(1)[0]
		next_state_values = next_state_values.detach()

		next_state_values = non_final_mask * next_state_values
		expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

		loss = nn.functional.l1_loss(state_action_values, expected_state_action_values)
		self.optimizer.zero_grad()
		loss.backward()

		for param in self.policy_net.parameters():
			param.grad.data.clamp_(-1, 1)

		self.optimizer.step()

		return loss
		# raise NotImplementedError

	def update_target_network(self):
		"""
		Update the target Q-network by copying the weights from the current Q-network
		"""
		# TODO update target_network parameters with policy_network parameters
		self.target_net.load_state_dict(self.policy_net.state_dict())
		# raise NotImplementedError

	def act(self, state: np.ndarray, sample, eps_threshold):
		"""
		Select an action greedily from the Q-network given the state
		:param state: the current state
		:return: the action to take
		"""
		# TODO Select action greedily from the Q-network given the state
		if sample < eps_threshold:
			return np.random.randint(self.action_space.n)
		else:
			with torch.no_grad():
				# return self.policy_net(torch.from_numpy(state).unsqueeze(1).float().to(device)).max(1)[1].view(1, 1).item()
				x = self.policy_net(torch.from_numpy(state).float().to(device))
				print(x)
				x = x.max(0)[1].view(1, 1).item()
				print(x)
				return x
		# raise NotImplementedError
