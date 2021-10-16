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
		# self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr)
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
		# transitions = self.memory.sample(self.BATCH_SIZE)
		# batch = Transition(*zip(*transitions))

		states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)

		actions = actions.astype(np.uint8)

		# Takes the done values and sets the 1s to 0s and the 0s to 1s. Only those state-action values from the target_network at next_states which are not terminal states are used to get the expected_state_action_values. 
		# non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = device, dtype=torch.uint8)
		# non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

		# non_final_next_states = non_final_next_states.unsqueeze(1).float().to(device)
		
		non_final_mask = 1 - dones
		non_final_mask = torch.from_numpy(non_final_mask).float().to(device)
		# print(non_final_mask.dtype)
		# print(states.dtype)
		# print(actions.dtype)
		# print(rewards.dtype)
		# print(next_states.dtype)
		# print(dones.dtype)

		# Concatenate our tensors
		# state_batch = torch.cat(batch.state).unsqueeze(1).float().to(device)
		# action_batch = torch.cat(batch.action)
		# reward_batch = torch.cat(batch.reward)
		# reward_batch = torch.tensor(batch.reward)

		state_batch = torch.from_numpy(states).float().to(device)
		action_batch = torch.from_numpy(actions).long().to(device)
		reward_batch = torch.from_numpy(rewards).float().to(device)
		next_states_batch = torch.from_numpy(next_states).float().to(device)

		# print(action_batch)
		action_batch = action_batch.unsqueeze(1)
		# print(action_batch)

		# state_action_values = self.policy_net(state_batch).gather(1, action_batch)

		# print(state_batch.size())
		# print(next_states_batch.size())
		state_action_values = self.policy_net(state_batch)
		# print(state_action_values.size())
		# print(action_batch.size())
		# print(reward_batch.size())
		# print(action_batch[0])
		# print(state_action_values)
		# print(state_action_values.size())
		state_action_values = state_action_values.gather(1, action_batch)
		# print(state_action_values[0])
		state_action_values = state_action_values.squeeze(1)
		# print(state_action_values.size())
		# print(state_action_values)

		# next_state_values = torch.zeros(self.BATCH_SIZE, device = device)
		# next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

		# print(next_states)
		# sup = True
		# sup2 = None
		# for i in next_states[non_final_mask]:
		# 	if sup2 is not None:
		# 		if not np.array_equal(sup2, i):
		# 			sup = False
		# 	sup2 = i
		# print(sup)

		# sup = True
		# sup2 = None
		# for i in next_states:
		# 	if sup2 is not None:
		# 		if not np.array_equal(sup2, i):
		# 			sup = False
		# 	sup2 = i
		# print(sup)
		# print(next_states[non_final_mask].shape)
		# print(next_states.shape)
		# print(actions)
		# print(next_states_batch)
		# print(next_states_batch.size())

		# next_state_values = torch.zeros(self.BATCH_SIZE, device = device)
		next_state_values = self.target_net(next_states_batch)
		# temp = self.target_net(next_states_batch[non_final_mask])
		# print(temp)
		# print(next_state_values)
		# print(next_state_values.size())
		# print(temp.size())
		# next_state_values[non_final_mask] = self.target_net(next_states_batch[non_final_mask]).max(1)[0].detach()
		# next_state_values[non_final_mask] = temp.max(1)[0].detach()
		# temp = temp.max(1)[0]
		next_state_values = next_state_values.max(1)[0]
		# print(next_state_values)
		# print(temp)
		# print(next_state_values.size())
		# print(temp.size())
		# next_state_values[non_final_mask] = temp.clone().detach()
		next_state_values = next_state_values.detach()
		# print(non_final_mask)
		# print(next_states_batch[non_final_mask])
		# print(next_states_batch[non_final_mask].size())
		# print(next_state_values)
		# print(next_state_values.size())

		next_state_values = non_final_mask * next_state_values

		# print(next_state_values)
		# print(next_state_values.size())

		expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

		# print("Hello")
		# print(type(state_action_values))
		# print(state_action_values.size())
		# print(expected_state_action_values.size())

		# Compute loss between our state action and expectations
		# loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
		# loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
		# loss = nn.functional.l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
		loss = nn.functional.l1_loss(state_action_values, expected_state_action_values)

		# print(self.action_space.n)
		# print(loss.grad_fn)

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
		# self.memory.add(torch.from_numpy(state).unsqueeze(1).float().to(device), action, reward, next_state, done)
		# self.memory.add(torch.from_numpy(state).to(device), action, reward, torch.from_numpy(next_state).to(device), done)
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
			# return torch.tensor([[random.randrange(self.action_space.n)]], device = device, dtype=torch.long)
			return np.random.randint(self.action_space.n)
		else:
			# print(state)
			with torch.no_grad():
				# return self.policy_net(state).max(1)[1].view(1, 1)
				# return self.policy_net(torch.from_numpy(state[None, ...].astype(np.float32)).to(device)).max(1)[1].view(1, 1)

				return self.policy_net(torch.from_numpy(state).unsqueeze(1).float().to(device)).max(1)[1].view(1, 1).item()
		# raise NotImplementedError
