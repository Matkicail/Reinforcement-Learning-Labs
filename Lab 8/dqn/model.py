from gym import spaces
import torch
import torch.nn as nn

class DQN(nn.Module):
	"""
	A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
	Nature DQN paper.
	"""

	def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
		"""
		Initialise the DQN
		:param observation_space: the state space of the environment
		:param action_space: the action space of the environment
		"""
		super().__init__()
		assert (
			type(observation_space) == spaces.Box
		), "observation_space must be of type Box"
		assert (
			len(observation_space.shape) == 3
		), "observation space must have the form channels x width x height"
		assert (
			type(action_space) == spaces.Discrete
		), "action_space must be of type Discrete"

		# TODO Implement DQN Network
		print(observation_space.shape)
		self.conv1 = nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.lin1 = nn.Linear(3136, 512)
		self.head = nn.Linear(512, action_space.n)
        # raise NotImplementedError

	def forward(self, x):
		# TODO Implement forward pass
		x = nn.functional.relu(self.bn1(self.conv1(x)))
		x = nn.functional.relu(self.bn2(self.conv2(x)))
		x = nn.functional.relu(self.bn3(self.conv3(x)))
		x = x.view(x.size(0), -1)
		x = nn.functional.relu(self.lin1(x))
		x = self.head(x)
		return x
		# raise NotImplementedError
