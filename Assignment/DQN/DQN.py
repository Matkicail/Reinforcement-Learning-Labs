# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
from collections import OrderedDict, deque, namedtuple, defaultdict
from typing import List, Tuple, Iterator

import gym
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import DistributedType
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

import wrapper

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())


# %%
class DQN(nn.Module):
	"""Simple MLP network."""

	def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 16):
		"""
		Args:
			obs_size: observation/state size of the environment
			n_actions: number of discrete actions available in the environment
			hidden_size: size of hidden layers
		"""
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, n_actions),
		)

	def forward(self, x):
		return self.net(x.float())

class LSTMDQN(nn.Module):
	"""Simple MLP network."""

	def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 16, hidden_layers: int = 3, stateless: bool = False):
		"""
		Args:
			obs_size: observation/state size of the environment
			n_actions: number of discrete actions available in the environment
			hidden_size: size of hidden layers
		"""
		super().__init__()
		self.stateless = stateless

		self.lstm_layer = nn.LSTM(obs_size, hidden_size, hidden_layers)
		self.net = nn.Sequential(
			nn.ReLU(),
			nn.Linear(hidden_size, n_actions),
		)

	def forward(self, x):
		if len(x.size()) == 1: x = x.view(1, -1)
		x = x.view(1, x.size(0), x.size(1)) if self.stateless else x.view(x.size(0), 1, x.size(1))
		h, _ = self.lstm_layer(x.float())
		h = h.view(h.size(0), h.size(2))
		return self.net(h)


# %%

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
	"Experience",
	field_names=["state", "action", "reward", "done", "new_state"],
)


# %%
class ReplayBuffer:
	"""Replay Buffer for storing past experiences allowing the agent to learn from them.

	Args:
		capacity: size of the buffer
	"""

	def __init__(self, capacity: int) -> None:
		self.buffer = deque(maxlen=capacity)

	def __len__(self) -> None:
		return len(self.buffer)

	def append(self, experience: Experience) -> None:
		"""Add experience to the buffer.

		Args:
			experience: tuple (state, action, reward, done, new_state)
		"""
		self.buffer.append(experience)

	# Sampler for random sampled batches (Linear Network)
	def sample(self, batch_size: int) -> Tuple:
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)
		states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

		return (
			np.array(states),
			np.array(actions, dtype=np.int64),
			np.array(rewards, dtype=np.float32),
			np.array(dones, dtype=np.bool),
			np.array(next_states),
		)

	# Sampler for stateless LSTMS
	# def sample(self, batch_size: int) -> Tuple:
	# 	indices = np.random.choice(len(self.buffer) - batch_size, 1, replace=False)
	# 	states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in np.arange(indices, indices + batch_size)))
	# 	return (
	# 		np.array(states),
	# 		np.array(actions, dtype=np.int64),
	# 		np.array(rewards, dtype=np.float32),
	# 		np.array(dones, dtype=np.bool),
	# 		np.array(next_states),
	# 	)


# %%
# Pytorch Lightning setup for iteratble datasets and applied to the replay buffer.
class RLDataset(IterableDataset):
	"""Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

	Args:
		buffer: replay buffer
		sample_size: number of experiences to sample at a time
	"""

	def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
		self.buffer = buffer
		self.sample_size = sample_size

	def __iter__(self) -> Iterator:
		states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
		for i in range(len(dones)):
			yield states[i], actions[i], rewards[i], dones[i], new_states[i]

# %%
# Agent that engages with the environment.
class Agent:
	"""Base Agent class handeling the interaction with the environment."""

	def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
		"""
		Args:
			env: training environment
			replay_buffer: replay buffer storing experiences
		"""
		self.env = env
		self.replay_buffer = replay_buffer
		self.reset()
		self.state = self.env.reset()

	def reset(self) -> None:
		"""Resents the environment and updates the state."""
		self.state = self.env.reset()

	def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
		"""Using the given network, decide what action to carry out using an epsilon-greedy policy.

		Args:
			net: DQN network
			epsilon: value to determine likelihood of taking a random action
			device: current device

		Returns:
			action
		"""
		if np.random.random() < epsilon:
			action = self.env.action_space.sample()
		else:
			state = torch.tensor([self.state])

			if device not in ["cpu"]:
				state = state.cuda(device)

			q_values = net(state)
			_, action = torch.max(q_values, dim=1)
			action = int(action.item())

		return action

	@torch.no_grad()
	def play_step(
		self,
		net: nn.Module,
		epsilon: float = 0.0,
		device: str = "cpu",
	) -> Tuple[float, bool]:
		"""Carries out a single interaction step between the agent and the environment.

		Args:
			net: DQN network
			epsilon: value to determine likelihood of taking a random action
			device: current device

		Returns:
			reward, done
		"""

		action = self.get_action(net, epsilon, device)

		# Do step in the environment.
		new_state, reward, done, _ = self.env.step(action)

		exp = Experience(self.state, action, reward, done, new_state)

		self.replay_buffer.append(exp)
		self.state = new_state
		if done:
			self.reset()
		return reward, done, new_state


# %%
# Pytorch Lightning module to log information about the agent.
class DQNCallback(Callback):
	
	def __init__(self):
		self.average_training_total_reward = 0.0
		self.training_current_epoch = 0
		self.training_current_global_step = 0

	def merge_list_of_dictionaries(self, dict_list):
		new_dict = {}
		for d in dict_list:
			for d_key in d:
				if d_key not in new_dict:
					new_dict[d_key] = []
				new_dict[d_key].append(d[d_key])
		return new_dict

	def update_running_total_reward_average(self, total_reward):
		self.average_training_total_reward = self.average_training_total_reward + (total_reward - self.average_training_total_reward) / (1 + self.training_current_epoch)

	def on_train_epoch_end(self, trainer, pl_module):

		outputs = pl_module.on_train_epoch_end_outputs

		losses, logs = self.merge_list_of_dictionaries(outputs).values()

		train_losses, total_rewards, rewards, steps, states = self.merge_list_of_dictionaries(logs).values()

		losses = torch.stack(losses)

		train_losses = torch.stack(train_losses)
		total_rewards = torch.stack(total_rewards)
		rewards = torch.stack(rewards)
		steps = torch.stack(steps)

		average_loss = train_losses.mean()
		total_reward = total_rewards.max()

		self.update_running_total_reward_average(total_reward.item())

		for index in np.arange(steps.size(dim=0)):
			pl_module.logger.experiment.add_scalar("Loss/Train", train_losses[index], self.training_current_global_step)
			pl_module.logger.experiment.add_scalar("Reward/Train", rewards[index], self.training_current_global_step)
			self.training_current_global_step += 1

		pl_module.logger.experiment.add_scalar("Average_Loss/Train", average_loss, self.training_current_epoch)
		pl_module.logger.experiment.add_scalar("Total_Reward/Train", total_reward, self.training_current_epoch)
		pl_module.logger.experiment.add_scalar("Average_Total_Reward/Train", self.average_training_total_reward, self.training_current_epoch)

		self.training_current_epoch += 1


# %%
class DQNLightning(LightningModule):
	"""Basic DQN Model."""

	def __init__(
		self,
		batch_size: int = 16,
		lr: float = 1e-2,
		env: str = "CartPole-v0",
		gamma: float = 0.99,
		sync_rate: int = 10,
		replay_size: int = 1000,
		warm_start_size: int = 1000,
		eps_last_frame: int = 1000,
		eps_start: float = 1.0,
		eps_end: float = 0.01,
		episode_length: int = 200,
		warm_start_steps: int = 1000,
		model: str = "ANN",
	) -> None:
		"""
		Args:
			batch_size: size of the batches")
			lr: learning rate
			env: gym environment tag
			gamma: discount factor
			sync_rate: how many frames do we update the target network
			replay_size: capacity of the replay buffer
			warm_start_size: how many samples do we use to fill our buffer at the start of training
			eps_last_frame: what frame should epsilon stop decaying
			eps_start: starting value of epsilon
			eps_end: final value of epsilon
			episode_length: max length of an episode
			warm_start_steps: max episode reward in the environment
		"""
		super().__init__()
		self.save_hyperparameters()

		self.env = wrapper.BasicWrapper(wrapper.customGym(maxLength=self.hparams.episode_length))
		obs_size = len(self.env.reset())
		n_actions = self.env.action_space.n

		if self.hparams.model in ["ANN"]:
			self.net = DQN(obs_size, n_actions)
			self.target_net = DQN(obs_size, n_actions)
		elif self.hparams.model in ["LSTM"]:
			self.net = LSTMDQN(obs_size, n_actions)
			self.target_net = LSTMDQN(obs_size, n_actions)

		self.buffer = ReplayBuffer(self.hparams.replay_size)
		self.agent = Agent(self.env, self.buffer)
		self.total_reward = 0
		self.episode_reward = 0
		self.populate(self.hparams.warm_start_steps)

	def populate(self, steps: int = 1000) -> None:
		"""Carries out several random steps through the environment to initially fill up the replay buffer with
		experiences.

		Args:
			steps: number of random steps to populate the buffer with
		"""
		for i in range(steps):
			self.agent.play_step(self.net, epsilon=1.0)

	def forward(self, x: Tensor) -> Tensor:
		"""Passes in a state x through the network and gets the q_values of each action as an output.

		Args:
			x: environment state

		Returns:
			q values
		"""
		output = self.net(x)
		return output

	def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
		"""Calculates the mse loss using a mini batch from the replay buffer.

		Args:
			batch: current mini batch of replay data

		Returns:
			loss
		"""
		states, actions, rewards, dones, next_states = batch

		state_action_values = self.net(states)
		state_action_values = state_action_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

		with torch.no_grad():
			next_state_values = self.target_net(next_states).max(1)[0]
			next_state_values[dones] = 0.0
			next_state_values = next_state_values.detach()

		expected_state_action_values = next_state_values * self.hparams.gamma + rewards

		return nn.MSELoss()(state_action_values, expected_state_action_values)

	def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
		"""Carries out a single step through the environment to update the replay buffer. Then calculates loss
		based on the minibatch recieved.

		Args:
			batch: current mini batch of replay data
			nb_batch: batch number

		Returns:
			Training loss and log metrics
		"""
		device = self.get_device(batch)
		epsilon = max(
			self.hparams.eps_end,
			self.hparams.eps_start - (self.global_step + 1) / self.hparams.eps_last_frame,
		)

		# Step through environment with agent
		reward, done, state = self.agent.play_step(self.net, epsilon, device)
		self.episode_reward += reward
	
		# Calculates training loss
		loss = self.dqn_mse_loss(batch)

		if done:
			self.total_reward = self.episode_reward
			self.episode_reward = 0

		# Soft update of target network
		if self.global_step % self.hparams.sync_rate == 0:
			self.target_net.load_state_dict(self.net.state_dict())

		logs = {
			"train_loss": loss.detach(),
			"total_reward": torch.tensor(self.total_reward).to(device),
			"reward": torch.tensor(reward).to(device),
			"step": torch.tensor(self.global_step).to(device),
			"state": torch.tensor(state).to(device)
		}

		return OrderedDict({"loss": loss, "log": logs,})

	def training_epoch_end(self, outputs) -> None:
		self.on_train_epoch_end_outputs = outputs

	def configure_optimizers(self) -> List[Optimizer]:
		"""Initialize Adam optimizer."""
		optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
		return [optimizer]

	def __dataloader(self) -> DataLoader:
		"""Initialize the Replay Buffer dataset used for retrieving experiences."""
		dataset = RLDataset(self.buffer, self.hparams.episode_length)
		dataloader = DataLoader(
			dataset=dataset,
			batch_size=self.hparams.batch_size,
			sampler=None,
		)
		return dataloader

	def train_dataloader(self) -> DataLoader:
		"""Get train loader."""
		return self.__dataloader()

	def get_device(self, batch) -> str:
		"""Retrieve device currently being used by minibatch."""
		return batch[0].device.index if self.on_gpu else "cpu"


# %%
model = DQNLightning(model = "ANN", episode_length=10000, warm_start_size=20000, warm_start_steps=20000, replay_size=20000, eps_last_frame=1000000000, batch_size=800)
logger = TensorBoardLogger("tb_logs", name="DQN_Minihack")

trainer = Trainer(
	gpus=AVAIL_GPUS,
	max_epochs=200001,
	logger=logger,
	callbacks=[DQNCallback()],
)

trainer.fit(model)