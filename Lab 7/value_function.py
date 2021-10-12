###
# Group Members
# Matthew: 1669326
# Mikayla: 1886648
# Michael-John: 1851234 
###

#######################################################################
# Following are some utilities for tile coding from Rich.
# To make each file self-contained, I copied them from
# http://incompleteideas.net/tiles/tiles3.py-remove
# with some naming convention changes
#
# Tile coding starts
from math import floor
import numpy as np
from gym.spaces import Discrete
import gym
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt
from gym import wrappers
class IHT:
    "Structure to handle collisions"

    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0: print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT): return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int): return hash(tuple(coordinates)) % m
    if m is None: return coordinates


def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    if ints is None:
        ints = []
    qfloats = [floor(f * num_tilings) for f in floats]
    tiles = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        tiles.append(hash_coords(coords, iht_or_size, read_only))
    return tiles


# Tile coding ends
#######################################################################

# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07


# wrapper class for state action value function
class ValueFunction:
    # In this example I use the tiling software instead of implementing standard tiling by myself
    # One important thing is that tiling is only a map from (state, action) to a series of indices
    # It doesn't matter whether the indices have meaning, only if this map satisfy some property
    # View the following webpage for more information
    # http://incompleteideas.net/sutton/tiles/tiles3.html
    # @max_size: the maximum # of indices
    def __init__(self, alpha, n_actions, num_of_tilings=8, max_size=2048):
        self.action_space = Discrete(n_actions)
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings

        # divide step size equally to each tiling
        self.step_size = alpha / num_of_tilings

        self.hash_table = IHT(max_size)

        # weight for each tile
        self.weights = np.zeros(max_size)

        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.num_of_tilings / (VELOCITY_MAX - VELOCITY_MIN)

    # get indices of active tiles for given state and action
    def _get_active_tiles(self, position, velocity, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        active_tiles = tiles(self.hash_table, self.num_of_tilings,
                             [self.position_scale * position, self.velocity_scale * velocity],
                             [action])
        return active_tiles

    # estimate the value of given state and action
    def __call__(self, state, action):
        position, velocity = tuple(state)
        if position == POSITION_MAX:
            return 0.0
        active_tiles = self._get_active_tiles(position, velocity, action)
        return np.sum(self.weights[active_tiles])

    # learn with given state, action and target
    def update(self, target, state, action):
        active_tiles = self._get_active_tiles(state[0], state[1], action)
        estimation = np.sum(self.weights[active_tiles])
        delta = self.step_size * (target - estimation)
        for active_tile in active_tiles:
            self.weights[active_tile] += delta

    def act(self, state, epsilon=0):
        if np.random.random() < epsilon:
            return self.action_space.sample()
        return np.argmax([self(state, action) for action in range(self.action_space.n)])



env= gym.make("MountainCar-v0")
alpha = 0.1
gamma = 1
goal = POSITION_MAX
iht = IHT(8)
numRuns = 1
maxEpisodes = 500
maxSteps = 200


# for a set of episodes
averageEpisode = np.zeros((maxEpisodes))
for run in range(numRuns):
    valFun = ValueFunction(alpha=alpha, n_actions=3, num_of_tilings=8)
    numSteps = []
    for episode in range(maxEpisodes):
        # get initial state and action of the episode
        state = env.reset()
        action = valFun.act(state, epsilon=0.1)
        print("Run {0}, Episode: {1}".format(run,episode))
        episodeFinished = False
        
        for step in range(maxSteps):
            if episode > 485:
                env.render()
            # if it is in the set of days that we would like to see - just for an example
            # of what the agent learnt, show it
            # if episode > 498:
            #     env.render()
            # take an action and observe
            obs = env.step(action)
            newState = obs[0]
            reward = obs[1]
            # if the position is now at/past the goal
            if newState[0] >= goal:
                # Update the Q-Values based on this new w - It seems to overwrite the wights based on what it learnt itself, so I am really uncertain here ***
                valFun.update(reward, state, action)
                # go to the next episode
                print("SUCCESS")
                break
            
            # get new action
            newAction = valFun.act(newState)

            # update w - not too certain how to deal with the grad in this case ****
            # valFun.update(goal, newState, newAction)
            # valFun.weights = valFun.weights + alpha * (reward + gamma*valFun.__call__(newState,newAction) - valFun.__call__(state,action)) * valFun.weights

            # Update the Q-Values based on this new w - It seems to overwrite the wights based on what it learnt itself, so I am really uncertain here
            # print(valFun.__call__(newState,newAction))
            valFun.update(reward + valFun.__call__(newState,newAction), state, action)
            state = newState
            action = newAction
        numSteps.append(step)
    if run == 0:
        averageEpisode = numSteps
    else:
        for val in range(len(averageEpisode)):
            averageEpisode[val] = averageEpisode[val] + numSteps[val]

for val in range(len(averageEpisode)):
    averageEpisode[val] = np.log(averageEpisode[val] / numRuns)
plt.plot(averageEpisode)
plt.title("Average Number of Steps Per Episode (max steps was 200)")
plt.ylabel("Num Steps (Logged)")
plt.xlabel("Episode")
plt.show()

