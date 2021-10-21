import gym
import minihack
from minihack import reward_manager
import numpy as np
from minihack import RewardManager


# how to interact with the "FakeWrapper"
# create it and only access reset, render and step
# the action space and the observation space should work as they stand
class FakeWrapper:

    def __init__(self):
        self.env = customGym() # set up of environment
        self.n_actions = self.env.action_space # a way to see the size of the action space
        self.o_space = len(self.reset()) # a way to get the number of states you will see (assuming messages are a constant size)

    def render(self):
        return self.env.render()

    def selectObs(self, obs, desired=["chars","message","inv_letters"]):
        tempState = np.array(())
        for desire in desired:
            temp = obs[desire]
            temp = np.array(temp)
            temp = temp.astype(int)
            temp = temp.flatten()
            tempState = np.append(tempState, temp)
        return tempState

    def reset(self):
        obs = self.env.reset()
        obs = self.selectObs(obs)
        return obs

    def step(self, action):
        observation_, reward, done, info = self.env.step(action)
        observation_ = self.selectObs(observation_)
        return observation_, reward, done, info

def customGym():
    reward_gen = RewardManager()
    reward_gen.add_eat_event("apple", reward=1, repeatable=False)
    # reward_gen.add_wield_event("dagger", reward=2)
    reward_gen.add_location_event("sink", reward=-1, terminal_required=False)
    reward_gen.add_kill_event("minotaur",reward=2, repeatable=False) #minotaur guards the exit and is in the maze, which requires the WoD to do
    # may need more rewards as we continue
    env = gym.make(
        "Quest-Hard-v0,
        observation_keys=("chars", "inv_letters", "message"),
            reward_manager = reward_gen
    )
    print("Environment created")
    return env

temp = FakeWrapper()
temp.reset()
temp.step(9)
