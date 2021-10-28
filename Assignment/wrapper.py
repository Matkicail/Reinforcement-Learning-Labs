import gym
import minihack
from minihack import reward_manager
import numpy as np
from minihack import RewardManager
from gym import spaces
from nle import nethack
from numpy.lib.function_base import select

class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self):
        state = self.env.reset()
        state = self.selectObs(state)
        return state

    def selectObs(self, obs, desired=["chars","message","inv_letters"]):
        tempState = np.array(())
        for desire in desired:
            temp = obs[desire]
            temp = np.array(temp)
            temp = temp.astype(int)
            temp = temp.flatten()
            tempState = np.append(tempState, temp)
        return tempState

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.selectObs(next_state)
        return next_state, reward, done, info

def createActionSpace():
    moves = tuple(nethack.CompassDirection)
    navActions = moves + (
        nethack.Command.APPLY,
        nethack.Command.AUTOPICKUP,
        nethack.Command.CAST,
        nethack.Command.CLOSE,
        nethack.Command.DROP,
        nethack.Command.EAT,
        nethack.Command.ESC,
        nethack.Command.FIRE,
        nethack.Command.FIGHT,
        nethack.Command.INVOKE,
        nethack.Command.KICK,
        nethack.Command.LOOK, 
        nethack.Command.LOOT,
        nethack.Command.OPEN,
        nethack.Command.PRAY,
        nethack.Command.PUTON,
        nethack.Command.QUAFF,
        nethack.Command.READ,
        nethack.Command.REMOVE,
        nethack.Command.RIDE,
        nethack.Command.RUB,
        nethack.Command.SEARCH,
        nethack.Command.TAKEOFF,
        nethack.Command.TAKEOFFALL,
        # nethack.Command.THROW,
        nethack.Command.TIP,
        nethack.Command.WEAR,
        nethack.Command.WIELD,
        nethack.Command.ZAP,
    )
    return navActions

def customGym(maxLength=2000, seed=10):
    reward_gen = RewardManager()
    reward_gen.add_eat_event("apple", reward=1, repeatable=False)
    # reward_gen.add_wield_event("dagger", reward=2)
    reward_gen.add_location_event("sink", reward=-1, terminal_required=False)
    reward_gen.add_kill_event("minotaur",reward=2, repeatable=False) #minotaur guards the exit and is in the maze, which requires the WoD to do
    
    # may need more rewards as we continue
    env = gym.make(
        "MiniHack-Quest-Hard-v0",
        observation_keys=("chars", "inv_letters", "message"),
            reward_manager = reward_gen,
            actions=createActionSpace()
    )
    print("Environment created")
    env._max_episode_steps = maxLength
    env.seed(seed)
    return env

#  Me checking stuff out again to see it all works and it seems to
# env = BasicWrapper(customGym())
# print("Action_Space Size: {0}".format(env.action_space.n))
# env.reset()
# move = 0
# while move != -1:
#     env.render()
#     move = int(input())
#     env.step(move)
# print("over")