import gym
import minihack
from minihack import reward_manager
import numpy as np
from minihack import RewardManager
from gym import spaces
from nle import nethack
from numpy.lib.function_base import select

class BasicWrapper(gym.Wrapper):
    def __init__(self, env, seed=0, maxSteps = 10000):
        super().__init__(env)
        self.env = env
        self.seedCustom = seed
        self.maxSteps = maxSteps
        self.augmentedReward = 0
        self.seenStates = None
        self.currStep = 0
        self.repeatNum = 0
        # self.oldest = None
        # self.secOldest = None
        # self.secNewest= None
        # self.newest = None
        # print("NOTE: in here the max steps are set in the construction - change this as desired but keep in mind many steps are needed since a lot of exploring is needed")
        # print("NOTE: Changing the seed if done, must be done here in initialising the class")
        # print("IFF this is done, he seed used in the custom gym will be different and hence this change must be accounted for there as well")
        # print("The default seed will be set to 0 in all cases as a standard seed - which allows for us to compare results")
        # print("hence changing the seed is not advisable \n environment has been initialised")
    def fullReset(self):
        self.seenStates = None
        self.currStep = 0
        self.repeatNum = 0


    def reset(self):
        self.env.seed(self.seedCustom)
        state = self.env.reset()
        state = self.selectObs(state)
        return state

    def lookThroughDict(self, temp):
        solidStone = [73,116,39,115,32,115,111,108,105,100,32,115,116,111,110,101,46]
                    #   73 116 39 115 32 115 111 108 105 100 32 115 116 111 110 101 46 
        whatStrange = [87,104 ,97, 116, 32, 97, 32, 115, 116, 114, 97, 110, 103, 101, 32, 100, 105, 114, 101, 99, 116 ,105 ,111 ,110 ,33 ,32 ,32 ,78 ,101 ,118  ] # ,101 ,114 ,32 ,109 ,105 ,110 ,100 ,46
        nothingZap = [89, 111, 117, 32 ,100 ,111 ,110 ,39 ,116, 32, 104, 97, 118, 101, 32, 97 ,110, 121, 116, 104, 105, 110, 103, 32] #  ,116, 111, 32, 122, 97, 112, 46 
        whatDir = [73, 110, 32 ,119 ,104 ,97 ,116 ,32 ,100 ,105 ,114 ,101 ,99] # ,116 ,105 ,111 ,110 ,63 
        noDoor = [89, 111, 117, 32, 115, 101, 101, 32 ,110 ,111 ,32 ,100 ,111 ,111 ,114 ,32 ,116] # ,101 ,114 ,101 ,46  at the end missing here
        cnt = 0 
        punishmentReward = 0.2
        isMessage = False
        solid = True
        for char in solidStone:
            if temp[cnt] != char:
                solid = False
                break
                
                # break
            cnt += 1
        if solid:
            self.augmentedReward -= punishmentReward
            return
        strange = True
        for char in whatStrange:
            if temp[cnt] != char:
                strange = False
                # self.augmentedReward -= 0.3
                # return
                break
            cnt += 1
        if strange:
            self.augmentedReward -= punishmentReward
            return

        zap = True
        for char in nothingZap:
            if temp[cnt] != char:
                strange = False
                # self.augmentedReward -= 0.3
                # return
                break
            cnt += 1
        if zap:
            self.augmentedReward -= punishmentReward
            return
        whatDirec = True
        for char in whatDir:
            if temp[cnt] != char:
                whatDirec = False
                # self.augmentedReward -= 0.3
                # return
                break
            cnt += 1
        if whatDirec:
            self.augmentedReward -= punishmentReward
            return
        noDoorHere = True
        for char in noDoor:
            if temp[cnt] != char:
                noDoorHere = False
                # self.augmentedReward -= 0.3
                # return
                break
            cnt += 1
        if noDoorHere:
            self.augmentedReward -= punishmentReward
            return
        else:
            self.augmentedReward += 0.01
            return

    def rewardExploration(self, temp):
        seenBefore = False
        
        # for state in self.seenStates:
        #     if np.all(state == temp):
        #         self.repeatNum += 1
        #         self.augmentedReward -= 0.0025 * self.repeatNum
        #         # seenBefore = True
        #         return
        # self.augmentedReward += 5
        
        

    def selectObs(self, obs, desired=["chars","message","inv_letters"]):
        tempState = np.array(())
        for desire in desired:
            temp = obs[desire]
            if desire == "message":
                self.lookThroughDict(temp)
            temp = np.array(temp)
            temp = temp.astype(int)
            temp = temp.flatten()
            tempState = np.append(tempState, temp)
        return tempState

    def step(self, action, maxLength=10000):
        self.currStep += 1
        self.augmentedReward = 0
        next_state, reward, done, info = self.env.step(action)
        # if np.any(self.seenStates != None):
        #     self.rewardExploration(next_state["chars"])
        #     self.seenStates.append(next_state["chars"])
        next_state = self.selectObs(next_state)
        self.lookThroughDict(next_state)
        # if np.any(self.seenStates != None):
        #     self.rewardExploration(next_state)
        #     self.seenStates.append(next_state)
        # else:
        #     self.seenStates = [next_state]
        reward += self.augmentedReward
        # if done == True:
        #     if reward != 2:
        #         reward = -0.0105*(self.maxSteps-self.currStep) - 0.0105*(self.maxSteps) 
        #     elif reward == 2: 
        #         reward = 50
        # elif reward != -0.01 and reward != 20 and reward != 40:
        #     reward -= 0.02
        # if reward == 20:
        #     print("it equipped the wand")
        # if reward == 40:
        #     print("it killed the minotaur")
        # if np.any(self.seenStates == None):
        #     self.seenStates = [[next_state]]
        # else:
        # self.augmentedReward = 0
        # self.oldest = None
        # self.secOldest = None
        # self.secNewest= None
        # self.newest = None
        # if np.all(self.newest == None):
        #     self.newest = next_state
        # elif np.all(self.newest != None):
        #     if np.all(self.secNewest == None):
        #         self.secNewest = self.newest
        #         self.newest = next_state
        #     else:
        #         if np.all(self.secOldest == None):
        #             self.secOldest = self.secNewest
        #             self.secNewest = self.newest
        #             self.newest = next_state
        #         else:
        #             if np.all(next_state == self.secOldest):
        #                 reward -= 0.02
        #             if np.all(next_state == self.secNewest):
        #                 reward -= 0.02
        #             if np.all(next_state == self.newest):
        #                 reward -= 0.02
        return next_state, reward, done, info

def createActionSpace():
    moves = tuple(nethack.CompassDirection)
    navActions = moves + (
        # nethack.Command.APPLY,
        # nethack.Command.AUTOPICKUP,
        # nethack.Command.CAST,
        # nethack.Command.CLOSE,
        # nethack.Command.DROP,
        # nethack.Command.EAT,
        # nethack.Command.ESC,
        # nethack.Command.FIRE,
        # nethack.Command.FIGHT,
        # nethack.Command.INVOKE,
        # nethack.Command.KICK,
        # nethack.Command.LOOK, 
        # nethack.Command.LOOT,
        # nethack.Command.OPEN,
        # nethack.Command.PRAY,
        # nethack.Command.PUTON,
        # nethack.Command.QUAFF,
        # nethack.Command.READ,
        # nethack.Command.REMOVE,
        # nethack.Command.RIDE,
        # nethack.Command.RUB,
        # nethack.Command.SEARCH,
        # nethack.Command.TAKEOFF,
        # nethack.Command.TAKEOFFALL,
        # nethack.Command.THROW,
        # nethack.Command.TIP,
        # nethack.Command.WEAR,
        # nethack.Command.WIELD,
        # nethack.Command.ZAP,
    )
    return navActions

def customGym(maxLength=10000, seed=0):
    reward_gen = RewardManager()
    reward_gen.add_eat_event("apple", reward=1, repeatable=False)
    reward_gen.add_wield_event("wand", reward=20, repeatable=False) # changed to convince the agent finding the wand and using it is good
    reward_gen.add_location_event("sink", reward=-1, terminal_required=False)
    reward_gen.add_kill_event("minotaur",reward=40, repeatable=False) #minotaur guards the exit and is in the maze, which requires the WoD to do
    
    # may need more rewards as we continue
    env = gym.make(
        "MiniHack-Quest-Hard-v0",
        observation_keys=("chars", "inv_letters", "message"),
            reward_manager = reward_gen,
            actions=createActionSpace()
    )
    # print("Environment created")
    # print("maxLength here represents the maxSteps possible for the agent, please ensure that the number given here is the same\n as the number given to the BasicWrapper otherwise there is a disconnect,\n by default their values if left unchanged and default then they are the same.")
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
#     try:
#         move = int(input("Please give me an input - you useless \n"))
#         if move >= env.action_space.n:
#             move = 0
#     except:
#         print("don't do that")
#         move = 0
#     env.step(move)
# print("over")