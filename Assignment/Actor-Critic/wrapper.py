import gym
import minihack
from minihack import reward_manager
import numpy as np
from minihack import RewardManager
from gym import spaces
from nle import nethack
from numpy.lib.function_base import select

class BasicWrapper(gym.Wrapper):
    # wrapper for assignment - this has been augmented for certain algorithms and has undergone multiple iterations
    # this wrapper in specific is associated with Actor-Critic

    def __init__(self, env, maxSteps, seed=0):
        super().__init__(env)
        self.env = env
        self.seedCustom = seed
        self.maxSteps = maxSteps
        self.augmentedReward = 0
        self.seenStates = None
        self.currStep = 0
        self.prevState = None
        self.firstDoor = False
        self.secondDoor = False
        self.openedFirstDoor = False
        self.seenRing = False
        self.seenRod = False
        self.seenLava = False
        self.nextToLava = False
        self.afterToLava = False
        self.beforeMazeRewards = []
        self.afterMazeRewards = []
        self.afterLavaRewards = []
        print("NOTE: in here the max steps are set in the construction - change this as desired but keep in mind many steps are needed since a lot of exploring is needed")
        print("NOTE: Changing the seed if done, must be done here in initialising the class")
        print("IFF this is done, he seed used in the custom gym will be different and hence this change must be accounted for there as well")
        print("The default seed will be set to 0 in all cases as a standard seed - which allows for us to compare results")
        print("hence changing the seed is not advisable \n environment has been initialised")
    
    def fullReset(self): # function to reset the environment at the end of an episode
        self.seenStates = None
        self.prevState = None
        self.currStep = 0
        self.repeatNum = 0
        self.countState = []
        self.firstDoor = False
        self.openedFirstDoor = False
        self.secondDoor = False
        self.seenRing = False
        self.seenRod = False
        self.seenLava = False
        self.nextToLava = False
        self.afterToLava = False
        self.beforeMazeRewards = []
        self.afterMazeRewards = []
        self.afterLavaRewards = []

    def reset(self):
        self.env.seed(self.seedCustom)
        self.fullReset()
        state = self.env.reset()
        state = self.selectObs(state)
        
        return state

    def lookThroughDict(self, temp):
        # a set of dictionary messsage for each possible message that was found to be important.
        # the names of these messages are related to the message the agent would have seen should it take an action to warrant this message
        solidStone = [73,116,39,115,32,115,111,108,105,100,32,115,116,111,110,101,46]
        whatStrange = [87,104 ,97, 116, 32, 97, 32, 115, 116, 114, 97, 110, 103, 101, 32, 100, 105, 114, 101, 99, 116 ,105 ,111 ,110 ,33 ,32 ,32 ,78 ,101 ,118  ] 
        nothingZap = [89, 111, 117, 32 ,100 ,111 ,110 ,39 ,116, 32, 104, 97, 118, 101, 32, 97 ,110, 121, 116, 104, 105, 110, 103, 32] 
        whatDir = [73, 110, 32 ,119 ,104 ,97 ,116 ,32 ,100 ,105 ,114 ,101 ,99] 
        noDoor = [89, 111, 117, 32, 115, 101, 101, 32 ,110 ,111 ,32 ,100 ,111 ,111 ,114 ,32 ,116] 
        greatRing = [102, 32, 45, 32, 97, 110, 32, 101, 109, 101, 114, 97, 108, 100, 32 ,114, 105, 110, 103, 46 ] # great
        uranWand = [103, 32, 45 ,32 ,97 ,32 ,117 ,114 ,97 ,110 ,105 ,117, 109, 32 ,119 ,97 ,110 ,100 ,46] # great
        pickUpOn = [65, 117, 116, 111, 112, 105, 99 ,107 ,117 ,112, 58 ,32 ,79 ,78 ,44 ,32 ,102, 111, 114 ,32 ,97 ,108, 108, 32 ,111 ,98 ,106, 101, 99 ,116 ,115 ,46 ,3, 2 ]
        pickUpOff = [65 ,117, 116, 111, 112, 105, 99 ,107 ,117 ,112, 58 ,32 ,79 ,70 ,70 ,46 ]
        wentThrough = [84 ,104 ,101 ,32 ,100 ,111 ,111 ,114, 32, 111, 112, 101, 110, 115, 46 ]
        whatInvoke = [87 ,104, 97 ,116 ,32 ,100 ,111 ,32 ,121, 111 ,117, 32 ,119 ,97 ,110 ,116 ,32 ,116, 111, 32 ,105 ,110, 118, 111, 107, 101, 63]
        nothingToWear = [89, 111, 117, 32 ,100 ,111, 110, 39 ,116 ,32 ,104, 97 ,118 ,101 ,32 ,97 ,110, 121, 116, 104, 105, 110, 103, 32 ,101 ,108 ,115 ,101, 32 ,116 ,111 ,32 ,112 ]
        cnt = 0 
        punishmentReward = 0.005 # a punishment reward which is multiplied depending on how bad the message was

        solid = True
        for char in nothingToWear:
            if temp[cnt] != char:
                solid = False
                break
            cnt += 1
        if solid:
            self.augmentedReward -= punishmentReward*50
            return
        cnt = 0
        solid = True
        for char in whatInvoke:
            if temp[cnt] != char:
                solid = False
                break
            cnt += 1
        if solid:
            self.augmentedReward -= punishmentReward*50
            return
        cnt = 0 
        solid = True
        for char in solidStone:
            if temp[cnt] != char:
                solid = False
                break
            cnt += 1
        if solid:
            self.augmentedReward -= punishmentReward*6
            return
        strange = True
        cnt = 0
        for char in whatStrange:
            if temp[cnt] != char:
                strange = False
                break
            cnt += 1
        if strange:
            self.augmentedReward -= punishmentReward * 5
            return

        zap = True
        cnt = 0
        for char in nothingZap:
            if temp[cnt] != char:
                strange = False
                break
            cnt += 1
        if zap:
            self.augmentedReward -= punishmentReward * 9
            return
        whatDirec = True
        cnt = 0
        for char in whatDir:
            if temp[cnt] != char:
                whatDirec = False
                break
            cnt += 1
        if whatDirec:
            self.augmentedReward -= punishmentReward
            return
        noDoorHere = True
        cnt = 0
        for char in noDoor:
            if temp[cnt] != char:
                noDoorHere = False
                break
            cnt += 1
        if noDoorHere:
            self.augmentedReward -= punishmentReward
            return

        turnedOffPickUp = True
        cnt = 0
        for char in pickUpOff:
            if temp[cnt] != char:
                turnedOffPickUp = False
                break
            cnt += 1
        
        if turnedOffPickUp:
            self.augmentedReward -= punishmentReward*8
            return

        foundRing = True
        cnt = 0
        for char in greatRing:
            if temp[cnt] != char:
                foundRing = False
                break
            cnt += 1
        if foundRing:
            self.augmentedReward += 2
            return

        found = True
        cnt = 0
        for char in uranWand:
            if temp[cnt] != char:
                uranWand = False
                break
            cnt += 1

        wentThroughDoor = True
        cnt = 0
        for char in wentThrough:
            if temp[cnt] != char:
                wentThroughDoor = False
                break
            cnt += 1
        
        if wentThroughDoor:
            self.augmentedReward += 2
            self.openedFirstDoor = True
            return

        turnedonPickUp = True
        cnt = 0
        for char in pickUpOn:
            if temp[cnt] != char:
                turnedonPickUp = False
                break
            cnt += 1
        
        if turnedonPickUp:
            self.augmentedReward += 2
            return


        else:
            self.augmentedReward += 0.5
            return

    def rewardExploration(self, temp):
        # this is a function that rewards the agent for taking actions that bring it to exceedingly good areas 
        # i.e places where good rewards can be found
        bestPlaces = [43, 47, 61, 126]
        goodAction = 20
        for best in bestPlaces:
            if best in temp:
                if best == 43 and self.firstDoor == False:
                    self.firstDoor == True
                    self.augmentedReward += goodAction
                elif best == 47 and self.seenRing == False:
                    self.seenRing == True
                    self.augmentedReward += goodAction
                elif best == 61 and self.seenRod == False:
                    self.seenRod = True
                    self.augmentedReward += goodAction
                elif best == 43 and self.seenRod == self.firstDoor == self.seenRing:
                    self.secondDoor = True
                    self.augmentedReward += goodAction
                elif best == 126 and self.seenLava == False:
                    self.seenLava = True
                    self.augmentedReward += goodAction
                elif best == 126 and self.nextToLava == False:
                    tempArr = np.array(temp)
                    temArr = tempArr.flatten()
                    indexAgent = np.where(temArr == 64)
                    if temArr[indexAgent+1] == 126:
                        self.augmentedReward += goodAction
                        self.nextToLava = True
                elif best == 126 and self.afterToLava == False:
                    tempArr = np.array(temp)
                    temArr = tempArr.flatten()
                    indexAgent = np.where(temArr == 64)
                    if temArr[indexAgent-1] == 126:
                        self.augmentedReward += goodAction
                        self.afterToLava = True

            else:
                tempArr = np.array(temp)
                tempArr[np.where(tempArr == 64)] = 35
                if np.any(self.prevState != None):
                    if np.any(tempArr != self.prevState):
                        self.augmentedReward += 0.06
                else:
                    self.augmentedReward -= 0.15
                self.prevState = tempArr


                

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

        if reward == -0.01 and done == False:
            self.augmentedReward += 0.02
        elif reward == 2 and done == True:
            reward += 100
        self.lookThroughDict(next_state["message"])
        next_state = self.selectObs(next_state)
        if np.any(self.seenStates != None):
            self.rewardExploration(next_state)
            self.seenStates = next_state
        else:
            self.seenStates = next_state
        reward += self.augmentedReward
        if self.openedFirstDoor == False:
            self.beforeMazeRewards.append(reward)

        elif self.openedFirstDoor == True and self.afterToLava == False:
            self.afterMazeRewards.append(reward)

        elif self.afterToLava == True:
            self.afterMazeRewards.append(reward)
        if done == True or self.currStep + 1 == self.maxSteps:

            if self.openedFirstDoor == False:
                arr = np.array(self.beforeMazeRewards)
                try:
                    with open("beforeMazeOpenedRewards-{0}-sum.txt".format(self.maxSteps), "a") as file_object:

                        val = np.sum(arr)
                        file_object.write(str(val) + "\n")

                except:
                    print("Failed to write")

            if self.openedFirstDoor:
                arr = np.array(self.afterMazeRewards)
                
                try:

                    with open("afterMazeOpenedRewards-{0}-sum.txt".format(self.maxSteps), "a") as file_object:

                        val = np.sum(arr)
                        file_object.write(str(val) + "\n")

                except:
                    print("Failed to write")

        
            if self.afterToLava:
                arr = np.array(self.afterLavaRewards)
                try:

                    with open("beforeMazeOpenedRewards-{0}-sum.txt".format(self.maxSteps), "a") as file_object:

                        val = np.sum(arr)
                        file_object.write(str(val) + "\n")

                except:
                    print("Failed to write")

        return next_state, reward, done, info

def createActionSpace():
    # set of moves that we allow the agent to take
    moves = tuple(nethack.CompassDirection) 
    navActions = moves + (
        # nethack.Command.APPLY,
        nethack.Command.AUTOPICKUP,
        # nethack.Command.CAST,
        # nethack.Command.CLOSE,
        # nethack.Command.DROP,
        # nethack.Command.EAT,
        # nethack.Command.ESC,
        # nethack.Command.FIRE,
        # nethack.Command.FIGHT,
        nethack.Command.INVOKE,
        # nethack.Command.KICK,
        # nethack.Command.LOOK, 
        # nethack.Command.LOOT,
        # nethack.Command.OPEN,
        # nethack.Command.PRAY,
        nethack.Command.PUTON,
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
        nethack.Command.ZAP,
    )
    return navActions

def customGym(maxLength=10000, seed=0):
    reward_gen = RewardManager()
    reward_gen.add_eat_event("apple", reward=1, repeatable=False)
    reward_gen.add_wield_event("wand", reward=20, repeatable=False) # changed to convince the agent finding the wand and using it is good
    reward_gen.add_location_event("sink", reward=-1, terminal_required=False)
    reward_gen.add_kill_event("minotaur",reward=100, repeatable=False) # minotaur guards the exit and is in the maze, which requires the WoD to do
    
    # may need more rewards as we continue
    env = gym.make(
        "MiniHack-Quest-Hard-v0",
        observation_keys=("chars", "inv_letters", "message"),
            reward_manager = reward_gen,
            actions=createActionSpace(),
            savedir = "./games/"
    )
    print("Environment created")
    print("maxLength here represents the maxSteps possible for the agent, please ensure that the number given here is the same\n as the number given to the BasicWrapper otherwise there is a disconnect,\n by default their values if left unchanged and default then they are the same.")
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
#         move = input("Please give me an input - you useless \n")
#         if move != "g" or move != "f":
#             move = int(move)
#         env.step(move)
#     except:
#         print("don't do that")
#         move = 0
# print("over")