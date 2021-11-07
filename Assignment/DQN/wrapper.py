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
    def fullReset(self):
        self.seenStates = None
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
        greatRing = [102, 32, 45, 32, 97, 110, 32, 101, 109, 101, 114, 97, 108, 100, 32 ,114, 105, 110, 103, 46 ] # great
        uranWand = [103, 32, 45 ,32 ,97 ,32 ,117 ,114 ,97 ,110 ,105 ,117, 109, 32 ,119 ,97 ,110 ,100 ,46] # great
        pickUpOn = [65, 117, 116, 111, 112, 105, 99 ,107 ,117 ,112, 58 ,32 ,79 ,78 ,44 ,32 ,102, 111, 114 ,32 ,97 ,108, 108, 32 ,111 ,98 ,106, 101, 99 ,116 ,115 ,46 ,3, 2 ]
        pickUpOff = [65 ,117, 116, 111, 112, 105, 99 ,107 ,117 ,112, 58 ,32 ,79 ,70 ,70 ,46 ]
        wentThrough = [84 ,104 ,101 ,32 ,100 ,111 ,111 ,114, 32, 111, 112, 101, 110, 115, 46 ]

        cnt = 0 
        punishmentReward = 0.0005
        isMessage = False
        solid = True
        for char in solidStone:
            if temp[cnt] != char:
                solid = False
                break
            cnt += 1
        if solid:
            self.augmentedReward -= punishmentReward
            return
        strange = True
        cnt = 0
        for char in whatStrange:
            if temp[cnt] != char:
                strange = False
                break
            cnt += 1
        if strange:
            self.augmentedReward -= punishmentReward
            return

        zap = True
        cnt = 0
        for char in nothingZap:
            if temp[cnt] != char:
                strange = False
                break
            cnt += 1
        if zap:
            self.augmentedReward -= punishmentReward
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
            self.augmentedReward -= punishmentReward*3
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
            # self.step()
            return

        found = True
        cnt = 0
        for char in uranWand:
            if temp[cnt] != char:
                uranWand = False
                break
            cnt += 1

        # wentThrough
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
            self.augmentedReward += 0.0001
            return

    def rewardExploration(self, temp):
        bestPlaces = [43, 47, 61, 126]
        for best in bestPlaces:
            if best in temp:
                if best == 43 and self.firstDoor == False:
                    self.firstDoor == True
                    self.augmentedReward += 5
                elif best == 47 and self.seenRing == False:
                    self.seenRing == True
                    self.augmentedReward += 5
                elif best == 61 and self.seenRod == False:
                    self.seenRod = True
                    self.augmentedReward += 5
                elif best == 43 and self.seenRod == self.firstDoor == self.seenRing:
                    self.secondDoor = True
                    self.augmentedReward += 5
                elif best == 126 and self.seenLava == False:
                    self.seenLava = True
                    self.augmentedReward += 5
                elif best == 126 and self.nextToLava == False:
                    tempArr = np.array(temp)
                    temArr = tempArr.flatten()
                    indexAgent = np.where(temArr == 64)
                    if temArr[indexAgent+1] == 126:
                        self.augmentedReward += 5
                        self.nextToLava = True
                elif best == 126 and self.afterToLava == False:
                    tempArr = np.array(temp)
                    temArr = tempArr.flatten()
                    indexAgent = np.where(temArr == 64)
                    if temArr[indexAgent-1] == 126:
                        self.augmentedReward += 5
                        self.afterToLava = True
                

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

        if reward == -0.01:
            self.augmentedReward += 0.0001

        # msg = input("tell me what to do")
        # if  msg == "s":
        #     img = np.array(charImage)
        #     self.newest.append(list([img.flatten()]))
        # elif msg == "write":
        #     arr = np.array(self.newest)
        #     arr = np.squeeze(arr)
        #     np.savetxt("coolPlaces.txt", arr)

        self.lookThroughDict(next_state["message"])
        next_state = self.selectObs(next_state)
        if np.any(self.seenStates != None):
            self.rewardExploration(next_state)
            # self.seenStates.append(next_state)
            self.seenStates = next_state
        else:
            # self.seenStates = [next_state]
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
                    currFile = np.loadtxt("beforeMazeOpenedRewards-{0}.txt".format(self.maxSteps))
                    currFile = np.append(currFile, arr)
                except:
                    np.savetxt("beforeMazeOpenedRewards-{0}.txt".format(self.maxSteps), arr)

            if self.openedFirstDoor:
                arr = np.array(self.afterMazeRewards)
                try:
                    currFile = np.loadtxt("afterMazeOpenedRewards-{0}.txt".format(self.maxSteps))
                    currFile = np.append(currFile, arr)
                except:
                    np.savetxt("afterMazeOpenedRewards-{0}.txt".format(self.maxSteps), arr)

        
            if self.afterToLava:
                arr = np.array(self.afterLavaRewards)
                try:
                    currFile = np.loadtxt("afterLavaRewards-{0}.txt".format(self.maxSteps))
                    currFile = np.append(currFile, arr)
                except:
                    np.savetxt("afterLavaRewards-{0}.txt".format(self.maxSteps), arr)

        return next_state, reward, done, info

def createActionSpace():
    moves = tuple(nethack.CompassDirection)
    navActions = moves + (
        # nethack.Command.APPLY,
        nethack.Command.AUTOPICKUP,
        nethack.Command.CAST,
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
    reward_gen.add_kill_event("minotaur",reward=40, repeatable=False) #minotaur guards the exit and is in the maze, which requires the WoD to do
    
    # may need more rewards as we continue
    env = gym.make(
        "MiniHack-Quest-Hard-v0",
        observation_keys=("chars", "inv_letters", "message"),
            reward_manager = reward_gen,
            actions=createActionSpace()
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