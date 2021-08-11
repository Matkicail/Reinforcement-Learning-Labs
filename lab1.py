import numpy as np
from matplotlib import pyplot as plt

# STUDENT NUMBERS
# 1886648
# 1851234 
# 1669326

class Arm:
    def __init__(self, armMean, armVar):
        # the mean given to this arm by the bandit - this is the true mean of the arm - which we need to learn
        self.armMean = armMean
        # the variance this arm has as determined by the bandit
        self.armVar = armVar
        # number of times this arm was pulled
        self.numPulls = 0
        # reward history of this arm
        self.rewardHist = 0
        # old estimate 
        self.currEstimate = 0
        # storing the alpha updated estimate
        self.alphaEstimate = 0
    
    def updateEstimate(self, reward):
        self.currEstimate = self.currEstimate + (1/self.numPulls)*(reward - self.currEstimate)

    def updateAlphaEstimate(self, reward):
        self.alphaEstimate += self.alphaEstimate + alpha *(reward - self.alphaEstimate)

    def pullArm(self):
        self.numPulls += 1
        reward = np.random.normal(loc=self.armMean, scale=self.armVar, size=1)[0]
        self.rewardHist += reward
        self.updateEstimate(reward)
        self.updateAlphaEstimate(reward)
        return reward

    def printArm(self):
        print("Mean: " + str(self.armMean) + ", Var: " + str(self.armVar))

class Bandit:

    def __init__(self, numArms, armVar, banditMean, banditVar):
        # How many arms does this bandit have 
        self.numArms = numArms
        # This bandits variance for an arm around it's mean
        self.armVar = armVar
        # the mean around which the bandit pulls arm means
        self.banditMean = banditMean
        # the variance around which the bandit creates a gaussian
        self.banditVar = banditVar
        # the arms form the action space
        self.arms = []
        # optimal arm
        self.optimalArm = None
        self.initArms()
        self.optimisticInit()

    def initArms(self):
        for i in range(self.numArms):
            arm = Arm(np.random.normal(loc=self.banditMean, scale=self.banditVar, size = 1), self.armVar)
            self.arms.append(arm)

    def printBanditArms(self):
        for i in range(self.numArms):
            print("--------------------------------")
            print("Arm number " + str(i))
            self.arms[i].printArm()
            print("--------------------------------")

    def findOptimal(self):
        if len(self.arms) > 0 :
            rewards = np.array(())
            for i in range(self.numArms):
                # find the best arm - the one that we thing is best at current
                rewards = np.append(rewards, self.arms[i].currEstimate)
            self.optimalArm = np.argmax(rewards)

    def findOptimalAlpha(self):
        if len(self.arms) > 0 :
            rewards = np.array(())
            for i in range(self.numArms):
                # find the best arm - the one that we thing is best at current
                rewards = np.append(rewards, self.arms[i].alphaEstimate)
            self.optimalArm = np.argmax(rewards)

    def epsilonGreedy(self):
        self.findOptimal()
        prob = np.random.uniform(low=0.0, high=1.0, size=1)[0]
        if prob < epsilon:
            # explore
            index = np.random.randint(0, high=self.numArms, size=1)[0]
            return self.arms[index].pullArm()
        else:
            return self.arms[self.optimalArm].pullArm()

    def optimisticInit(self):
        # find the maximal mean
        # initialise current guess of arms to be maximal mean * 2 ( just to ensure above )
        if len(self.arms) > 0:
            for i in range(len(self.arms)):
                self.arms[i].alphaEstimate = 5
    
    def optimistic(self):
        self.findOptimal()
        return self.epsilonGreedy()

    def UCB(self, time):
        # we know the action space and their current estimates based on an epsilon greedy - i.e find optimal arm
        temp = np.array(())
        for i in range(self.numArms):
            numPull = self.arms[i].numPulls
            if numPull == 0:
                numPull = 1
            temp = np.append(self.arms[i].currEstimate + c * (np.log(time+1)/numPull), temp)
        
        return self.arms[np.argmax(temp)].pullArm()

# Driver code
# pay attention to the names of the bandits to figure out who's parameters are relevant and who's are not (just dummy values so the program does not crash)
steps = 1000
epsilon = 0.1
alpha = 0.2

banditVeryGreedy = Bandit(10, 1.0, 0, 3.0)
banditVeryGreedyRewards = np.array(())
banditVeryGreedyRewards = np.append(banditVeryGreedyRewards, 0)
temp = np.array(())
for i in range (steps):
    temp = np.append(temp, banditVeryGreedy.epsilonGreedy())
    if i % 100 == 0 and i != 0:
        inx = np.arange(100) + (i-100)
        banditVeryGreedyRewards = np.append(banditVeryGreedyRewards, np.mean(temp[inx]))

epsilon = 0.005
banditMedGreedy = Bandit(10, 1.0, 0, 3.0)
banditMedGreedyRewards = np.array(())
banditMedGreedyRewards = np.append(banditMedGreedyRewards, 0)
temp = np.array(())
for i in range (steps):
    temp = np.append(temp, banditMedGreedy.epsilonGreedy())
    if i % 100 == 0 and i != 0:
        inx = np.arange(100) + (i-100)
        banditMedGreedyRewards = np.append(banditMedGreedyRewards, np.mean(temp[inx]))

epsilon = 0.001
banditlowGreedy = Bandit(10, 1.0, 0, 3.0)
banditLowGreedyRewards = np.array(())
banditLowGreedyRewards = np.append(banditLowGreedyRewards, 0)
temp = np.array(())
for i in range (steps):
    temp = np.append(temp, banditlowGreedy.epsilonGreedy())
    if i % 100 == 0 and i != 0:
        inx = np.arange(100) + (i-100)
        banditLowGreedyRewards = np.append(banditLowGreedyRewards, np.mean(temp[inx]))

# making the epsilon value zero so purely based on initialisation
# ignore the alpha value - that was just when testing something else
# in the running it will run it but via epsilon greedy but it'll first check out the optimisitic stuff
alpha = 0.1
banditHighAlpha = Bandit(10, 1.0, 0, 3.0)
banditHighAlphaRewards = np.array(())
banditHighAlphaRewards = np.append(banditHighAlphaRewards, 0)
temp = np.array(())
for i in range (steps):
    temp = np.append(temp, banditHighAlpha.optimistic())
    if i % 100 == 0 and i != 0:
        inx = np.arange(100) + (i-100)
        banditHighAlphaRewards = np.append(banditHighAlphaRewards, np.mean(temp[inx]))

c = 2 
banditUCB = Bandit(10, 1.0, 0, 3.0)
banditUCBRewards = np.array(())
banditUCBRewards = np.append(banditUCBRewards, 0)
temp = np.array(())
for i in range (steps):
    temp = np.append(temp, banditUCB.UCB(i))
    if i % 100 == 0 and i != 0:
        inx = np.arange(100) + (i-100)
        banditUCBRewards = np.append(banditUCBRewards, np.mean(temp[inx]))

# change the values of epsilon and alpha as needed if you change them above
plt.title("Epsilon Greedy Agents")
plt.xlabel("Pull Number (x * 100 Runs)")
plt.ylabel("Average Reward")
plt.plot(banditVeryGreedyRewards, label="epsilon = 0.1")
plt.plot(banditMedGreedyRewards, label="epsilon = 0.05")
plt.plot(banditLowGreedyRewards, label="epsilon = 0.001")
plt.legend()
plt.show()

#print(banditHighAlphaRewards)

plt.title("Optimistic (Greedy with Optimistic Init) Agents")
plt.xlabel("Pull Number (x * 100 Runs)")
plt.ylabel("Average Reward")
plt.plot(banditHighAlphaRewards, label="Q5 = 5")
plt.legend()
plt.show()

plt.title("UCB Agents")
plt.xlabel("Pull Number (x * 100 Runs)")
plt.ylabel("Average Reward")
plt.plot(banditUCBRewards, label="C = 2")
plt.legend()
plt.show()

plt.title("Agents")
plt.xlabel("Pull Number (x * 100 Runs)")
plt.ylabel("Average Reward")
plt.plot(banditVeryGreedyRewards, label="Greedy: epsilon = 0.1")
plt.plot(banditHighAlphaRewards, label="Optimistic: Q5 = 5")
plt.plot(banditUCBRewards, label="UCB: C = 2")
plt.legend()
plt.show()
plt.savefig("ComparisonPlots.jpg")


#ignore this for now
#plt.title("Agents")
#plt.xlabel("Steps")
#plt.ylabel("Average Reward")
#plt.plot(banditVeryGreedyRewards, label="Greedy: epsilon = 0.1")
#plt.plot(banditHighAlphaRewards, label="Optimistic: Q5 = 5")
#plt.plot(banditUCBRewards, label="UCB: C = 2")
#plt.legend()