import numpy as np
from matplotlib import pyplot as plt
plt.rcdefaults()

# class of grid world
class GridWorld():

    def __init__ (self, rows, cols):
        # map to denote space
        self.map = np.zeros((rows, cols))
        self.rows = rows
        self.cols = cols
        # define the value map
        self.valueMap = np.zeros((rows,cols))
        # denote an obstacle with a sentinel value
        self.obstacle = -9

    def addObsticales(self, obstacles):
        # obstacles is an array of (row,col) co-ordinates
        # we can add to each point that this is an obstacle (denoted by massive negative sentinel value)
        for i in obstacles:
            self.map[i[0]][i[1]] = -1
            self.valueMap[i[0]][i[1]] = self.obstacle
    
    def addGoal(self, goal, goalVal):
        self.valueMap[goal[0]][goal[1]] = goalVal 

    def getEnvironmentMap(self):
        return self.map, self.rows, self.cols

    def printMap(self, mapToPrint):
        print("====== PRINTING " + mapToPrint + " ======")
        if mapToPrint == "map":
            for i in range(self.rows):
                temp = ""
                for j in self.map[i,:]:
                    temp += str(j) + " "
                print(temp)
            print("======================================")
            return
        if mapToPrint == "valueMap":
            for i in range(self.rows):
                temp = ""
                for j in self.valueMap[i,:]:
                    temp += str(j) + " "
                print(temp)
            print("======================================")
            return

class Agent:

    def __init__ (self, rows, cols, epsilon, goal):
        self.rewards = 0
        # bottom left
        self.stateX = rows-1
        self.stateY = 0
        self.rows = rows
        self.cols = cols
        # goal value
        self.obstacle = -1
        self.epsilon = epsilon
        self.goal = goal
        self.foundGoal = False
        self.gamma = 1
        self.trajectory = []

    def actionsSet(self, map):
        # so the rules are that we cannot move of the edge
        # after checking this, check to make sure that we are not moving into an obstacle
        actions = []
        # check if we can go up
        if self.stateX != 0 :
            if map[self.stateX-1][self.stateY] != self.obstacle == -1:
                actions.append("N")
        # check if we can go down
        if self.stateX !=self.rows - 1:
            if map[self.stateX + 1][self.stateY] != self.obstacle == -1:
                actions.append("S")
        # check if we can go left
        if self.stateY != 0 :
            if map[self.stateX][self.stateY - 1] != self.obstacle == -1:
                actions.append("W")
        # check if we can go right
        if self.stateY !=self.cols - 1:
            if map[self.stateX][self.stateY + 1] != self.obstacle == -1:
                actions.append("E")
        return actions

    def findOptimal(self, actions, valueMap):
        optimalAction = ""
        greatestReward = 0
        for action in np.arange(len(actions)):
            if action == 0:
                if actions[action] == "N":
                    greatestReward = valueMap[self.stateX - 1][self.stateY]
                    optimalAction = "N"
                elif actions[action] == "S":
                    greatestReward = valueMap[self.stateX + 1][self.stateY]
                    optimalAction = "S"
                elif actions[action] == "E":
                    greatestReward = valueMap[self.stateX][self.stateY + 1]
                    optimalAction = "E"
                elif actions[action] == "W":
                    greatestReward = valueMap[self.stateX][self.stateY - 1]
                    optimalAction = "W"
            else:
                thisReward = 0
                thisAction = ""
                if actions[action] == "N":
                    thisReward = valueMap[self.stateX - 1][self.stateY]
                    thisAction = "N"
                elif actions[action] == "S":
                    thisReward = valueMap[self.stateX + 1][self.stateY]
                    thisAction = "S"
                elif actions[action] == "E":
                    thisReward = valueMap[self.stateX][self.stateY + 1]
                    thisAction = "E"
                elif actions[action] == "W":
                    thisReward = valueMap[self.stateX][self.stateY - 1]
                    thisAction = "W"
                if greatestReward < thisReward:
                    optimalAction = thisAction
        if optimalAction == "":
            print("ERROR NEED A VALID ACTION")
        else:
            return optimalAction

    def epsilonGreedy(self, map, valueMap):
        actions = self.actionsSet(map)
        optimalAction = self.findOptimal(actions, valueMap)
        prob = np.random.uniform(low=0.0, high=1.0, size=1)[0]
        if prob < self.epsilon:
            # explore
            index = np.random.randint(0, high=len(actions), size=1)[0]
            return self.executeAction(actions[index])
        else:
            return self.executeAction(optimalAction)
    
    def executeAction(self, action):
        if action == "N":
            self.stateX -= 1
        elif action == "S":
            self.stateX += 1
        elif action == "E":
            self.stateY += 1
        elif action == "W":
            self.stateY -= 1
        else:
            print("ERROR NEED A VALID ACTION")
        self.rewards -= 1
        self.trajectory.append("(" + str(self.stateX) + ", " + str(self.stateY) + ") - " + action)
        # print(action)
        if self.stateX == self.goal[0] and self.stateY == self.goal[1]:
            self.rewards += 20
            self.foundGoal = True

    def printTrajectory(self):
        print("Destination | Direction Taken")
        for state in self.trajectory:
            print(state)


gridWorld = GridWorld(7,7)
gridWorld.printMap("valueMap")
obstacles = np.array(([[2,0],[2,1],[2,2],[2,3],[2,4],[2,5]]))
gridWorld.addObsticales(obstacles)
goal = np.array([0,0])
gridWorld.addGoal(goal, goalVal=20)
optimalValueFunction = np.array([[20, 19, 18, 17, 16, 15, 14], [19, 18, 17, 16, 15, 14, 13], [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, 12], [5, 6, 7, 8, 9, 10, 11], [4, 5, 6, 7, 8, 9, 10], [3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8]])
gridWorld.valueMap = optimalValueFunction
gridWorld.printMap("valueMap")
randomAgent = Agent(gridWorld.rows, gridWorld.cols, 2, goal)
greedyAgent = Agent(gridWorld.rows, gridWorld.cols, -1, goal)
averagedRandomAgentsReward = 0
averagedGreedyAgentsReward = 0

for i in np.arange(20):
    randomAgent = Agent(gridWorld.rows, gridWorld.cols, 2, goal)
    greedyAgent = Agent(gridWorld.rows, gridWorld.cols, -1, goal)
    for i in np.arange(50):
        randomAgent.epsilonGreedy(gridWorld.map, gridWorld.valueMap)
        # print(randomAgent.rewards)
    while greedyAgent.foundGoal == False:
        greedyAgent.epsilonGreedy(gridWorld.map, gridWorld.valueMap)
    
    averagedRandomAgentsReward += randomAgent.rewards
    averagedGreedyAgentsReward += greedyAgent.rewards

averagedRandomAgentsReward /= 20
averagedGreedyAgentsReward /= 20


# Plotting of agents' total reward average over 20 runs
objects = ('Random', 'Greedy')
y_pos = np.arange(len(objects))
performance = [averagedRandomAgentsReward, averagedGreedyAgentsReward]

plt.bar(y_pos, performance, align='center', alpha=0.5)
# Get the axes object
ax = plt.gca()
# remove the existing ticklabels
ax.set_xticklabels([])
# remove the extra tick on the negative bar
ax.set_xticks([idx for (idx, x) in enumerate(performance) if x > 0])
ax.spines["bottom"].set_position(("data", 0))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# placing each of the x-axis labels individually
label_offset = 0.5
for language, (x_position, y_position) in zip(objects, enumerate(performance)):
    if y_position > 0:
        label_y = -label_offset
    else:
        label_y = y_position - label_offset
    ax.text(x_position, label_y, language, ha="center", va="top")
# Placing the x-axis label, note the transformation into `Axes` co-ordinates
# previously data co-ordinates for the x ticklabels
ax.text(0.5, -0.05, "Agent's Total Reward", ha="center", va="top", transform=ax.transAxes)

plt.show()

# Agents' trajectories

print("====== Random Agent ======")
randomAgent.printTrajectory()
print("====== Greedy Agent ======")
greedyAgent.printTrajectory()

