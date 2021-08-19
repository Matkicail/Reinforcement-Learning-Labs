import numpy as np
from matplotlib import pyplot as plt

# STUDENT NUMBERS
# 1886648
# 1851234 
# 1669326

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
    
    # Setting the initial state value for the terminal state
    def addGoal(self, goal, goalVal):
        self.valueMap[goal[0]][goal[1]] = goalVal 

    # Returns the grid and it's dimensions
    def getEnvironmentMap(self):
        return self.map, self.rows, self.cols

    # Prints either the grid or the state values in the grid's format, based on the argument string provided
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
        # bottom left which is the starting state of the agent
        self.stateX = rows-1
        self.stateY = 0
        # Grid size agent works within
        self.rows = rows
        self.cols = cols
        # Obstacle values to account for when transitioning between states
        self.obstacle = -1
        self.epsilon = epsilon
        self.goal = goal
        self.foundGoal = False
        self.gamma = 1
        # Setup for the future display of the agent's trajectories
        self.trajectory = ["(" + str(self.stateX) + ", " + str(self.stateY) + ") \t    - Starting State"]
        self.trajectoryMap = np.zeros((rows, cols))

    # Creating the set of actions the agent can take.
    def actionsSet(self):
        actions = []
        
        actions.append("N")
        actions.append("S")
        actions.append("W")
        actions.append("E")
        
        return actions

    # Greedy policy implementation for the evaluation of the best action to execute. Takes into account bumping into obstacles and the boundry of the grid.
    def findOptimal(self, actions, valueMap, map):
        optimalAction = ""
        greatestReward = 0
        for action in np.arange(len(actions)):
            if action == 0:
                if actions[action] == "N":
                    if self.stateX != 0 and map[self.stateX-1][self.stateY] != self.obstacle:
                        greatestReward = valueMap[self.stateX - 1][self.stateY]
                    else:
                        greatestReward = valueMap[self.stateX][self.stateY]
                    optimalAction = "N"
                elif actions[action] == "S":
                    if self.stateX != self.rows - 1 and map[self.stateX + 1][self.stateY] != self.obstacle:
                        greatestReward = valueMap[self.stateX + 1][self.stateY]
                    else:
                        greatestReward = valueMap[self.stateX][self.stateY]
                    optimalAction = "S"
                elif actions[action] == "E":
                    if self.stateY != self.cols - 1 and map[self.stateX][self.stateY + 1] != self.obstacle:
                        greatestReward = valueMap[self.stateX][self.stateY + 1]
                    else:
                        greatestReward = valueMap[self.stateX][self.stateY]
                    optimalAction = "E"
                elif actions[action] == "W":
                    if self.stateY != 0 and map[self.stateX][self.stateY - 1] != self.obstacle:
                        greatestReward = valueMap[self.stateX][self.stateY - 1]
                    else:
                        greatestReward = valueMap[self.stateX][self.stateY]
                    optimalAction = "W"
            else:
                thisReward = 0
                thisAction = ""
                if actions[action] == "N":
                    if self.stateX != 0 and map[self.stateX-1][self.stateY] != self.obstacle:
                        thisReward = valueMap[self.stateX - 1][self.stateY]
                    else:
                        thisReward = valueMap[self.stateX][self.stateY]
                    thisAction = "N"
                elif actions[action] == "S":
                    if self.stateX != self.rows - 1 and map[self.stateX + 1][self.stateY] != self.obstacle:
                        thisReward = valueMap[self.stateX + 1][self.stateY]
                    else:
                        thisReward = valueMap[self.stateX][self.stateY]
                    thisAction = "S"
                elif actions[action] == "E":
                    if self.stateY != self.cols - 1 and map[self.stateX][self.stateY + 1] != self.obstacle:
                        thisReward = valueMap[self.stateX][self.stateY + 1]
                    else:
                        thisReward = valueMap[self.stateX][self.stateY]
                    thisAction = "E"
                elif actions[action] == "W":
                    if self.stateY != 0 and map[self.stateX][self.stateY - 1] != self.obstacle:
                        thisReward = valueMap[self.stateX][self.stateY - 1]
                    else:
                        thisReward = valueMap[self.stateX][self.stateY]
                    thisAction = "W"
                if greatestReward < thisReward:
                    optimalAction = thisAction
        if optimalAction == "":
            print("ERROR NEED A VALID ACTION")
        else:
            return optimalAction

    # Decision made whether to exploit or explore. Epsilon values < 0 produces the random agent and epsilon values > 1 produces the greedy agent
    def epsilonGreedy(self, map, valueMap):
        actions = self.actionsSet()
        optimalAction = self.findOptimal(actions, valueMap, map)
        prob = np.random.uniform(low=0.0, high=1.0, size=1)[0]
        if prob < self.epsilon:
            # explore
            index = np.random.randint(0, high=len(actions), size=1)[0]
            return self.executeAction(actions[index], map)
        else:
            # exploit
            return self.executeAction(optimalAction, map)
    
    # Performs the state transitions for the agent. Agent executes an action, but only transitions state if it can. The agent's trajectory history recording, total reward computation and evaluation if it entered the terminal state also takes place in the function.
    def executeAction(self, action, map):
        if action == "N":
            if self.stateX != 0 and map[self.stateX-1][self.stateY] != self.obstacle:
                self.stateX -= 1
        elif action == "S":
            if self.stateX != self.rows - 1 and map[self.stateX + 1][self.stateY] != self.obstacle:
                self.stateX += 1
        elif action == "E":
            if self.stateY != self.cols - 1 and map[self.stateX][self.stateY + 1] != self.obstacle:
                self.stateY += 1
        elif action == "W":
            if self.stateY != 0 and map[self.stateX][self.stateY - 1] != self.obstacle:
                self.stateY -= 1
        else:
            print("ERROR NEED A VALID ACTION")
        self.rewards -= 1
        self.trajectory.append("(" + str(self.stateX) + ", " + str(self.stateY) + ") \t    -       " + action)
        self.trajectoryMap[self.stateX][self.stateY] -= 1
        if self.stateX == self.goal[0] and self.stateY == self.goal[1]:
            self.rewards += 20
            self.foundGoal = True

    # Prints to the console the trajectory the agent took for the episode.
    def printTrajectory(self):
        print("Destination | Direction Taken")
        for state in self.trajectory:
            print(state)

# The creation of the grid world alongside the agents.
gridWorld = GridWorld(7,7)
obstacles = np.array(([[2,0],[2,1],[2,2],[2,3],[2,4],[2,5]]))
gridWorld.addObsticales(obstacles)
goal = np.array([0,0])
gridWorld.addGoal(goal, goalVal=20)
# By hand optimal value function values
optimalValueFunction = np.array([[20, 19, 18, 17, 16, 15, 14], [19, 18, 17, 16, 15, 14, 13], [np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, np.NINF, 12], [5, 6, 7, 8, 9, 10, 11], [4, 5, 6, 7, 8, 9, 10], [3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8]])
gridWorld.valueMap = optimalValueFunction
randomAgent = Agent(gridWorld.rows, gridWorld.cols, 2, goal)
greedyAgent = Agent(gridWorld.rows, gridWorld.cols, -1, goal)
averagedRandomAgentsReward = 0
averagedGreedyAgentsReward = 0

# 20 episodes evaluated for each type of agent with new agents each episode. The cumulative reward across episodes in computed as well.  
for i in np.arange(20):
    randomAgent = Agent(gridWorld.rows, gridWorld.cols, 2, goal)
    greedyAgent = Agent(gridWorld.rows, gridWorld.cols, -1, goal)
    # Random agent executes 50 actions.
    for i in np.arange(50):
        randomAgent.epsilonGreedy(gridWorld.map, gridWorld.valueMap)
    # Greedy agent executes actions until the terminal state is reached.
    while greedyAgent.foundGoal == False:
        greedyAgent.epsilonGreedy(gridWorld.map, gridWorld.valueMap)
    
    averagedRandomAgentsReward += randomAgent.rewards
    averagedGreedyAgentsReward += greedyAgent.rewards

# Cumulative rewards are averaged for 20 episodes.
averagedRandomAgentsReward /= 20
averagedGreedyAgentsReward /= 20


# Plotting of agents' total reward average over 20 episodes. This bar graph plot code was taken from Stack Overflow and modified to plot the lab's data. The link to the code is in the readMe.
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

# Agents' trajectories printed to the console with the action they took to transition to the state. This is more detailed than the heat maps below.
print("====== Random Agent ======")
randomAgent.printTrajectory()
print("====== Greedy Agent ======")
greedyAgent.printTrajectory()

# Agents' trajectories|path taken represented using a heat map plot.
trajectoryFig = plt.figure()
trajectoryGrid = trajectoryFig.add_gridspec(1, 2)
trajectoryPlot = trajectoryGrid.subplots()

# Greatest number of transitions to a state from both agents. Used for normalization of the heat maps' values to better represent each agent's trajectory|path. 
largestValue = np.amax(np.array([np.amax(np.abs(randomAgent.trajectoryMap)), np.amax(np.abs(greedyAgent.trajectoryMap))]))

trajectoryFig.suptitle("Agent Trajectories")
trajectoryPlot[0].set_title("Random")
trajectoryPlot[0].imshow(randomAgent.trajectoryMap, vmin = -largestValue, vmax=0, alpha=0.8, cmap='magma')
trajectoryPlot[1].set_title("Greedy")
trajectoryPlot[1].imshow(greedyAgent.trajectoryMap, vmin = -largestValue, vmax=0, alpha=0.8, cmap='magma')

plt.show()