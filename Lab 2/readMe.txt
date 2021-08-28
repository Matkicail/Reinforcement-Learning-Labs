
# STUDENT NUMBERS
# 1886648
# 1851234 
# 1669326

When run, the bar graph for the comparison between the random and greedy agent (Using the optimal value function) of the total reward averaged for 20 runs is shown first.
Then upon closing the graph, the bar graph for the comparison between the random and greedy agent (Using the optimal value function) of the standard deviation over 20 runs is shown.
There is no variation in the standard deviation given that the random agent will only earn negative values of 1 except in the extremely rare case it manages to get to the goal by random.
The agent that follows an optimal path always has the same value - since it aims to get to the goal and it follows a predetermined path. Thus, this agent will have no variance and hence no standard deviation.
Then upon closing the graph, the heat maps graph showing a sample run of the trajectories of the agents is shown.
When this graph is closed, greater detail of the agents' trajectories is printed to the console.

The link to the code used for the bar graphs: https://stackoverflow.com/questions/60519582/matplotlib-bar-chart-negative-values-below-x-axis
