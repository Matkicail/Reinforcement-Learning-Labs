from wrapper import BasicWrapper, customGym
import gym
import minihack
from minihack import reward_manager
import numpy as np
from minihack import RewardManager
from collections import defaultdict
import random
import csv

dummyEnv = BasicWrapper(customGym())
actionSpaceSize = dummyEnv.action_space.n
actionsTaken = []
class State():
    def __init__(self, state, reward=None,done=False,info=None):
        # print('init state')
        self.state_s = state
        self.reward = reward
        self.done = done
        self.info = info
        if done:
            self.legal_actions = []
        else:
            self.legal_actions = list(np.arange(actionSpaceSize))

class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):

        self.state = state #observation,reward,done,info,actions
        self.parent = parent #parent node in tree
        self.parent_action = parent_action # action parent took to get here
        self.children = [] 
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[0] = 0
        self._results[1] = 0
        self._results[-0.01] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
		
        return

    def untried_actions(self):
        # print('untried_actions' , self.state)
        self._untried_actions = self.state.legal_actions
        return self._untried_actions

    def q(self):
        loses = self._results[0]
        wins = self._results[1]
        # steps = self._results[-0.01]
        # print('los' , loses)
        # print('win' , wins)
        # print('st' , steps)
        return wins - loses

    # def qs(self):
    #     loses = self._results[0]
    #     wins = self._results[1]
    #     steps = self._results[-0.01]
    #     print('los' , loses)
    #     print('win' , wins)
    #     print('st' , steps)
    #     return wins - steps - loses

    def n(self):
        return self._number_of_visits

    def expand(self, env):
        action = self._untried_actions.pop()

        obs, reward, done, info = env.step(action)
        # self.state.reward += reward

        next_state = State(obs,reward,done,info)
        child_node = MonteCarloTreeSearchNode(state = next_state, parent=self, parent_action=action)

        self.children.append(child_node)
        return child_node , env

    def is_terminal_node(self):
        # print('is_terminal_node')
        return self.state.done

    def rollout(self , env):
        current_rollout_state = self.state
        
        while not current_rollout_state.done:
            
            possible_moves = current_rollout_state.legal_actions
            
            action = self.rollout_policy(possible_moves)
            obs, reward, done, info = env.step(action)
            # self.state.reward += reward
            current_rollout_state = State(obs,reward,done,info)

        return current_rollout_state.reward

    def backpropagate(self, result):
        self._number_of_visits += 1.
        # print(self._results)
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        # for c in self.children:
        #     print('q',c.q())
        #     print('n',c.n())
        #     print('sn',np.log(self.n()))
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        child = np.random.choice(np.flatnonzero(choices_weights == max(choices_weights)))
        return self.children[child]

    # def best_childf(self, c_param=0.1):
    #     for c in self.children:
    #         print(c)
    #         print('q',c.qs())
    #         print('n',c.n())
    #         print('sn',np.log(self.n()))
    #     choices_weights = [(c.qs() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
    #     child = np.random.choice(np.flatnonzero(choices_weights == max(choices_weights)))
    #     return self.children[child]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self,env):
        current_node = self
        while not current_node.is_terminal_node():

            if len(current_node.children)==0:
                return current_node.expand(env)
            elif random.uniform(0,1)<.5:
                current_node = current_node.best_child()
            else:
                if not current_node.is_fully_expanded():	
                    return current_node.expand(env)
                else:
                    current_node = current_node.best_child()
            # if not current_node.is_fully_expanded():
            #     return current_node.expand(env)
            # else:
            #     current_node = current_node.best_child()
        return current_node , env

    def best_action(self,seeds,actions, simulation_no=1000):
        print('------------------ children' , self.children)

        
        for i in range(simulation_no):
            print('\t\tSimulation {0}'.format(i), end="\r")
            choices_weights = [(c.q() / c.n()) + 0.1 * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
            # print('be cw ', choices_weights)
            # # print(np.argmax(choices_weights))
            # # print(self.children[np.argmax(choices_weights)])
            # print('------------------ children' , self.children)
            # print(self.children[np.argmax(choices_weights)].parent_action)


            env2 = BasicWrapper(customGym())
            initial_state2 = env2.reset()

            for act in actions:
                obs, r, done, info = env2.step(act)

            # env2.render()

            v = self._tree_policy(env2)
            reward = v[0].rollout(v[1])

            p = v[0].backpropagate(reward)
            
        choices_weights = [(c.q() / c.n()) + 0.1 * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        print(choices_weights)
        print(np.argmax(choices_weights))
        # print(self.children[np.argmax(choices_weights)])
        # print(self.children)
        print(self.children[np.argmax(choices_weights)].parent_action)
        
        
        return self.best_child(c_param=0.1)

	


def main():

    csv_save = open('minihack_rewards.csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(csv_save)

	# creates random seed

    # make_seeds = lambda: (int(random.random()*1e19), int(random.random()*1e19), False)
    # seed = make_seeds()
    for mpx in range(100):
        seed = (907, 101, False)
        prev_actions = []
        rewards = []
        actionsTaken = []
        # creates env
        env = BasicWrapper(customGym())
        # sets seed for env
        rewardArr = []
        initial_state = env.reset()
        initial_state = State(state = initial_state)
        env.render()

        # print(env.action_space.n)
        # print(env.action_space)

        # gets seed from env
        seeds = env.get_seeds()

        root = MonteCarloTreeSearchNode(state = initial_state)
        selected_node = root.best_action(seeds,prev_actions)
        print('results' , selected_node._results)

        print('action : ' , selected_node.parent_action)
        prev_actions.append(selected_node.parent_action)
        

        next_state, reward, done, info = env.step(selected_node.parent_action)
        rewards.append(reward)
        next_state = State(next_state,reward,done,info)


        env.render()
        cnt = 0
        while not done:
            cnt += 1
            print("Current Game {0}, current step {1}".format(mpx, cnt))
            selected_node = MonteCarloTreeSearchNode(state = next_state)
            selected_node = selected_node.best_action(seeds,prev_actions)
            print('results' , selected_node._results)

            
            print('action : '  , selected_node.parent_action)
            prev_actions.append(selected_node.parent_action)

            next_state, reward, done, info = env.step(selected_node.parent_action)
            rewards.append(reward)

            next_state = State(next_state,reward,done,info)

            selected_node = MonteCarloTreeSearchNode(state = next_state)
            # print(env.action_space.n)
            # print(env._actions)
            # print('**********************its children' , selected_node.children)
            env.render()
        rewardArr = np.array(rewards)
        np.savetxt("rewardArr-{0}".format(mpx), rewardArr)
        try:
            writer.writerow(rewards) 
        except:
            pass

main()
