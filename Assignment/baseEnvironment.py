import gym
import minihack
env = gym.make("MiniHack-Quest-Hard-v0")
env.reset() # each reset generates a new environment instance
env.step(1)  # move agent '@' north
env.render()