import gym

environment = gym.make("CartPole-v0")
environment.reset()

for _ in range(1000):
    environment.render()
    environment.step(environment.action_space.sample())
