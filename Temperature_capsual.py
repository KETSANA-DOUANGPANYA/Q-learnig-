import gym
import gym_toytext
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


class Agent():
    def __init__(self):
        self.q_table = np.zeros((8, 5))
        self.learning_rate = 0.05
        self.discount_factor = 0.95
        self.epsilon = 0.5
        self.decay_factor = 0.999
        self.reward_for_each_episode = []

    def play(self, env, number_of_episode=50):
        for i_episode in range(number_of_episode):
            print("Episode {} of {}".format(i_episode+1, number_of_episode))
            state = env.reset()
            self.epsilon *= self.decay_factor
            total_reward = 0

            end_game = False
            while not end_game:
                if self.__qTableIsEmpty(state) or self.__probability(self.epsilon):
                    action = self.__getActionByRandomly(env)
                else:
                    action = self.__getActionWithHighestedReward(state)
                new_state, reward, end_game, _ = env.step(action)
                # update q table
                self.q_table[state, action] += self.learning_rate * (
                    reward + self.discount_factor * self.__getExpectedReward(new_state) - self.q_table[state, action])
                total_reward += reward
                state = new_state
            self.reward_for_each_episode.append(total_reward)
            print(tabulate(self.q_table, showindex="always", headers=[
                  "State", "Action0 (OpenPum 20s)", "Action1 (OpenPum 30s)", "Action2 (OpenPum 40s)", "Action3 (OpenPum 50s)", "Action4 (OpenPum 60s)"]))

    def __qTableIsEmpty(self, state):
        return np.sum(self.q_table[state, :]) == 0

    def __probability(self, probability):
        return np.random.random() < probability

    def __getActionByRandomly(self, env):
        return env.action_space.sample()

    def __getActionWithHighestedReward(self, state):
        return np.argmax(self.q_table[state, :])

    def __getExpectedReward(self, state):
        return np.max(self.q_table[state, :])

env = gym.make('Capsule-v0')

agent = Agent()

agent.play(env)

plt.plot(agent.reward_for_each_episode)

plt.title('Reward timeming')

plt.ylabel('Total reward')
plt.xlabel('Episode')

plt.show()