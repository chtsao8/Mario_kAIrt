import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('rewards_v4/rewards_13100.pickle', 'rb') as handle:
    rewards = pickle.load(handle)

summation = 0
mean_rewards = []
for i, value in enumerate(rewards):
    summation += value
    if i%100 == 0:
        mean = summation/100
        mean_rewards.append(mean)
        summation = 0

mean_rewards = mean_rewards[1:-1]

x = range(0,len(mean_rewards))

plot = plt.plot(x,mean_rewards)
plt.ylabel('Avg Reward, Every 100 Episodes')
plt.xlabel('Episodes / 100')
plt.title('Average Reward vs Episodes')
plt.show(plot)
