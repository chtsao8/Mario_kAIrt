import retro
import random
import numpy as np
from wrappers import wrapper
import pickle

with open('q_tables_v4/q_table_13000.pickle', 'rb') as handle:
    q_table = pickle.load(handle)

actions_list = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]

actions_size = len(actions_list)

env = retro.make(game='SuperMarioKart-Snes')
env = wrapper(env)
start_state = np.array_str(env.reset())

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.5

rewards = []
for i in range(1, 1000):
    state = np.array_str(env.reset())
    total_reward = 0
    done = False
    iterations = 0

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions_list)
            action_index = actions_list.index(action)
        else:
            action_index = np.argmax(q_table[state])
            action = actions_list[action_index]

        next_state, reward, done, _ = env.step(action) 

        total_reward += reward

        state = np.array_str(next_state)

        iterations += 1
        env.render()

    rewards.append(total_reward / iterations)
    
    print("Episode: " + str(i), "epsilon = " + str(epsilon), "Mean Reward = " + str(np.mean(rewards[-100:])))


print("Training finished.\n")