import retro
import random
import numpy as np
import pickle
from wrappers import wrapper

actions_list = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]

actions_size = len(actions_list)

env = retro.make(game='SuperMarioKart-Snes')
env = wrapper(env)
start_state = np.array_str(env.reset())
initial_vec = np.zeros(actions_size)

q_table = {start_state: initial_vec}

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 1
eps_decay = 0.99999975
eps_min = 0.1

rewards = []
for i in range(1, 50000):
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
        epsilon *= eps_decay
        epsilon = max(eps_min, epsilon)

        next_state, reward, done, info = env.step(action) 
        next_state = np.array_str(next_state)

        total_reward += reward

        old_value = q_table[state][action_index]

        if next_state not in q_table.keys():
            q_table[next_state] = initial_vec
            #print('triggered ' + str(iterations))

        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state][action_index] = new_value

        state = next_state

        iterations += 1
        #env.render()

    rewards.append(total_reward / iterations)

    print(i)
    if i % 100 == 0:
        print("Episode: " + str(i), "epsilon = " + str(epsilon), "Mean Reward = " + str(np.mean(rewards[-100:])))
        with open('q_tables_v4/q_table_' + str(i) + '.pickle' , 'wb') as handle:
            pickle.dump(q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('rewards_v4/rewards_' + str(i) + '.pickle', 'wb') as f:
            pickle.dump(rewards, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('q_tables_v4/q_table_FINAL.pickle', 'wb') as handle:
        pickle.dump(q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('rewards_v4/rewards_FINAL.pickle', 'wb') as f:
            pickle.dump(rewards, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Training finished.\n")