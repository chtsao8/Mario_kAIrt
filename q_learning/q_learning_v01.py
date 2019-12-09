import retro
import random
import numpy as np
import pickle

actions_list = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
                # [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]

actions_size = len(actions_list)

def to_grayscale(img):
    return np.mean(img, axis=0).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return np.array_str(to_grayscale(to_grayscale(img)))

env = retro.make(game='SuperMarioKart-Snes')
start_state = env.reset()
preprocessed_start_state = preprocess(start_state)
initial_vec = np.zeros(actions_size)

q_table = {preprocessed_start_state: initial_vec}

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 1
eps_decay = 0.99999970
eps_min = 0.1

for i in range(1, 100000):
    env.reset()
    state = preprocessed_start_state
    reward = 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            # action = env.action_space.sample() # Explore action space
            action = random.choice(actions_list)
            action_index = actions_list.index(action)
        else:
            action_index = np.argmax(q_table[state])
            action = actions_list[action_index]
        epsilon *= eps_decay
        epsilon = max(eps_min, epsilon)

        pre_next_state, reward, done, info = env.step(action) 
        next_state = preprocess(pre_next_state)

        old_value = q_table[state][action_index]

        if next_state not in q_table.keys():
            q_table[next_state] = initial_vec
            #print("TRIGGERED", count)

        #print(q_table)
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state][action_index] = new_value

        state = next_state
        #env.render()

    if i % 100 == 0:
        print("Episode: " + str(i), "epsilon = " + str(epsilon))
        with open('q_tables/q_table_' + str(i) + '.pickle' , 'wb') as handle:
            pickle.dump(q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('q_tables/q_table_FINAL.pickle', 'wb') as handle:
        pickle.dump(q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Training finished.\n")