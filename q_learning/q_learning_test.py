import retro
import random
import numpy as np
import pickle

with open('q_tables/q_table_11000.pickle', 'rb') as handle:
    q_table = pickle.load(handle)

actions_list = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
                # [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]

def to_grayscale(img):
    return np.mean(img, axis=0).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return np.array_str(to_grayscale(to_grayscale(img)))

env = retro.RetroEnv(game='SuperMarioKart-Snes',obs_type =retro.Observations.IMAGE)
start_state = env.reset()
state = preprocess(start_state)
epsilon = 0.1
done = False
for i in range(1, 10000):
    start_state = env.reset()
    state = preprocess(start_state)
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            # action = env.action_space.sample() # Explore action space
            action = random.choice(actions_list)
            action_index = actions_list.index(action)
        else:
            action_index = np.argmax(q_table[state])
            action = actions_list[action_index]
                
        pre_state, _, done, _ = env.step(action) 

        state = preprocess(pre_state)

        env.render()

print("Testing finished.\n")