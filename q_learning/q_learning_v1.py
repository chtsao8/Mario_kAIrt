import retro
import random
import numpy as np

actions_list = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1,0,0,0,0,0,1,0,0,0,0,0],
                [1,0,0,0,0,0,0,1,0,0,0,0]]
actions_size = len(actions_list)
MAX_STATES = 7548


env = retro.make(game='SuperMarioKart-Snes')
env.reset()

q_table = np.zeros([MAX_STATES, actions_size])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.2

for i in range(1, 5000):
    env.reset()
    state = 0
    reward = 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            # action = env.action_space.sample() # Explore action space
            action = random.choice(actions_list)
        else:
            for index in range(actions_size):
                if np.argmax(q_table[state]) == index:
                    action = actions_list[index]

        _, reward, done, info = env.step(action) 
        
        next_state = state + 1

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state
        env.render()

    if i % 100 == 0:
        print(f"Episode: {i}")

print("Training finished.\n")