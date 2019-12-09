import retro
import random
import time
import numpy as np
from agent import DQNAgent
from wrappers import wrapper

actions_list = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
#                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
#                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]

env = retro.make(game='SuperMarioKart-Snes')
env = wrapper(env)

# Parameters
states = (84,84,4)
actions = 3

# Agent
agent = DQNAgent(states=states, actions=actions, max_memory=10000, double_q=True)

# Episodes
episodes = 50000
rewards = []

# Timing
start = time.time()
step = 0

# Main loop
for e in range(episodes):

    state = env.reset()
    total_reward = 0
    i = 0

    while True:
        env.render()

        action_index = agent.run(state=state)
        action = actions_list[action_index]
        
        next_state, reward, done, info = env.step(action)
        action = action_index
        agent.add(experience=(state, next_state, action, reward, done))
        
        agent.learn()

        total_reward += reward

        state = next_state

        i += 1

        if done:
            break

    # Rewards
    rewards.append(total_reward / i)

    # Print
    print(e)
    if e % 100 == 0:
        print('Episode {e} - +'
              'Frame {f} - +'
              'Frames/sec {fs} - +'
              'Epsilon {eps} - +'
              'Mean Reward {r}'.format(e=e,
                                       f=agent.step,
                                       fs=np.round((agent.step - step) / (time.time() - start)),
                                       eps=np.round(agent.eps, 4),
                                       r=np.mean(rewards[-100:])))
        start = time.time()
        step = agent.step

print("Training finished.\n")

# Save rewards
np.save('rewards.npy', rewards)
