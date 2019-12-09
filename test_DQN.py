import retro
import numpy as np
from test_agent import DQNAgent
import tensorflow as tf
from time import time
from wrappers import wrapper
import gym
from gym import wrappers

actions_list = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
                # [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
                # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]

env = retro.make(game='SuperMarioKart-Snes')
env = wrapper(env)
#env = wrappers.Monitor(env, './videos/'+ str(time()), video_callable=lambda episode_id: True,force=True)

states = (84,84,4)
actions = 3

agent = DQNAgent(states=states, actions=actions, max_memory=1, double_q=True)

episodes = 1000
rewards = []

start = time()
step = 0

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

        total_reward += reward

        state = next_state

        i += 1

        if done:
            action_index = agent.run(state=state)
            action = actions_list[action_index]
            
            next_state, reward, done, info = env.step(action)

            action = action_index

            total_reward += reward

            state = next_state
            break

    rewards.append(total_reward / i)

    print(e)
    if e % 100 == 0:
        print('Episode {e} - +'
              'Frame {f} - +'
              'Frames/sec {fs} - +'
              'Epsilon {eps} - +'
              'Mean Reward {r}'.format(e=e,
                                       f=agent.step,
                                       fs=np.round((agent.step - step) / (time() - start)),
                                       eps=np.round(agent.eps, 4),
                                       r=np.mean(rewards[-100:])))
        start = time()
        step = agent.step

# Save rewards
print("Training finished.\n")

np.save('rewards.npy', rewards)