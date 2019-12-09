import retro

# Value of track terrain variable
TRACK = 64

# Function defining the inner dirt area of the track
# Returns True in Mario is in the box, False if otherwise
def inBoundingBox(x,y):
    return y > 0.523*x - 53.87 and y > -0.3518*x + 170.07 and x > 94 \
    and y < 1.488*x + 526.04 and y < -0.404*x + 789.2366 and y < 0.72*x + 220.185 and x < 910 

# The main function running the baseline algorithm.
def main():
    env = retro.make(game='SuperMarioKart-Snes')
    obs = env.reset()
    actions = {'forward': [1,0,0,0,0,0,0,0,0,0,0,0],'left': [1,0,0,0,0,0,1,0,0,0,0,0],\
    'right': [1,0,0,0,0,0,0,1,0,0,0,0]}
    action = actions['forward']
    counter = 50
    while True:
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        x_pos = info['x_pos_mario']
        y_pos = info['y_pos_mario']
        surface = info['surface_type']
        
        # left side of track
        if inBoundingBox(x_pos, y_pos) and surface != TRACK:
            action = actions['right']
            counter -= 1
            if counter <= 0:
                action = actions['forward']
        # right side of track
        elif surface != TRACK:
            action = actions['left']
            counter -= 1
            if counter <= 0:
                action = actions['forward']
        else:
            action = actions['forward']
            counter = 50
        # get unstuck
        if counter <= -450:
            counter = 50

        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()