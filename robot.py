import gym

#env = gym.make('FetchPush-v1')
env = gym.make('BipedalWalker-v3')

#check space
print(env.observation_space) 
print(env.action_space) 
print(env.action_space.high) 
print(env.action_space.low) 

env.reset()
max_steps = env.spec.max_episode_steps 
print(max_steps)

for _ in range(10):
    done = False
    totalr = 0
    while not done:
        obs,r,done,_=env.step(env.action_space.sample()) # take a random action
        env.render()
        totalr +=r
    print(totalr)













