import glob
import importlib
import os
import sys
import numpy as np
import gym
import torch as th
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout # Activation, Flatten, Reshape
import time
import yaml
from stable_baselines3.common.utils import set_random_seed
import utils.import_envs  # noqa: F401 pylint: disable=unused-import
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict


algo = "tqc"
env_id = "FetchPickAndPlace-v1"
folder = "rl-trained-agents/"
exp_id = "1"

#preparing for loading expert policy
log_path = os.path.join(folder, algo, f"{env_id}_{exp_id}")
seed = 0
found = False

for ext in ["zip"]:
    model_path = os.path.join(log_path, f"{env_id}.{ext}")
    found = os.path.isfile(model_path)
    if found:
        break

set_random_seed(seed)
th.set_num_threads(1)
stats_path = os.path.join(log_path, env_id)
hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

# load env_kwargs if existing
env_kwargs = {}
args_path = os.path.join(log_path, env_id, "args.yml")
if os.path.isfile(args_path):
    with open(args_path, "r") as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  
        if loaded_args["env_kwargs"] is not None:
            env_kwargs = loaded_args["env_kwargs"]

kwargs = dict(seed=seed)
kwargs.update(dict(buffer_size=1))

env = create_test_env(
        env_id,
        seed=seed,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

# Check if we are running python 3.8+
# we need to patch saved model under python 3.6/3.7 to load them
newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

custom_objects = {}
if newer_python_version:
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }


#calculate the difference
def check_diff(exp_act,bc_act): 
    x = np.array([exp_act,bc_act])
    diff = np.squeeze(np.diff(x, axis=0))
    for i in diff:
        i = np.abs(i) #save the difference as absolute value
        if i > 0.4:
            return i
        else: return 0

def safe_dagger(): 

    #check rendering or not
    render = False #True #False

    #dagger first loop
    print('dagger #0')
    
    returns = []
    dagger_observations = []
    dagger_actions = []
    totalr = 0.
    steps = 0
    done = False
    j=0
    k=0

    #load expert policy
    exp_model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
    #load bc policy
    bc_model = load_model('models/' + env_id + '_bc_model.h5')

    #env2 = gym.make(env_id)
    obs = env.reset()
    max_steps = 1000

    while not done:
        
        exp_action, state = exp_model.predict(obs, state=None, deterministic=True)
        bc_action = bc_model.predict(obs['observation'][None, :], batch_size = 64, verbose = 0)
        #print(exp_action,np.squeeze(bc_action,axis=0)) #check the dimensions between two actions

        #check difference
        tau = check_diff(exp_action,np.squeeze(bc_action,axis=0))

        if tau>0.4:
            dagger_observations.append(obs['observation'])
            dagger_actions.append(exp_action)
            obs, r, done, _ = env.step(exp_action)
            j+=1
        else :
            obs, r, done, _ = env.step(np.squeeze(bc_action,axis=0))
            k+=1

        totalr += r[0]
        steps += 1
        if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
        if steps >= max_steps:
            break
        if render:
            env.render("human")

    env.close()

    print('exp_action:', j,'bc_action :', k)
    print('reward : ',totalr)
    
    print(np.shape(dagger_actions))
    returns.append(totalr)
    print(returns)
    load_npz = np.load('exp_data/' + algo +'_'+ env_id+'.npz')

    obs_data = load_npz['x']
    act_data = load_npz['y']
    load_npz.close()

    
    #data aggregation
    dagger_actions = np.array(dagger_actions)
    dagger_observations = np.array(dagger_observations)
    obs_data = np.concatenate((obs_data, np.array(dagger_observations.reshape(dagger_observations.shape[0], dagger_observations.shape[2]))))
    act_data = np.concatenate((act_data, np.array(dagger_actions.reshape(dagger_actions.shape[0], dagger_actions.shape[2]))))
    
    print(np.shape(act_data))
    print(np.shape(obs_data))

    #dagger main loop
    for p in range(4):
        print('dagger #',p+1)

        model = Sequential()
        model.add(Dense(96, activation = "relu", input_shape = (obs_data.shape[1],)))
        model.add(Dense(96, activation = "relu"))
        model.add(Dense(96, activation = "relu"))
        model.add(Dense(act_data.shape[1], activation = "linear"))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='mse')
        model.fit(x=obs_data, y=act_data, batch_size=64, epochs=30, verbose=1)
        model.save('models/' + env_id + '_safedagger_model.h5') # save dagger policy

        dagger_observations = []
        dagger_actions = []
        totalr = 0.
        steps = 0
        done = False
        j=0
        k=0

        obs = env.reset()
        max_steps = 1000

        #load expert policy
        exp_model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
        #load dagger policy
        dagger_model = load_model('models/' + env_id + '_safedagger_model.h5')

        while not done:
        
            exp_action, state = exp_model.predict(obs, state=None, deterministic=True)
            dagger_action = dagger_model.predict(obs['observation'][None, :], batch_size = 64, verbose = 0)

            #check difference
            tau = check_diff(exp_action,np.squeeze(dagger_action,axis=0))

            if tau>0.4:
                dagger_observations.append(obs['observation'])
                dagger_actions.append(exp_action)
                obs, r, done, _ = env.step(exp_action)
                j+=1
            else :
                obs, r, done, _ = env.step(np.squeeze(dagger_action,axis=0))
                k+=1

            totalr += r[0]
            steps += 1
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break

            if render:
                env.render("human")

        env.close()

        print('exp_action:', j,'bc_action :', k)
        print('reward : ',totalr)
        print(np.shape(dagger_actions))
        returns.append(totalr)
        print(returns)
        #data aggregation
        dagger_actions = np.array(dagger_actions)
        dagger_observations = np.array(dagger_observations)
        obs_data = np.concatenate((obs_data, np.array(dagger_observations.reshape(dagger_observations.shape[0], dagger_observations.shape[2]))))
        act_data = np.concatenate((act_data, np.array(dagger_actions.reshape(dagger_actions.shape[0], dagger_actions.shape[2]))))
        
        print(np.shape(act_data))
        print(np.shape(obs_data))

    return returns


def behavior_cloning():
    
    #load expert data
    load_npz = np.load('exp_data/' + algo +'_'+ env_id+'.npz')

    #print(load_npz.files)
    obs_data = load_npz['x']
    act_data = load_npz['y']

    #version check
    print(tf.__version__)
    print(np.shape(obs_data))
    print(np.shape(act_data))

    #exit()
    #behavior cloning
    model = Sequential()
    model.add(Dense(96, activation = "relu", input_shape = (obs_data.shape[1],)))
    #model.add(Dropout(0.1))
    model.add(Dense(96, activation = "relu"))
    #model.add(Dropout(0.1))
    model.add(Dense(96, activation = "relu"))
    #model.add(Dropout(0.1))
    model.add(Dense(act_data.shape[1], activation = "linear"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='mse')
    model.fit(x=obs_data, y=act_data, batch_size=64, epochs=100, verbose=1)
    model.save('models/' + env_id + '_bc_model.h5') # bc policy

    bc_observations =[]
    bc_actions = []
    returns = []

    for _ in range(5):
        obs = env.reset()

        totalr = 0.
        steps = 0
        done = False
        bc_model = load_model('models/' + env_id + '_bc_model.h5')

        while not done:
            bc_action = bc_model.predict(obs['observation'][None, :], batch_size = 64, verbose = 0)
            obs, r, done, _ = env.step(np.squeeze(bc_action,axis=0))
            totalr += r[0]
            steps += 1
            env.render("human")
        print(totalr)


# def reward_plot(returns):
#     #pass
#     plt.plot([1,2,3,4,5],[-8,-8,-8,-8,-8], 'mo--',label='Expert')     # 파란색 + 마커 + 점선 #695
#     plt.plot([1,2,3,4,5],returns , 'bo--',label='SafeDAgger') #[-50,-50,-50,-50,-50]
#     plt.plot([1,2,3,4,5], [-50,-50,-50,-50,-50], 'yo--',label='BC')
#     plt.xlabel('SafeDAgger loop')
#     plt.ylabel('Rewards')
#     plt.legend(loc='lower right')
#     plt.show()

def reward_plot():
    #pass
    plt.rc('axes', labelsize=15)   # x,y축 label 폰트 크기
    plt.rc('xtick', labelsize=15)  # x축 눈금 폰트 크기 
    plt.rc('ytick', labelsize=15)  # y축 눈금 폰트 크기
    plt.plot(['BC',1,2,3,4,5],[-8,-8,-8,-8,-8,-8], 'mo--',label='Expert')     # 파란색 + 마커 + 점선 #695
    plt.plot(['BC',1,2,3,4,5],[-50,-50,-12,-12,-8,-6] , 'bo--',label='SafeDAgger') #
    #plt.plot([1,2,3,4,5], [-50,-50,-50,-50,-50], 'yo--',label='BC')
    plt.xlabel('SafeDAgger loop')
    plt.ylabel('Rewards')
    plt.legend(fontsize = 15,loc='lower right')  #'lower right',(0.7, 0.1)
    plt.show()

#behavior cloning
#behavior_cloning()

returns = safe_dagger()
#reward_plot(returns)
#reward_plot()

