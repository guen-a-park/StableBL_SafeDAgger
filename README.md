# StableBL_SafeDAgger

expert_pickandplace.py and safe_pickandplace.py need python 3.6/3.7

## Getting Started

1. Clone this repository  
rl trained agent 다운을 위해서는 아래 명령어로 실행(--recursive 
```sh
git clone --recursive https://github.com/guen-a-park/StableBL_SafeDAgger.git
```

2. Install dependencies
```sh
apt-get install swig cmake ffmpeg
pip install -r requirements.txt
```

3. Install mujoco-py
    1. Get mujoco license key file from <a href="https://www.roboti.us/license.html">its website</a>
    2. Create a .mujoco folder in the home directory and copy the given mjpro150 directory and your license key into it
      ```sh
      mkdir ~/.mujoco/
      cd <location_of_your_license_key>
      cp mjkey.txt ~/.mujoco/
      cd <this_repo>/mujoco
      cp -r mjpro150 ~/.mujoco/
      ```
    3. Add the following line to bottom of your .bashrc file: 
      ```sh
      export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin/
      ```

## File explanation

- **robot.py** : random action 확인가능

- **expert_pickandplace.py**, **expert_bipedwalker.py** : expert policy 실행 및 expert data 저장

- **safe_bipedwalker.py**, **safe_pickandplace.py** : bc policy 생성, safe dagger policy 생성

  

## Enjoy a Trained Agent

**example)**
```sh
python3 expert_bipedwalker.py --algo tqc --env BipedalWalker-v3 --no-render --folder rl-trained-agents/ -n 1000
```
--algo : 사용가능한 rl 알고리즘은 [rl-trained-agents](https://github.com/DLR-RM/rl-trained-agents)에서 확인가능.

--env : 실행환경 결정

--no-render : rendering 원하지 않을 경우 삽입

-n : 저장/실행하고 싶은 data의 수 지정



## Run SafeDAgger

models 폴더 안의 bc policy와 exp_data 폴더의 expert data 유무 확인. 

- expert data가 없다면  **expert_bipedwalker.py** 또는 **expert_pickandplace.py**부터 실행해 데이터 생성

- bc policy가 없다면 **safe_bipedwalker.py**에서 *behavior_cloning()* 함수를 실행해 bc policy 생성 




## Troubleshooting
If you have this error : ImportError: No module named cython, glfw, imageio, pyglet  
using 'pip install' to download each module

If you have this error : ValueError: unsupported pickle protocol: 5
```sh
pip install cloudpickle==1.6.0
```

**Note**: If you have problem with the line "you need to install mujoco_py..." 
check https://github.com/openai/mujoco-py/ and see the **Ubuntu installation troubleshooting**

## Reference
https://stable-baselines3.readthedocs.io/en/v1.0/guide/rl_zoo.html  
https://github.com/DLR-RM/rl-baselines3-zoo

