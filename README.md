# stablebl_safedagger

## Getting Started
1. Clone this repository

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
4.

## Troubleshooting
If the command line show you t
cython, glfw, imageio, pyglet

If you have this error
ValueError: unsupported pickle protocol: 5​
```sh
pip install cloudpickle==1.6.0
```
