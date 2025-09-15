# Running RL/ML policies on Husky

(Pre-download) The container is based on the NVIDIA container toolkit. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
Please ensure the NVIDIA container toolkit is installed on your Ubuntu computer that ensure the container's access to the NVIDIA GPU. 

1. Download this repositiory locally. Build the docker file in the repositiory using `./docker_run.sh`. The command can take several minutes to execute as it will download and build the docker container. The container will persists as the shell script does not contain the 'delete container on exit' command. This is useful as the docker container is larger and does not make sense to build it on every use. After building the container for the first time, the container id can be found out be `docker ps -a` and used to run the container again with `docker start -i _tag_`.
2. In the directory '/HuskyVisServo' create a new directory '/trained_models'. Download one of the models (bslnCnst.zip) located at https://drive.google.com/drive/u/0/folders/1uKjiQakRyRJekMSZcmaSau9RZih8Us6l in that directory
3. In the directory '/HuskyVSDocker/HuskyVisServo/ros_ws/src/husky_LF/src/' find the find 'rl_clone_clahe.py' and ensure path to the trained model is right.
4. Run rl_clone_clahe.py with python based execution.


