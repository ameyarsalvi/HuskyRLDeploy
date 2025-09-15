# Running RL/ML policies on Husky

(Pre-download) The container is based on the NVIDIA container toolkit. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
Please ensure the NVIDIA container toolkit is installed on your Ubuntu computer that ensure the container's access to the NVIDIA GPU. 

1. Download this repositiory locally. Build the docker file in the repositiory using `./docker_run.sh`. The command can take several minutes to execute as it will download and build the docker container. The container will persists as the shell script does not contain the 'delete container on exit' command. This is useful as the docker container is larger and does not make sense to build it on every use. After building the container for the first time, the container id can be found out be `docker ps -a` and used to run the container again with `docker start -i _tag_`.
2. The first line of the shell script `$(pwd)/HuskyVisServo:/home/HuskyVisServo` mounts the path to your local 'HuskyVisServo' directory (that is part of the downloaded repository) to a '/home/HuskyVisServo' directory within the docker container. This is known as mounting and allows to exchange files between the host computer and the docker container without bloating the container size. Log output from the container and policies needed to run the python scripts can be added in this folder to exchange to and fro the container. This directory has also been configured as a ROS Workspace that allows to add ros/python scripts to control the Husky. It follows the standard ROS Workspace structure as :
>HuskyVisServo
>>ros_ws
>>>build
>>>devel
>>>src
>>>>axis_camera (rospkg)
>>>>husky_lf(rospkg)
>>>>>CMakeLists.txt
>>>>>packgage.xml
>>>>>src
>>>>>>script1.py
>>>>>>script2.py
3. In the directory '/HuskyVisServo' create a new directory '/trained_models'. Download one of the models (bslnCnst.zip) located at https://drive.google.com/drive/u/0/folders/1uKjiQakRyRJekMSZcmaSau9RZih8Us6l in that directory
4. In the directory '/HuskyVSDocker/HuskyVisServo/ros_ws/src/husky_LF/src/' find the find 'rl_clone_clahe.py' and ensure path to the trained model is right. This script gives example of running policies trained in stablebaselines3 on the Husky. Similar scripts (by following this script) can be now used to create deployment of machine learning policies.
5. Run `python3 rl_clone_clahe.py` to deploy the policy on the Husky.


### Docker container detials
The primary contents of the container inclued
1. Slim ubuntu20 OS
2. ROS Noetic (complete install)
3. Husky Complete packages available at (https://www.clearpathrobotics.com/assets/guides/kinetic/ros/Drive%20a%20Husky.html). These packages are critical as the build the necessary ROS message structure to parse the ROS data.
4. April tag packages.

Most of the python packgages can be tracked and updated in the requirements.txt file found at the highest level of the repository.




