# Running RL/ML policies on Husky

1. Download this repositiory locally. Build the docker file in the repositiory.
2. In the directory '/HuskyVisServo' create a new directory '/trained_models'. Download one of the models (bslnCnst.zip) located at https://drive.google.com/drive/u/0/folders/1uKjiQakRyRJekMSZcmaSau9RZih8Us6l in that directory
3. In the directory '/HuskyVSDocker/HuskyVisServo/ros_ws/src/husky_LF/src/' find the find 'rl_clone_clahe.py' and ensure path to the trained model is right.
4. Run rl_clone_clahe.py with python based execution.

