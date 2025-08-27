#!/usr/bin/env bash

xhost +local:docker

sudo docker run -v $(pwd)/HuskyVisServo:/home/HuskyVisServo \
	   --gpus all -it \
           --ipc host \
	   --network host \
           --env DISPLAY=$DISPLAY \
           asalvi179/husky_base_demo:deploy_readyV2 bash
