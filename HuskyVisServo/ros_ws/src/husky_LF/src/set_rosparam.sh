#!/bin/bash

rosparam set /apriltag_min_pose/image_topic "/usb_cam/image_raw"
rosparam set /apriltag_min_pose/family "tag36h11"
rosparam set /apriltag_min_pose/tag_size 0.162
rosparam set /apriltag_min_pose/camera_frame "camera_link"
rosparam set /apriltag_min_pose/fx 615.0
rosparam set /apriltag_min_pose/fy 615.0
rosparam set /apriltag_min_pose/cx 320.0
rosparam set /apriltag_min_pose/cy 240.0

python3 apriltag_min_pose_full.py 
