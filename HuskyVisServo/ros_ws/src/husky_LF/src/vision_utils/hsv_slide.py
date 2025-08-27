#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
#bridge = CvBridge()

class HSVSlider:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('hsv_slider', anonymous=True)

        # Create a named window
        cv2.namedWindow('HSV Slider')

        # Create trackbars for color change
        cv2.createTrackbar('HMin', 'HSV Slider', 0, 179, self.nothing)
        cv2.createTrackbar('SMin', 'HSV Slider', 0, 255, self.nothing)
        cv2.createTrackbar('VMin', 'HSV Slider', 0, 255, self.nothing)
        cv2.createTrackbar('HMax', 'HSV Slider', 0, 179, self.nothing)
        cv2.createTrackbar('SMax', 'HSV Slider', 0, 255, self.nothing)
        cv2.createTrackbar('VMax', 'HSV Slider', 0, 255, self.nothing)

        # Set default value for MAX HSV trackbars
        cv2.setTrackbarPos('HMax', 'HSV Slider', 179)
        cv2.setTrackbarPos('SMax', 'HSV Slider', 255)
        cv2.setTrackbarPos('VMax', 'HSV Slider', 255)

        # Initialize HSV min/max values
        self.h_min = self.s_min = self.v_min = 0
        self.h_max = 179
        self.s_max = self.v_max = 255

        # Initialize the CvBridge class
        self.bridge = CvBridge()

        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber("/axis/image_raw/compressed", CompressedImage, self.image_callback)

        rospy.spin()

    def nothing(self, x):
        pass

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Get current positions of all trackbars
        self.h_min = cv2.getTrackbarPos('HMin', 'HSV Slider')
        self.s_min = cv2.getTrackbarPos('SMin', 'HSV Slider')
        self.v_min = cv2.getTrackbarPos('VMin', 'HSV Slider')
        self.h_max = cv2.getTrackbarPos('HMax', 'HSV Slider')
        self.s_max = cv2.getTrackbarPos('SMax', 'HSV Slider')
        self.v_max = cv2.getTrackbarPos('VMax', 'HSV Slider')

        # Convert the image to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Create a binary thresholded image using the HSV ranges
        lower_hsv = np.array([self.h_min, self.s_min, self.v_min])
        upper_hsv = np.array([self.h_max, self.s_max, self.v_max])
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Show the original image
        cv_image_resized = cv2.resize(cv_image, (640, 480))
        cv2.imshow('Original Image', cv_image_resized)

        # Show the binary thresholded image
        mask_resized = cv2.resize(mask, (640, 480))
        cv2.imshow('Binary Image', mask_resized)

        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        HSVSlider()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()

