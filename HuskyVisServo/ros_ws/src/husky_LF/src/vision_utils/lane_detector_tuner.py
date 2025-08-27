#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

# Initialize bridge
bridge = CvBridge()

# Window name
WINDOW_NAME = "Canny Tuner"

def nothing(x):
    pass

def init_gui():
    """Create GUI window and sliders."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Canny Low", WINDOW_NAME, 50, 255, nothing)
    cv2.createTrackbar("Canny High", WINDOW_NAME, 150, 255, nothing)

def get_canny_params():
    """Read slider values for Canny thresholds."""
    low = cv2.getTrackbarPos("Canny Low", WINDOW_NAME)
    high = cv2.getTrackbarPos("Canny High", WINDOW_NAME)
    return max(0, low), max(low + 1, high)  # Ensure high > low

class CannyTunerNode:
    def __init__(self):
        rospy.init_node('canny_tuner_node', anonymous=True)
        rospy.Subscriber("/axis/image_raw/compressed", CompressedImage, self.image_callback, queue_size=1)
        init_gui()
        self.latest_edges = None

    def image_callback(self, data):
        try:
            cv_image = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Crop region for neural network (bottom part)
        imgNN_crop = cv_image[288:480, 0:640]
        imgNN_crop = cv2.resize(imgNN_crop, (320, 96))

        # Convert to grayscale and blur
        gray = cv2.cvtColor(imgNN_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Read current slider values
        low, high = get_canny_params()

        # Run Canny
        edges = cv2.Canny(blurred, low, high)
        self.latest_edges = edges

    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.latest_edges is not None:
                edges_rgb = cv2.cvtColor(self.latest_edges, cv2.COLOR_GRAY2BGR)
                cv2.imshow(WINDOW_NAME, edges_rgb)
            cv2.waitKey(1)
            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    node = CannyTunerNode()
    node.spin()
