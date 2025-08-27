#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from numpy.linalg import inv
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from stable_baselines3 import PPO
import torch
import pickle
from lane_detector import LaneDetector

bridge = CvBridge()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spec = "bsln"

model = PPO.load("/home/HuskyVisServo/trained_models/bslnCnst", device='cuda')
#model = PPO.load("/home/HuskyVisServo/trained_models/cone", device='cuda')

class LineFollower(object):

    def __init__(self, spec):
        self.i = 0
        self.throttle = 1
        self.cent = []
        self.agentV = []
        self.agentW = []
        self.bridge_object = CvBridge()
        self.detector = LaneDetector()
        self.image_sub = rospy.Subscriber("/axis/image_raw/compressed", CompressedImage, self.camera_callback)
        self.track_vel = spec
        self.obs_ = []
        print("HERE")

    def camera_callback(self, data):
        print("HERE")
        # Convert the compressed ROS image to an OpenCV image
        try:
            cv_image = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        ###### Pre-process images for Neural Network ##########

        disp_img = cv_image[0:480, 0:640]  # Display image size
        imgNN_crop = cv_image[288:480, 0:640]  # Cropped for processing
        imgNN_crop = cv2.resize(imgNN_crop, (320, 96))  # Resize to match the required observation size
        edges, has_lanes = self.detector.detect_lanes(imgNN_crop)
        cv2.imshow("Canny Edges", edges)

        # Convert to HSV for color thresholding
        hsv = cv2.cvtColor(imgNN_crop, cv2.COLOR_BGR2HSV)

        # Define the orange color range in HSV
        lower_orange = np.array([5, 50, 50])
        upper_orange = np.array([15, 255, 255])

        # Method 3: CLAHE with Thresholding
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv_clahe = hsv.copy()
        hsv_clahe[:, :, 2] = clahe.apply(hsv_clahe[:, :, 2])
        mask_clahe = cv2.inRange(hsv_clahe, lower_orange, upper_orange)
        #mask_clahe = cv2.GaussianBlur(mask_clahe,(9,9),35)

        # Ensure the mask is correctly shaped
        #im_bw = mask_clahe.reshape(96, 320, 1)  # Ensure the mask is in the required observation size
        im_bw = edges.reshape(96, 320, 1)

        '''
        # Debugging: Check if mask is empty
        if np.count_nonzero(mask_clahe) == 0:
            print("Warning: The mask is completely black!")

        # Overlay the mask on the original display image
        im_bw_color = cv2.cvtColor(mask_clahe, cv2.COLOR_GRAY2BGR)
        overlay_img = cv2.addWeighted(cv2.resize(disp_img, (320, 96)), 0.7, im_bw_color, 0.3, 0)
        #cv2.imshow("Overlay Image", overlay_img)

        # Display the neural network input image
        #cv2.imshow("Neural Network Input", im_bw)
        '''


        # Updating the observation for the neural network
        if self.i == 0:
            self.obs_ = im_bw
        else:
            if self.i % self.throttle == 0:
                self.obs_ = im_bw

        pred = model.policy.predict(self.obs_)
        a = pred[0]
        V_pred = 0.025 * a[0].item() + 0.75
        omega_pred = 0.5 * a[1].item()

        ####

        #################################
        ###   ENTER CONTROLLER HERE   ###
        #################################

        pub_ = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=10)
        msg = Twist()

        # Inverse Kinematics
        self.t_a = 0.0770  # Virtual Radius
        self.t_b = 0.0870  # Virtual Radius/ Virtual Trackwidth

        A = np.array([[self.t_a, self.t_a], [-self.t_b, self.t_b]])
        velocity = np.array([V_pred, omega_pred])
        phi_dots = np.matmul(inv(A), velocity)  # Inverse Kinematics
        Left = phi_dots[0].item()
        Right = phi_dots[1].item()
        wheel_vel = np.array([Left, Right])

        # Message Level Conversion: desired wheel velocities as ROS Twist Message
        twist_vel = np.matmul(A, wheel_vel)
        twist_lin = twist_vel[0].item()
        twist_ang = twist_vel[1].item()

        msg.linear.x = twist_lin
        msg.angular.z = 1 * twist_ang
        pub_.publish(msg)
        self.i = self.i + 1

        rospy.loginfo("ANGULAR VALUE SENT===>" + str(msg.angular.z))
        cv2.waitKey(1)

    def clean_up(self):
        cv2.destroyAllWindows()

def main():
    rospy.init_node('line_following_node', anonymous=True)
    line_follower_object = LineFollower(spec)
    rospy.spin()

if __name__ == '__main__':
    main()

