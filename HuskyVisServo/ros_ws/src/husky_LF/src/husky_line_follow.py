#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
bridge = CvBridge()
#from move_robot import MoveTurtlebot3

class LineFollower(object):

    def __init__(self):
        self.i=1
        self.bridge_object = CvBridge()
        self.image_sub = rospy.Subscriber("/axis/image_raw/compressed", CompressedImage, self.camera_callback)
        print("HERE")
        #self.moveTurtlebot3_object = MoveTurtlebot3()

    def camera_callback(self, data):
        print("HERE")
        # We select bgr8 because its the OpneCV encoding by default
        cv_image = bridge.compressed_imgmsg_to_cv2(data, "bgr8")

        # We get image dimensions and crop the parts of the image we dont need
        height, width, channels = cv_image.shape
        #crop_img = cv_image[int((height/2)+100):int((height/2)+120)][1:int(width)]
        crop_img = cv_image[255:][200:int(width-200)]
        #crop_img = cv_image[340:360][1:640]

        # Convert from RGB to HSV
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        # Define the Yellow Colour in HSV

        """
        To know which color to track in HSV use ColorZilla to get the color registered by the camera in BGR and convert to HSV. 
        """

        # Threshold the HSV image to get only yellow colors
        s=70
        lower_yellow = np.array([0,0,255-s])
        upper_yellow = np.array([255,s,255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Calculate centroid of the blob of binary image using ImageMoments
        m = cv2.moments(mask, False)

        try:
            cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
        except ZeroDivisionError:
            cx, cy =  width/2, height/2
        
        # Draw the centroid in the resultut image
        # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) 
        cv2.circle(mask,(int(cx), int(cy)), 10,(0,0,255),-1)
        cv2.imshow("Original", crop_img)
        cv2.imshow("MASK", mask)
        cv2.waitKey(1)

        #################################
        ###   ENTER CONTROLLER HERE   ###
        #################################

        pub_= rospy.Publisher('/husky_velocity_controller/cmd_vel',Twist, queue_size=10)
        msg=Twist()
        k = 0.01/1
        angular_z = k*(cx-width/2)
        if self.i < 300:
            linear_x = 0.5
        else:
            linear_x = 0.5

        thresh = 1.5
        if angular_z > thresh:
            angular_z = thresh
        elif angular_z < -thresh:
            angular_z = -thresh
        print("Errors:")
        print([cx,cy,width/2,height/2])
        print("Controls")
        print([angular_z,linear_x])
        print(self.i)
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        pub_.publish(msg)
        self.i = self.i+1

        rospy.loginfo("ANGULAR VALUE SENT===>"+str(msg.angular.z))
        # Make it start turning
        #self.moveTurtlebot3_object.move_robot(msg)
        

    def clean_up(self):
        #self.moveTurtlebot3_object.clean_class()
        cv2.destroyAllWindows()

def main():
    rospy.init_node('line_following_node', anonymous=True)
    line_follower_object = LineFollower()
    rate = rospy.Rate(5)
    
    
    ctrl_c = False
    def shutdownhook():
        # works better than the rospy.is_shut_down()
        line_follower_object.clean_up()
        rospy.loginfo("shutdown time!")
        ctrl_c = True
        pub_= rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=1)
        msg=Twist()
        msg.linear.x = 0
        msg.angular.z = 0
        pub_.publish(msg)
    
    rospy.on_shutdown(shutdownhook)
    
    rospy.spin()
    
if __name__ == '__main__':
     main()
        
