#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from numpy.linalg import inv
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
bridge = CvBridge()
from stable_baselines3 import PPO
import torch
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spec = "bsln"

model = PPO.load("/home/HuskyVisServo/trained_models/Studie2/idx/idx4",device = 'cuda')
#model = PPO.load("/home/HuskyVisServo/trained_models/cone",device = 'cuda')
#model = PPO.load("/home/HuskyVisServo/trained_models/EvalNew/eval_policies/wps/2160_PathVel0.75",device = 'cuda')
#model = PPO.load("/home/HuskyVisServo/trained_models/cone",device = 'cuda')

class LineFollower(object):

    def __init__(self,spec):
        self.i=0
        self.throttle = 1
        self.cent = []
        self.agentV = []
        self.agentW = []
        self.bridge_object = CvBridge()
        self.image_sub = rospy.Subscriber("/axis/image_raw/compressed", CompressedImage, self.camera_callback)
        self.track_vel = spec
        self.obs_ = []
        print("HERE")
        #self.moveTurtlebot3_object = MoveTurtlebot3()

    def camera_callback(self, data):
        print("HERE")
        # We select bgr8 because its the OpneCV encoding by default
        cv_image = bridge.compressed_imgmsg_to_cv2(data, "bgr8")


        ###### Pre-process images for Neural Network ##########

        #imgNN = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) #Convert to Grayscale
        disp_img = cv_image[0:480, 0:640]
        imgNN_crop = cv_image[288:480, 0:640]
        imgNN_crop2 = cv_image[270:480, 0:640]
        imgNN_crop = cv2.resize(imgNN_crop, (0,0), fx=0.5, fy=0.5)
        cv2.imshow("Camera Image", disp_img)
        #cv2.imshow("Cropped2", imgNN_crop2)
        
        # HSV Thresholding
        # Convert from RGB to HSV
        hsv = cv2.cvtColor(imgNN_crop, cv2.COLOR_BGR2HSV)
        disp_img_hsv = cv2.cvtColor(disp_img, cv2.COLOR_BGR2HSV)
        # Define the Orange Colour in HSV

        """
        To know which color to track in HSV use ColorZilla to get the color registered by the camera in BGR and convert to HSV. 
        """

        # Threshold the HSV image to get only yellow colors
        s=200
        lower_orange = np.array([5,50,50])
        upper_orange = np.array([15,255,255])
        im_bw0 = cv2.inRange(hsv, lower_orange, upper_orange)
        disp_img_bw = cv2.inRange(disp_img_hsv, lower_orange, upper_orange)
        #im_bwX = im_bw0.reshape(96, 320, 3)
        im_bwX = np.empty(shape=(480, 640, 3))
        im_bwX[:,:,0] = disp_img_bw
        im_bwX[:,:,1] = disp_img_bw
        im_bwX[:,:,2] = disp_img_bw
        
        #im_bw = cv2.threshold(imgNN_crop, 125, 200, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        #cv2.imshow("Binary Thresh", im_bw0)
        noise = np.random.normal(0, 25, im_bw0.shape).astype(np.uint8)
        im_bw = np.frombuffer(im_bw0, dtype=np.uint8).reshape(96, 320, 1) # Reshape to required observation size
        im_bw_obs = im_bw
        cv2.imshow("Input", im_bw_obs)
        
        alpha = 0.5
        beta = (1.0 - alpha)
        #dst = cv2.addWeighted(imgNN_crop, alpha, im_bwX, beta, 0.0)
        dst = np.uint8(alpha*(disp_img)+beta*(im_bwX))
        cv2.imshow('Features', dst)
        
        cv2.waitKey(1)
        inputs = np.array(im_bw,dtype = np.uint8)
        
        ## Centroid calculation for logging ##
        
        #path = '/home/HuskyVisServo/output_data/sim2realCone/'
        #specifier = self.track_vel
        m = cv2.moments(im_bw, False)

        try:
            cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
        except ZeroDivisionError:
            cx, cy =  width/2, height/2
            
        cent_err = 128 - cx
        #self.cent.append(cent_err)
        #with open(path + 'cent_err_vp' + specifier,"wb") as fp:
        #	pickle.dump(self.cent, fp)
        
        if self.i == 0:
        	self.obs_ = inputs
        	
        else:
        	if self.i % self.throttle == 0:
        		self.obs_ = inputs
        	
        	else:
        		pass
        	
        cv2.imshow('RL Input', self.obs_)	

        pred = model.policy.predict(self.obs_)
        a= pred[0]
        V_pred = 0.025*a[0].item() + 0.75 # >> Constrain to [0.6 0.7] >> Complete space 0 -> 1  
        #V_pred = 0.5*a[0].item() + 0.5
        #self.agentV.append(V_pred)
        #with open(path + 'V_pred_vp' + specifier,"wb") as fp:
        #	pickle.dump(self.agentV, fp)
        	
        #omega_pred = 0.6*a[1].item()#Omega range : [-0.5 0.5]
        omega_pred = 0.5*a[1].item()
        #omega_pred = 1*a[1].item()
        #self.agentW.append(V_pred)
        #with open(path + 'omega_pred_vp' + specifier,"wb") as fp:
        #	pickle.dump(self.agentW, fp)
        	
        
        
        #print("Predicted Lin Velocity:")
        #print(V_pred)
        #print("Predicted Lin Velocity:")
        #print(omega_pred)


        ####

        #################################
        ###   ENTER CONTROLLER HERE   ###
        #################################

        pub_= rospy.Publisher('/husky_velocity_controller/cmd_vel',Twist, queue_size=10)
        msg=Twist()
        
        # Inverse Kinematics
        
        #self.t_a = 0.7510
        #self.t_b = 1.5818
        
        #A = np.array([[self.t_a*0.0825,self.t_a*0.0825],[-0.1486/self.t_b,0.1486/self.t_b]])
        
        self.t_a = 0.0770 # Virtual Radius
        self.t_b = 0.0870 #Virtual Radius/ Virtual Trackwidth
        
        
        #A = np.array([[self.t_a*0.0825,self.t_a*0.0825],[-0.1486/self.t_b,0.1486/self.t_b]])

        A = np.array([[self.t_a,self.t_a],[-self.t_b,self.t_b]])
        velocity = np.array([V_pred,omega_pred])
        phi_dots = np.matmul(inv(A),velocity) #Inverse Kinematics
        #phi_dots = phi_dots.astype(float)
        Left = phi_dots[0].item()
        Right = phi_dots[1].item()
        wheel_vel = np.array([Left,Right])
        
        # Message Level Conversion : desired wheel velocities as ROS Twist Message
        
        
        #twist_map = np.array([[0.0825,0.0825],[-0.29729,0.29729]])
        twist_vel = np.matmul(A,wheel_vel)
        twist_lin = twist_vel[0].item()
        #twist_lin = 0.75
        twist_ang = twist_vel[1].item()
        
        ##################################################
        msg.linear.x = twist_lin
        msg.angular.z = 1*twist_ang
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
    line_follower_object = LineFollower(spec)
    rate = rospy.Rate(50)
    
    
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
        

