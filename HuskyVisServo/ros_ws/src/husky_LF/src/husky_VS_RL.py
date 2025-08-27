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

spec = "95"

model = PPO.load("/home/HuskyVisServo/trained_models/eval_models/vel_p" + spec,device = 'cuda')

class LineFollower(object):

    def __init__(self,spec):
        self.i=1
        self.cent = []
        self.agentV = []
        self.agentW = []
        self.bridge_object = CvBridge()
        self.image_sub = rospy.Subscriber("/axis/image_raw/compressed", CompressedImage, self.camera_callback)
        self.track_vel = spec
        print("HERE")
        #self.moveTurtlebot3_object = MoveTurtlebot3()

    def camera_callback(self, data):
        print("HERE")
        # We select bgr8 because its the OpneCV encoding by default
        cv_image = bridge.compressed_imgmsg_to_cv2(data, "bgr8")


        ###### Pre-process images for Neural Network ##########

        imgNN = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) #Convert to Grayscale
        imgNN_crop = imgNN[288:480, 192:448]
        cv2.imshow("Cropped", imgNN_crop)
        im_bw = cv2.threshold(imgNN_crop, 175, 255, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        cv2.imshow("Binary Thresh", im_bw)
        noise = np.random.normal(0, 25, im_bw.shape).astype(np.uint8)
        im_bw = np.frombuffer(im_bw, dtype=np.uint8).reshape(192, 256, 1) # Reshape to required observation size
        im_bw_obs = im_bw
        cv2.imshow("Input", im_bw_obs)
        cv2.waitKey(1)
        inputs = np.array(im_bw,dtype = np.uint8)
        
        ## Centroid calculation for logging ##
        
        path = '/home/HuskyVisServo/output_data/sim2real/'
        specifier = self.track_vel
        m = cv2.moments(im_bw, False)

        try:
            cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
        except ZeroDivisionError:
            cx, cy =  width/2, height/2
            
        cent_err = 128 - cx
        self.cent.append(cent_err)
        with open(path + 'cent_err_vp' + specifier,"wb") as fp:
        	pickle.dump(self.cent, fp)

        pred = model.policy.predict(inputs)
        a= pred[0]
        V_pred = 0.25*a[0].item() + 0.95 # >> Constrain to [0.6 0.7] >> Complete space 0 -> 1  
        self.agentV.append(V_pred)
        with open(path + 'V_pred_vp' + specifier,"wb") as fp:
        	pickle.dump(self.agentV, fp)
        	
        omega_pred = 0.6*a[1].item()#Omega range : [-0.5 0.5]
        self.agentW.append(V_pred)
        with open(path + 'omega_pred_vp' + specifier,"wb") as fp:
        	pickle.dump(self.agentW, fp)
        	
        
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
        
        self.t_a = 0.7510
        self.t_b = 1.5818
        
        A = np.array([[self.t_a*0.0825,self.t_a*0.0825],[-0.1486/self.t_b,0.1486/self.t_b]])
        velocity = np.array([V_pred,omega_pred])
        phi_dots = np.matmul(inv(A),velocity) #Inverse Kinematics
        #phi_dots = phi_dots.astype(float)
        Left = phi_dots[0].item()
        Right = phi_dots[1].item()
        wheel_vel = np.array([Left,Right])
        
        # Message Level Conversion : desired wheel velocities as ROS Twist Message
        
        twist_map = np.array([[0.0825,0.0825],[-0.29729,0.29729]])
        twist_vel = np.matmul(A,wheel_vel)
        twist_lin = twist_vel[0].item()
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
        

