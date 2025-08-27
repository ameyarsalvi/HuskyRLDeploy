#!/usr/bin/env python3

#RosDeployment
import rospy
import cv2
import numpy as np
from numpy.linalg import inv
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
bridge = CvBridge()
import torch
import pickle


import time
import math
from numpy import savetxt
from Image2Waypoints import Image2Waypoints
from Image2Waypoints2 import Image2Waypoints2
import casadi as ca
from scipy.interpolate import interp1d



class LineFollower(object):

    def __init__(self):
        self.i=0
        self.throttle = 1
        self.cent = []
        self.error_old = 0
        self.agentV = []
        self.agentW = []
        self.bridge_object = CvBridge()
        self.image_sub = rospy.Subscriber("/axis/image_raw/compressed", CompressedImage, self.camera_callback)
        self.track_vel = 0.75
        self.obs_ = []
        print("HERE")
        #self.moveTurtlebot3_object = MoveTurtlebot3()

    def interpolate_waypoints_to_horizon(self,waypoints_meters, N):
        """Interpolate waypoints to get evenly spaced reference for NMPC."""
        x_path = waypoints_meters[:, 0]
        y_path = waypoints_meters[:, 1]

        # Compute cumulative distances along the path
        distances = np.insert(np.cumsum(np.linalg.norm(np.diff(waypoints_meters, axis=0), axis=1)), 0, 0)
        total_length = distances[-1]

        # Create interpolators
        fx = interp1d(distances, x_path, kind='linear', fill_value='extrapolate')
        fy = interp1d(distances, y_path, kind='linear', fill_value='extrapolate')

        # Interpolate along arc-length
        interp_dists = np.linspace(0, total_length, N + 1)
        x_ref = fx(interp_dists)
        y_ref = fy(interp_dists)

        # Approximate heading (theta) using gradient
        dx = np.gradient(x_ref)
        dy = np.gradient(y_ref)
        theta_ref = np.arctan2(dy, dx)

        return np.vstack([x_ref, y_ref, theta_ref])  # shape: (3, N+1)

    def create_nmpc_solver(self,N, dt,
                        Q_weights=[0.1, 0.1, 0.01],
                        R_weights=[0.1, 0.1],
                        v_des=0.75,
                        alpha=0.9,
                        v_bounds=(0.0, 1.0),
                        omega_bounds=(-1.0, 1.0)):
        """
        Creates a nonlinear MPC solver for a differential drive robot using CasADi,
        including soft tracking of a desired linear velocity and bounded controls.

        Parameters:
            N : int
                Horizon length.
            dt : float
                Timestep (s).
            Q_weights : list of 3 floats
                State cost weights [x, y, theta].
            R_weights : list of 2 floats
                Control cost weights [v, omega].
            v_des : float
                Desired linear velocity (m/s).
            alpha : float
                Penalty weight on velocity tracking (soft constraint).
            v_bounds : tuple (v_min, v_max)
                Bounds on linear velocity (m/s).
            omega_bounds : tuple (omega_min, omega_max)
                Bounds on angular velocity (rad/s).

        Returns:
            solver : CasADi solver object
            solver_vars : dict with symbolic variables and parameters
            lbx : lower bounds on decision variables
            ubx : upper bounds on decision variables
        """

        # Symbolic variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')

        states = ca.vertcat(x, y, theta)
        controls = ca.vertcat(v, omega)

        # Dynamics
        rhs = ca.vertcat(
            v * ca.cos(theta),
            v * ca.sin(theta),
            omega
        )
        f = ca.Function('f', [states, controls], [rhs])

        # Optimization variables
        X = ca.SX.sym('X', 3, N+1)
        U = ca.SX.sym('U', 2, N)
        X_ref = ca.SX.sym('X_ref', 3, N+1)

        # Cost function weights
        Q = ca.diag(ca.SX(Q_weights))
        R = ca.diag(ca.SX(R_weights))

        # Objective and constraints
        obj = 0
        g = []

        g.append(X[:, 0] - X_ref[:, 0])  # initial condition

        for k in range(N):
            x_next = X[:, k] + dt * f(X[:, k], U[:, k])
            g.append(X[:, k+1] - x_next)

            # Path tracking cost
            obj += ca.mtimes([(X[:, k] - X_ref[:, k]).T, Q, (X[:, k] - X_ref[:, k])])
            obj += ca.mtimes([U[:, k].T, R, U[:, k]])

            # Soft velocity tracking penalty
            obj += alpha * (U[0, k] - v_des)**2

        g = ca.vertcat(*g)

        # Flatten decision variables and parameters
        decision_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        params = ca.reshape(X_ref, -1, 1)

        # Define bounds
        n_X = X.size1() * (N + 1)  # 3 * (N+1)
        n_U = U.size1() * N        # 2 * N

        lbx = [-ca.inf] * n_X
        ubx = [ ca.inf] * n_X

        for _ in range(N):
            lbx += [v_bounds[0], omega_bounds[0]]
            ubx += [v_bounds[1], omega_bounds[1]]

        lbx = ca.vertcat(*lbx)
        ubx = ca.vertcat(*ubx)

        # Solver setup
        nlp = {'x': decision_vars, 'f': obj, 'g': g, 'p': params}
        opts = {"ipopt.print_level": 0, "print_time": 0}

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Return solver and useful handles
        solver_vars = {
            'X': X,
            'U': U,
            'X_ref': X_ref,
            'N': N,
            'dt': dt,
            'n_states': 3,
            'n_controls': 2
        }

        return solver, solver_vars, lbx, ubx
    

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

        #im_bw0 = cv2.inRange(hsv, lower_orange, upper_orange)
        #disp_img_bw = cv2.inRange(disp_img_hsv, lower_orange, upper_orange)

        ########### CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv_clahe = hsv.copy()
        hsv_clahe[:, :, 2] = clahe.apply(hsv_clahe[:, :, 2])
        im_bw0 = cv2.inRange(hsv_clahe, lower_orange, upper_orange)
        #im_bw0 = cv2.GaussianBlur(im_bw0,(5,5),0)
        disp_img_bw = cv2.inRange(disp_img_hsv, lower_orange, upper_orange)
        #disp_img_bw = cv2.GaussianBlur(disp_img_bw,(5,5),0)

        ###########

        im_bwX = np.empty(shape=(480, 640, 3))
        im_bwX[:,:,0] = disp_img_bw
        im_bwX[:,:,1] = disp_img_bw
        im_bwX[:,:,2] = disp_img_bw
        
        #im_bw = cv2.threshold(imgNN_crop, 125, 200, cv2.THRESH_BINARY)[1]  # convert the grayscale image to binary image
        #cv2.imshow("Binary Thresh", im_bw0)
        noise = np.random.normal(0, 25, im_bw0.shape).astype(np.uint8)
        im_bw = np.frombuffer(im_bw0, dtype=np.uint8).reshape(96, 320, 1) # Reshape to required observation size
        im_bw_obs = im_bw
        im_bw_not = cv2.bitwise_not(im_bw0)
        #im_bw_not = im_bw
        im_bw_not = np.array(im_bw_not,dtype=np.uint8)
        cv2.imshow("Input", im_bw_obs)
        
        alpha = 0.5
        beta = (1.0 - alpha)
        #dst = cv2.addWeighted(imgNN_crop, alpha, im_bwX, beta, 0.0)
        dst = np.uint8(alpha*(disp_img)+beta*(im_bwX))
        cv2.imshow('Features', dst)
        
        cv2.waitKey(1)
        inputs = np.array(im_bw,dtype = np.uint8)
        
        ## Centroid calculation for logging and PD Controller ##

        '''
        This section implements the senosr drop-out throttle option
        '''
        
        if self.i == 0:
        	self.obs_ = inputs
        	
        else:
        	if self.i % self.throttle == 0:
        		self.obs_ = inputs
        	
        	else:
        		pass
        	
        cv2.imshow('RL Input', self.obs_)	


        ##############################################
        #               MPC Controller               #
        ##############################################

        H = Image2Waypoints.getHomography()
        scale = 100
        canvas_width = int(3.5 * scale) #OG 3.5
        canvas_height = int(5.0 * scale) #OG 5
        warped = cv2.warpPerspective(im_bw_not, H, (canvas_width, canvas_height))
        warped = cv2.bitwise_not(warped)
        height, width = warped.shape  # Or: warped.shape[:2]


        #fitted_path, band_points = Image2Waypoints.fit_quadratic_on_raw_image(im_bw_not)
        left_path, right_path, center_path, vis_img = Image2Waypoints2.fit_dual_quadratics_on_raw_image(im_bw_not)

        #if vis_img is not None:
        #        cv2.imshow("Dual Lane Fit", vis_img)

        #Image2Waypoints.warp_and_overlay_path(fitted_path, warped, H)
        '''
        if fitted_path is not None:
            waypoints_meters = Image2Waypoints.convert_pixel_path_to_waypoints_in_meters(
                pixel_path=fitted_path,
                warped_shape=(height, width),
                scale=scale
            )
            last_valid_waypoints = waypoints_meters  # Update if valid
        else:
            print("[MPC Loop] Warning: fitted_path is None. Reusing last valid waypoints.")

        # If no valid path has ever been found, skip this frame
        if last_valid_waypoints is None:
            print("[MPC Loop] No valid waypoints available. Skipping MPC step.")
            #client.step()
        #continue
        '''

        H_img_to_world = np.array([
                [ 1.04660635e-02,  2.54905363e-03, -1.87768491e+00],
                [-3.84114181e-04, -1.30146589e-02,  4.17121481e+00],
                [-1.13487030e-03,  1.14988151e-02,  1.00000000e+00]
            ])

        if center_path is not None:
            waypoints = Image2Waypoints2.convert_pixel_path_to_waypoints(center_path, H_img_to_world)
            last_valid_waypoints = waypoints
        else:
            print("[MPC Loop] Warning: fitted_path is None. Reusing last valid waypoints.")

        #Image2Waypoints2.plot_waypoints(last_valid_waypoints)

        if last_valid_waypoints is None:
            print("[MPC Loop] No valid waypoints available. Skipping MPC step.")
            #client.step()
            #continue


        print(f"length of valid waypoints is :{len(last_valid_waypoints)}")

        # --- Step 2: Setup NMPC and Interpolate Waypoints ---
        #solver, vars = self.create_nmpc_solver(N=10, dt=0.02)
        solver, vars, lbx, ubx = self.create_nmpc_solver(N=10, dt=0.02)
        X_ref_np = self.interpolate_waypoints_to_horizon(last_valid_waypoints, N=vars['N'])
        X_ref_flat = X_ref_np.reshape((-1, 1))

        # Initial guess and state
        x0 = np.array([0.0, 0.0, 0.0])
        X_init = np.tile(x0.reshape(3, 1), (1, vars['N']+1))
        U_init = np.zeros((2, vars['N']))
        initial_guess = np.concatenate([X_init.reshape(-1, 1), U_init.reshape(-1, 1)], axis=0)

        # --- Step 3: Solve NMPC and Get Control ---
        sol = solver(x0=initial_guess, p=X_ref_flat, lbg=0, ubg=0)
        solution = sol['x'].full().flatten()
        U_opt = solution[3 * (vars['N']+1):].reshape((2, vars['N']))
        V_pred, omega_pred = U_opt[:, 0]
        print(V_pred)
        print(omega_pred)
       
        #################################
        ###   ENTER CONTROLLER HERE   ###
        #################################

        '''
        This section converts V and Omega to twist vel. This will stay same in all codes
        '''

        pub_= rospy.Publisher('/husky_velocity_controller/cmd_vel',Twist, queue_size=10)
        msg=Twist()
        '''
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
        '''
        ##################################################
        #msg.linear.x = twist_lin
        #msg.angular.z = 1*twist_ang
        msg.linear.x = V_pred
        msg.angular.z = omega_pred
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
        

