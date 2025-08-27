import rospy
import cv2
from cv_bridge import CvBridge
bridge = CvBridge()
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import os
import numpy as np
from lane_detector import LaneDetector


class ImageToVideo(object):
    def __init__(self):
        self.i = 0
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/axis/image_raw/compressed", CompressedImage, self.image_callback)
        self.video_writer = None
        self.video_writer2 = None   
        self.detector = LaneDetector()
        self.im_bwX3 = np.empty(shape=(96, 320, 3))
        self.obs_ = []
        
        self.frame_width = 640
        self.frame_height = 480
        self.frame_rate = 30
        
        output_dir = '/home/HuskyVisServo/output_data/'
        self.output_filename = os.path.join(output_dir, 'output_videoTest.mp4')
        
        self.frame_width2 = 320
        self.frame_height2 = 96
        
        output_dir = '/home/HuskyVisServo/output_data/'
        self.output_filename2 = os.path.join(output_dir, 'output_videoTest2.mp4')
        
        # Dynamics Data
        #self.agentV = []
        #self.agentW = []

    def image_callback(self, data):
        # Convert the ROS Image message to a CV2 image
        cv_image = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        
        #image processing to visualize how data is fed
        # Raw image
        disp_img_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        s=200
        lower_orange = np.array([5,50,50])
        upper_orange = np.array([15,255,255])
        disp_img_bw = cv2.inRange(disp_img_hsv, lower_orange, upper_orange)
        im_bwX = np.empty(shape=(480, 640, 3))
        im_bwX[:,:,0] = disp_img_bw
        im_bwX[:,:,1] = disp_img_bw
        im_bwX[:,:,2] = disp_img_bw
        alpha = 0.5
        beta = (1.0 - alpha)
        dst = np.uint8(alpha*(cv_image)+beta*(im_bwX))
        

        # Initialize the video writer if it hasn't been already
        if self.video_writer is None:
            self.video_writer = cv2.VideoWriter(
                self.output_filename,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.frame_rate,
                (self.frame_width, self.frame_height)
            )

        # Write the frame to the video file
        self.video_writer.write(dst)
        
        
        # Image (throttled and input to NN)
        imgNN_crop = cv_image[288:480, 0:640]
        imgNN_crop = cv2.resize(imgNN_crop, (0,0), fx=0.5, fy=0.5)
        edges, has_lanes = self.detector.detect_lanes(imgNN_crop)
        im_bw0 = edges
        #cv2.imshow("Canny Edges", edges)
        #hsv = cv2.cvtColor(imgNN_crop, cv2.COLOR_BGR2HSV)
        #im_bw0 = cv2.inRange(hsv, lower_orange, upper_orange)
        #im_bw0 = cv2.GaussianBlur(im_bw0,(9,9),35)
        im_bwX2 = np.empty(shape=(96, 320, 3))
        im_bwX2[:,:,0] = im_bw0
        im_bwX2[:,:,1] = im_bw0
        im_bwX2[:,:,2] = im_bw0
        im_bwX2 = np.uint8(im_bwX2)
        
        
        if self.i == 0:
        	self.obs_ = im_bwX2
        	
        else:
        	if self.i % 1 == 0:
        		self.obs_ = im_bwX2
        	
        	else:
        		pass
        
        # Initialize second video writer if it hasn't been already
        if self.video_writer2 is None:
            self.video_writer2 = cv2.VideoWriter(
                self.output_filename2,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.frame_rate,
                (self.frame_width2, self.frame_height2)
            )

        # Write the frame to the video file
        self.video_writer2.write(self.obs_)
        		
        		
        
        self.i = self.i+1
    
    #def data_rec(self):
    
    

    def cleanup(self):
        if self.video_writer:
            self.video_writer.release()
        if self.video_writer2:
            self.video_writer2.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('image_to_video_converter', anonymous=True)
    converter = ImageToVideo()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        converter.cleanup()

