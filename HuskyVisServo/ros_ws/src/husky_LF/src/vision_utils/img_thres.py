#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage

bridge = CvBridge()

#some test change line 10

class ImageProcessor:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/axis/image_raw/compressed", CompressedImage, self.camera_callback)

    def camera_callback(self, data):
        try:
            # Convert the compressed ROS image to an OpenCV image
            cv_image = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        
        # Resize image to reduce computational load
        imgNN_crop = cv2.resize(cv_image, (640, 480))
        
        # Convert to HSV for color-based thresholding
        hsv = cv2.cvtColor(imgNN_crop, cv2.COLOR_BGR2HSV)

        # Define the orange color range in HSV
        lower_orange = np.array([5, 50, 50])
        upper_orange = np.array([15, 255, 255])
        
        # Method 1: Adaptive Thresholding
        gray_image = cv2.cvtColor(imgNN_crop, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY,
                                                11, 2)
        
        # Method 2: Histogram Equalization with Thresholding
        hsv_eq = hsv.copy()
        hsv_eq[:, :, 2] = cv2.equalizeHist(hsv_eq[:, :, 2])
        mask_eq = cv2.inRange(hsv_eq, lower_orange, upper_orange)
        
        # Method 3: CLAHE with Thresholding
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv_clahe = hsv.copy()
        hsv_clahe[:, :, 2] = clahe.apply(hsv_clahe[:, :, 2])
        mask_clahe = cv2.inRange(hsv_clahe, lower_orange, upper_orange)
        
        # Method 4: LAB Thresholding
        lab = cv2.cvtColor(imgNN_crop, cv2.COLOR_BGR2LAB)
        lab_thresh = cv2.inRange(lab, np.array([0, 128, 0]), np.array([255, 255, 255]))
        
        # Method 5: Illumination Invariant with Thresholding
        invariant_img = self.illumination_invariant(cv_image)
        _, invariant_thresh = cv2.threshold(invariant_img, 128, 255, cv2.THRESH_BINARY)
        
        # Display all methods in separate windows
        cv2.imshow("Adaptive Thresholding", adaptive_thresh)
        cv2.imshow("Histogram Equalization", mask_eq)
        cv2.imshow("CLAHE", mask_clahe)
        cv2.imshow("LAB Thresholding", lab_thresh)
        cv2.imshow("Illumination Invariant", invariant_thresh)
        
        # Wait for 1 ms for a key press and then destroy the windows
        cv2.waitKey(1)
    
    def illumination_invariant(self, img):
        r = img[:, :, 2]
        g = img[:, :, 1]
        b = img[:, :, 0]
        img_invariant = np.arctan(r / (np.maximum(g, b) + 1e-6))
        img_invariant = (img_invariant - np.min(img_invariant)) / (np.max(img_invariant) - np.min(img_invariant)) * 255
        return img_invariant.astype(np.uint8)
    
    def clean_up(self):
        cv2.destroyAllWindows()

def main():
    rospy.init_node('image_processing_comparison', anonymous=True)
    image_processor = ImageProcessor()
    
    ctrl_c = False
    def shutdownhook():
        # This will be executed when the node is shut down
        image_processor.clean_up()
        rospy.loginfo("Shutting down node...")
        ctrl_c = True
    
    rospy.on_shutdown(shutdownhook)
    
    rospy.spin()

if __name__ == '__main__':
    main()

