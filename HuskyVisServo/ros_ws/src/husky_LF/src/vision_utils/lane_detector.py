import cv2
import numpy as np

class LaneDetector:
    def __init__(self, low_threshold=175, high_threshold=250, edge_density_thresh=0.01):
        """
        :param low_threshold: Lower bound for Canny edge detection
        :param high_threshold: Upper bound for Canny edge detection
        :param edge_density_thresh: Minimum edge pixel ratio to classify as lane presence
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.edge_density_thresh = edge_density_thresh

    def detect_lanes(self, imgNN_crop):
        """
        Detects lane-like features using Canny edge detection.

        :param imgNN_crop: RGB image (numpy array)
        :return: Tuple (edges, lanes_present: bool)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(imgNN_crop, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)

        # Heuristic: Check edge density in lower half (assume lanes are in bottom part of the image)
        height, width = edges.shape
        bottom_half = edges[int(0.5 * height):, :]

        edge_pixel_count = np.count_nonzero(bottom_half)
        total_pixel_count = bottom_half.size
        edge_density = edge_pixel_count / total_pixel_count

        lanes_present = edge_density > self.edge_density_thresh
        return edges, lanes_present
