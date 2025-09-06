import cv2
import numpy as np

class LaneLines:
    """ 
    Class to detect lane lines using a simple and fast algorithm.
    """
    def __init__(self):
        """ Init LaneLines.
        This is a simple and fast lane detection algorithm that uses 
        HoughLinesP to detect lines and draw them on the screen.
        """
        # Parameters for lane detection
        self.low_threshold = 50
        self.high_threshold = 150
        self.rho = 1
        self.theta = np.pi / 180
        self.threshold = 15
        self.min_line_len = 40
        self.max_line_gap = 20
        
    def forward(self, img):
        """ 
        Take an image and detect lane lines. This is the main function.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        
        # Define a region of interest
        mask = np.zeros_like(edges)
        height, width = img.shape[:2]
        roi_vertices = np.array([[
            (0, height),
            (width / 2, height / 2),
            (width, height)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        
        # Apply mask
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            masked_edges, 
            self.rho, 
            self.theta, 
            self.threshold, 
            np.array([]), 
            self.min_line_len, 
            self.max_line_gap
        )
        
        # Create a blank image to draw lines on
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        
        # Draw the lines on the blank image
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 10)
        
        # Combine the original image with the detected lines
        combined = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
        
        return combined

# Simple test to verify that the new algorithm works
if __name__ == '__main__':
    # Create a test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw two lines to simulate lanes
    cv2.line(test_img, (100, 480), (250, 300), (255, 255, 255), 10)
    cv2.line(test_img, (540, 480), (400, 300), (255, 255, 255), 10)
    
    # Save the test image
    cv2.imwrite("test_ultra_lane_input.jpg", test_img)
    
    # Run the lane detection
    lane_lines = LaneLines()
    result = lane_lines.forward(test_img)
    
    # Save the output image
    cv2.imwrite("test_ultra_lane_output.jpg", result)
    
    print("Lane detection test complete. Check 'test_ultra_lane_output.jpg'.")
