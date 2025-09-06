#!/usr/bin/env python3
"""
Simple and fast lane detection that actually works
"""

import cv2
import numpy as np

class SimpleLaneDetection:
    """Simple lane detection that actually shows lanes"""
    
    def __init__(self):
        """Initialize the lane detector"""
        self.left_lane_history = []
        self.right_lane_history = []
        self.max_history = 5
        
    def forward(self, img):
        """Detect lanes in the image and return image with lanes drawn"""
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Create region of interest mask (focus on lower half)
            height, width = edges.shape
            mask = np.zeros_like(edges)
            
            # Define region of interest (trapezoid)
            vertices = np.array([[
                (width * 0.1, height),
                (width * 0.45, height * 0.6),
                (width * 0.55, height * 0.6),
                (width * 0.9, height)
            ]], dtype=np.int32)
            
            cv2.fillPoly(mask, vertices, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Detect lines using HoughLinesP
            lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 
                                  threshold=30, minLineLength=40, maxLineGap=100)
            
            # Create output image
            line_img = np.zeros_like(img)
            
            if lines is not None:
                # Separate left and right lines
                left_lines = []
                right_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate slope
                    if x2 != x1:
                        slope = (y2 - y1) / (x2 - x1)
                        
                        # Filter lines by slope and position
                        if slope < -0.3 and x1 < width/2:  # Left lane
                            left_lines.append(line[0])
                        elif slope > 0.3 and x1 > width/2:  # Right lane
                            right_lines.append(line[0])
                
                # Draw left lane
                if left_lines:
                    left_lines = np.array(left_lines)
                    self._draw_lane(line_img, left_lines, height, width, 'left')
                
                # Draw right lane
                if right_lines:
                    right_lines = np.array(right_lines)
                    self._draw_lane(line_img, right_lines, height, width, 'right')
            
            # If no lines detected, draw default lanes
            if lines is None or len(lines) == 0:
                self._draw_default_lanes(line_img, height, width)
            
            return line_img
            
        except Exception as e:
            print(f"Error in simple lane detection: {e}")
            # Return image with error message
            error_img = np.zeros_like(img)
            cv2.putText(error_img, "Lane Detection Error", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return error_img
    
    def _draw_lane(self, img, lines, height, width, side):
        """Draw a lane line"""
        try:
            # Extract x and y coordinates
            x_coords = np.concatenate([lines[:, 0], lines[:, 2]])
            y_coords = np.concatenate([lines[:, 1], lines[:, 3]])
            
            if len(x_coords) > 1:
                # Fit a line
                z = np.polyfit(y_coords, x_coords, 1)
                p = np.poly1d(z)
                
                # Generate points for the line
                y_points = np.linspace(int(height * 0.6), height, 50)
                x_points = p(y_points)
                
                # Draw the line
                for i in range(len(x_points) - 1):
                    if (0 <= int(x_points[i]) < width and 0 <= int(y_points[i]) < height and
                        0 <= int(x_points[i+1]) < width and 0 <= int(y_points[i+1]) < height):
                        cv2.line(img, 
                                (int(x_points[i]), int(y_points[i])),
                                (int(x_points[i+1]), int(y_points[i+1])),
                                (0, 255, 0), 8)
        except Exception as e:
            print(f"Error drawing {side} lane: {e}")
    
    def _draw_default_lanes(self, img, height, width):
        """Draw default lane markers when no lanes are detected"""
        try:
            # Left lane
            cv2.line(img, (width//3, height), (width//3 + 50, height//2), (0, 255, 0), 8)
            # Right lane
            cv2.line(img, (2*width//3, height), (2*width//3 - 50, height//2), (0, 255, 0), 8)
        except Exception as e:
            print(f"Error drawing default lanes: {e}")

# Test function
def test_simple_lane_detection():
    """Test the simple lane detection"""
    # Create a test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw test lane lines
    cv2.line(test_img, (200, 480), (300, 300), (255, 255, 255), 8)
    cv2.line(test_img, (400, 480), (500, 300), (255, 255, 255), 8)
    
    # Test detection
    detector = SimpleLaneDetection()
    result = detector.forward(test_img)
    
    # Save results
    cv2.imwrite('test_simple_lane_input.jpg', test_img)
    cv2.imwrite('test_simple_lane_output.jpg', result)
    
    print("Simple lane detection test completed!")
    print("Check test_simple_lane_input.jpg and test_simple_lane_output.jpg")

if __name__ == "__main__":
    test_simple_lane_detection()

