import cv2
import numpy as np

def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)
    
    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255

def sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """Apply Sobel edge detection with thresholding."""
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    return threshold_abs(scaled_sobel, thresh[0], thresh[1])

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """Apply magnitude thresholding."""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    scaled_magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    return threshold_abs(scaled_magnitude, mag_thresh[0], mag_thresh[1])

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """Apply direction thresholding."""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    direction = np.arctan2(abs_sobely, abs_sobelx)
    return threshold_abs(direction, thresh[0], thresh[1])

class Thresholding:
    """ This class is for extracting relevant pixels in an image with improved algorithms.
    """
    def __init__(self):
        """ Init Thresholding with optimized parameters."""
        # Sobel parameters - very sensitive for lane detection
        self.sobel_kernel = 3
        self.sobel_x_thresh = (5, 100)   # Very sensitive
        self.sobel_y_thresh = (5, 100)   # Very sensitive
        self.mag_thresh = (10, 100)      # Very sensitive
        self.dir_thresh = (0.3, 1.7)     # Very sensitive
        
        # Color space parameters - very sensitive
        self.h_thresh = (10, 40)         # Very sensitive
        self.s_thresh = (50, 255)        # Very sensitive
        self.l_thresh = (80, 255)        # Very sensitive
        self.v_thresh = (100, 255)       # Very sensitive

    def forward(self, img):
        """ Take an image and extract all relevant pixels with improved thresholding.

        Parameters:
            img (np.array): Input image

        Returns:
            binary (np.array): A binary image representing all positions of relevant pixels.
        """
        # Convert to different color spaces
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        v_channel = hsv[:,:,2]

        # Sobel edge detection
        sobelx = sobel_threshold(gray, orient='x', sobel_kernel=self.sobel_kernel, thresh=self.sobel_x_thresh)
        sobely = sobel_threshold(gray, orient='y', sobel_kernel=self.sobel_kernel, thresh=self.sobel_y_thresh)
        mag_binary = mag_threshold(gray, sobel_kernel=self.sobel_kernel, mag_thresh=self.mag_thresh)
        dir_binary = dir_threshold(gray, sobel_kernel=self.sobel_kernel, thresh=self.dir_thresh)
        
        # Combine Sobel outputs
        sobel_combined = np.zeros_like(gray)
        sobel_combined[((sobelx == 1) & (sobely == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        # Color thresholding
        h_binary = threshold_abs(h_channel, self.h_thresh[0], self.h_thresh[1])
        s_binary = threshold_abs(s_channel, self.s_thresh[0], self.s_thresh[1])
        l_binary = threshold_abs(l_channel, self.l_thresh[0], self.l_thresh[1])
        v_binary = threshold_abs(v_channel, self.v_thresh[0], self.v_thresh[1])
        
        # Combine color thresholds
        color_combined = np.zeros_like(gray)
        color_combined[((h_binary == 1) & (s_binary == 1)) | (l_binary == 1) | (v_binary == 1)] = 1

        # Combine Sobel and color thresholds
        combined = np.zeros_like(gray)
        combined[(sobel_combined == 1) | (color_combined == 1)] = 1

        # Apply region of interest mask
        mask = np.zeros_like(combined)
        height, width = combined.shape
        vertices = np.array([[(width*0.1, height), (width*0.45, height*0.6), 
                            (width*0.55, height*0.6), (width*0.9, height)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 1)
        masked = combined & mask

        return masked * 255
