import cv2
import numpy as np
import matplotlib.image as mpimg
from collections import deque

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    """ Class containing information about detected lane lines.

    Attributes:
        left_fit (np.array): Coefficients of a polynomial that fit left lane line
        right_fit (np.array): Coefficients of a polynomial that fit right lane line
        parameters (dict): Dictionary containing all parameters needed for the pipeline
        debug (boolean): Flag for debug/normal mode
    """
    def __init__(self):
        """Init Lanelines.

        Parameters:
            left_fit (np.array): Coefficients of polynomial that fit left lane
            right_fit (np.array): Coefficients of polynomial that fit right lane
            binary (np.array): binary image
        """
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []
        
        # Smoothing parameters
        self.smoothing_factor = 10
        self.left_fit_history = deque(maxlen=self.smoothing_factor)
        self.right_fit_history = deque(maxlen=self.smoothing_factor)
        
        # Lane detection confidence
        self.left_detected = False
        self.right_detected = False
        self.detection_failures = 0
        self.max_failures = 5
        
        # Load direction indicator images
        try:
            self.left_curve_img = mpimg.imread('left_turn.png')
            self.right_curve_img = mpimg.imread('right_turn.png')
            self.keep_straight_img = mpimg.imread('straight.png')
            
            # Normalize images
            self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.keep_straight_img = cv2.normalize(src=self.keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        except Exception as e:
            print(f"Warning: Could not load direction images: {e}")
            self.left_curve_img = None
            self.right_curve_img = None
            self.keep_straight_img = None

        # HYPERPARAMETERS - Optimized for better lane detection
        self.nwindows = 9  # More windows for better detection
        self.margin = 100   # Standard margin
        self.minpix = 20    # Very low threshold for detection
        
        # Curve detection parameters
        self.curve_threshold = 0.0001  # Very sensitive
        self.straight_threshold = 0.00005  # Very sensitive

    def forward(self, img):
        """Take a image and detect lane lines.

        Parameters:
            img (np.array): An binary image containing relevant pixels

        Returns:
            Image (np.array): An RGB image containing lane lines pixels and other details
        """
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        """ Return all pixel that in a specific window

        Parameters:
            center (tuple): coordinate of the center of the window
            margin (int): half width of the window
            height (int): height of the window

        Returns:
            pixelx (np.array): x coordinates of pixels that lie inside the window
            pixely (np.array): y coordinates of pixels that lie inside the window
        """
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx&condy], self.nonzeroy[condx&condy]

    def extract_features(self, img):
        """ Extract features from a binary image

        Parameters:
            img (np.array): A binary image
        """
        self.img = img
        # Height of of windows - based on nwindows and image shape
        self.window_height = int(img.shape[0]//self.nwindows)

        # Identify the x and y positions of all nonzero pixel in the image
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        """Find lane pixels from a binary warped image.

        Parameters:
            img (np.array): A binary warped image

        Returns:
            leftx (np.array): x coordinates of left lane pixels
            lefty (np.array): y coordinates of left lane pixels
            rightx (np.array): x coordinates of right lane pixels
            righty (np.array): y coordinates of right lane pixels
            out_img (np.array): A RGB image that use to display result later on.
        """
        assert(len(img.shape) == 2)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))

        histogram = hist(img)
        midpoint = histogram.shape[0]//2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Current position to be update later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height//2

        # Create empty lists to reveice left and right lane pixel
        leftx, lefty, rightx, righty = [], [], [], []

        # Step through the windows one by one
        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            # Append these indices to the lists
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        """Find the lane line from an image and draw it.

        Parameters:
            img (np.array): a binary warped image

        Returns:
            out_img (np.array): a RGB image that have lane line drawn on that.
        """
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        # Fit polynomials with improved validation
        left_fit = None
        right_fit = None
        
        # Very lenient detection thresholds for better detection
        if len(lefty) > 100:  # Very low threshold
            try:
                left_fit = np.polyfit(lefty, leftx, 2)
                self.left_detected = True
            except:
                self.left_detected = False
                left_fit = None
        else:
            self.left_detected = False
            left_fit = None

        if len(righty) > 100:  # Very low threshold
            try:
                right_fit = np.polyfit(righty, rightx, 2)
                self.right_detected = True
            except:
                self.right_detected = False
                right_fit = None
        else:
            self.right_detected = False
            right_fit = None

        # Apply smoothing if we have previous fits
        if self.left_detected and len(self.left_fit_history) > 0:
            # Smooth the left fit
            self.left_fit_history.append(left_fit)
            self.left_fit = np.mean(self.left_fit_history, axis=0)
        elif self.left_detected:
            self.left_fit = left_fit
            self.left_fit_history.append(left_fit)
        else:
            # Use previous fit if available
            if len(self.left_fit_history) > 0:
                self.left_fit = self.left_fit_history[-1]
            else:
                self.left_fit = np.array([0, 0, 0])

        if self.right_detected and len(self.right_fit_history) > 0:
            # Smooth the right fit
            self.right_fit_history.append(right_fit)
            self.right_fit = np.mean(self.right_fit_history, axis=0)
        elif self.right_detected:
            self.right_fit = right_fit
            self.right_fit_history.append(right_fit)
        else:
            # Use previous fit if available
            if len(self.right_fit_history) > 0:
                self.right_fit = self.right_fit_history[-1]
            else:
                self.right_fit = np.array([0, 0, 0])

        # Generate x and y values for plotting
        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))

        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))

        ploty = np.linspace(miny, maxy, img.shape[0])

        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        # Visualization with improved drawing
        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            if 0 <= l < img.shape[1] and 0 <= r < img.shape[1]:
                cv2.line(out_img, (l, y), (r, y), (0, 255, 0), 2)

        return out_img

    def plot(self, out_img):
        """Plot lane detection results with improved visualization."""
        np.set_printoptions(precision=6, suppress=True)
        lR, rR, pos = self.measure_curvature()

        # Improved curve detection
        value = None
        if abs(self.left_fit[0]) > abs(self.right_fit[0]):
            value = self.left_fit[0]
        else:
            value = self.right_fit[0]

        # More robust curve detection
        if abs(value) <= self.straight_threshold:
            self.dir.append('F')
        elif value < -self.curve_threshold:
            self.dir.append('L')
        elif value > self.curve_threshold:
            self.dir.append('R')
        else:
            self.dir.append('F')
        
        if len(self.dir) > 10:
            self.dir.pop(0)

        # Create info widget
        W = 400
        H = 500
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0,:] = [0, 0, 255]
        widget[-1,:] = [0, 0, 255]
        widget[:,0] = [0, 0, 255]
        widget[:,-1] = [0, 0, 255]
        out_img[:H, :W] = widget

        direction = max(set(self.dir), key = self.dir.count)
        msg = "Keep Straight Ahead"
        curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
        
        # Draw direction indicators
        if self.left_curve_img is not None and direction == 'L':
            y, x = self.left_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.left_curve_img[y, x, :3]
            msg = "Left Curve Ahead"
        elif self.right_curve_img is not None and direction == 'R':
            y, x = self.right_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.right_curve_img[y, x, :3]
            msg = "Right Curve Ahead"
        elif self.keep_straight_img is not None and direction == 'F':
            y, x = self.keep_straight_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.keep_straight_img[y, x, :3]

        # Add text overlays
        cv2.putText(out_img, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        # Lane keeping status
        status_color = (0, 255, 0) if abs(pos) < 0.5 else (0, 255, 255) if abs(pos) < 1.0 else (0, 0, 255)
        cv2.putText(
            out_img,
            "Good Lane Keeping" if abs(pos) < 0.5 else "Caution: Drifting" if abs(pos) < 1.0 else "Warning: Lane Departure",
            org=(10, 400),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2,
            color=status_color,
            thickness=2)

        cv2.putText(
            out_img,
            "Vehicle is {:.2f} m away from center".format(pos),
            org=(10, 450),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.66,
            color=(255, 255, 255),
            thickness=2)

        return out_img

    def measure_curvature(self):
        """Measure lane curvature and vehicle position."""
        ym = 30/720
        xm = 3.7/700

        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym

        # Compute R_curve (radius of curvature) with safety checks
        try:
            left_curveR = ((1 + (2*left_fit[0] *y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
            right_curveR = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        except:
            left_curveR = right_curveR = 1000  # Default large radius

        # Calculate vehicle position
        try:
            xl = np.dot(self.left_fit, [700**2, 700, 1])
            xr = np.dot(self.right_fit, [700**2, 700, 1])
            pos = (1280//2 - (xl+xr)//2)*xm
        except:
            pos = 0

        return left_curveR, right_curveR, pos 
    
    