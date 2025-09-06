import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PerspectiveTransformation:
    """ Enhanced perspective transformation class with adaptive region selection.

    Attributes:
        src (np.array): Coordinates of 4 source points
        dst (np.array): Coordinates of 4 destination points
        M (np.array): Matrix to transform image from front view to top view
        M_inv (np.array): Matrix to transform image from top view to front view
        img_size (tuple): Default image size for transformation
        adaptive_mode (bool): Whether to use adaptive region selection
    """
    def __init__(self, img_size=(1280, 720), adaptive_mode=True):
        """Init PerspectiveTransformation with enhanced features.

        Parameters:
            img_size (tuple): Default image size (width, height)
            adaptive_mode (bool): Enable adaptive region selection
        """
        self.img_size = img_size
        self.adaptive_mode = adaptive_mode
        self.width, self.height = img_size
        
        # Default source points (relative to image size)
        self.src_relative = np.float32([
            [0.43, 0.64],  # top-left
            [0.12, 1.0],   # bottom-left
            [0.94, 1.0],   # bottom-right
            [0.60, 0.64]   # top-right
        ])
        
        # Default destination points
        self.dst_relative = np.float32([
            [0.08, 0.0],   # top-left
            [0.08, 1.0],   # bottom-left
            [0.86, 1.0],   # bottom-right
            [0.86, 0.0]    # top-right
        ])
        
        # Initialize transformation matrices
        self._update_transformation_matrices()
        
        # Adaptive parameters
        self.adaptive_threshold = 0.1
        self.last_successful_src = None
        
    def _update_transformation_matrices(self):
        """Update transformation matrices based on current image size."""
        # Convert relative coordinates to absolute
        self.src = self.src_relative * np.array([self.width, self.height])
        self.dst = self.dst_relative * np.array([self.width, self.height])
        
        # Calculate transformation matrices
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)
        
        logger.debug(f"Updated transformation matrices for image size: {self.img_size}")
    
    def _detect_lane_region(self, img):
        """Detect optimal lane region for perspective transformation."""
        if not self.adaptive_mode:
            return self.src
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                  minLineLength=100, maxLineGap=50)
            
            if lines is None or len(lines) < 2:
                return self.src
            
            # Filter and sort lines
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                
                if abs(slope) > 0.3:  # Filter out nearly horizontal lines
                    if slope < 0:  # Left lane
                        left_lines.append(line[0])
                    else:  # Right lane
                        right_lines.append(line[0])
            
            if len(left_lines) < 1 or len(right_lines) < 1:
                return self.src
            
            # Find intersection points
            h, w = img.shape[:2]
            y_bottom = h
            y_top = int(h * 0.6)
            
            # Calculate lane boundaries
            left_x_bottom, left_x_top = self._extrapolate_lane(left_lines, y_bottom, y_top)
            right_x_bottom, right_x_top = self._extrapolate_lane(right_lines, y_bottom, y_top)
            
            # Create new source points
            new_src = np.float32([
                [left_x_top, y_top],      # top-left
                [left_x_bottom, y_bottom], # bottom-left
                [right_x_bottom, y_bottom], # bottom-right
                [right_x_top, y_top]      # top-right
            ])
            
            # Validate the new region
            if self._validate_region(new_src, img.shape):
                self.last_successful_src = new_src
                return new_src
            else:
                return self.src
                
        except Exception as e:
            logger.warning(f"Error in adaptive region detection: {e}")
            return self.src
    
    def _extrapolate_lane(self, lines, y_bottom, y_top):
        """Extrapolate lane lines to find x coordinates at given y positions."""
        if not lines:
            return 0, 0
        
        # Calculate average slope and intercept
        slopes = []
        intercepts = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                slopes.append(slope)
                intercepts.append(intercept)
        
        if not slopes:
            return 0, 0
        
        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)
        
        # Calculate x coordinates
        x_bottom = int((y_bottom - avg_intercept) / avg_slope) if avg_slope != 0 else 0
        x_top = int((y_top - avg_intercept) / avg_slope) if avg_slope != 0 else 0
        
        return x_bottom, x_top
    
    def _validate_region(self, src_points, img_shape):
        """Validate that the source region is reasonable."""
        h, w = img_shape[:2]
        
        # Check if all points are within image bounds
        for point in src_points:
            x, y = point
            if x < 0 or x >= w or y < 0 or y >= h:
                return False
        
        # Check if region forms a reasonable quadrilateral
        area = cv2.contourArea(src_points.astype(np.int32))
        if area < (w * h * 0.1):  # At least 10% of image area
            return False
        
        return True
    
    def forward(self, img, img_size=None, flags=cv2.INTER_LINEAR):
        """ Take a front view image and transform to top view

        Parameters:
            img (np.array): A front view image
            img_size (tuple): Size of the image (width, height)
            flags (int): flag to use in cv2.warpPerspective()

        Returns:
            Image (np.array): Top view image
        """
        if img is None:
            return None
        
        try:
            # Update image size if provided
            if img_size is not None and img_size != self.img_size:
                self.img_size = img_size
                self.width, self.height = img_size
                self._update_transformation_matrices()
            
            # Use adaptive region detection if enabled
            if self.adaptive_mode:
                adaptive_src = self._detect_lane_region(img)
                if adaptive_src is not None:
                    M_adaptive = cv2.getPerspectiveTransform(adaptive_src, self.dst)
                    return cv2.warpPerspective(img, M_adaptive, self.img_size, flags=flags)
            
            # Use default transformation
            return cv2.warpPerspective(img, self.M, self.img_size, flags=flags)
            
        except Exception as e:
            logger.error(f"Error in forward transformation: {e}")
            return img
    
    def backward(self, img, img_size=None, flags=cv2.INTER_LINEAR):
        """ Take a top view image and transform it to front view

        Parameters:
            img (np.array): A top view image
            img_size (tuple): Size of the image (width, height)
            flags (int): flag to use in cv2.warpPerspective()

        Returns:
            Image (np.array): Front view image
        """
        if img is None:
            return None
        
        try:
            # Update image size if provided
            if img_size is not None and img_size != self.img_size:
                self.img_size = img_size
                self.width, self.height = img_size
                self._update_transformation_matrices()
            
            return cv2.warpPerspective(img, self.M_inv, self.img_size, flags=flags)
            
        except Exception as e:
            logger.error(f"Error in backward transformation: {e}")
            return img
    
    def get_transformation_info(self):
        """Get information about the current transformation."""
        return {
            'source_points': self.src.tolist(),
            'destination_points': self.dst.tolist(),
            'image_size': self.img_size,
            'adaptive_mode': self.adaptive_mode
        }
    
    def set_adaptive_mode(self, enabled):
        """Enable or disable adaptive mode."""
        self.adaptive_mode = enabled
        logger.info(f"Adaptive mode {'enabled' if enabled else 'disabled'}")
    
    def update_region(self, src_points):
        """Manually update the source region points."""
        if len(src_points) != 4:
            raise ValueError("Source points must contain exactly 4 points")
        
        self.src = np.float32(src_points)
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)
        
        logger.info("Updated transformation region manually")
