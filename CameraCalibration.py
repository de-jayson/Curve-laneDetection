import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import pickle
import logging

logger = logging.getLogger(__name__)

class CameraCalibration():
    """ Enhanced camera calibration class with caching and improved error handling.

    Attributes:
        mtx (np.array): Camera matrix 
        dist (np.array): Distortion coefficients
        rvecs (list): Rotation vectors
        tvecs (list): Translation vectors
        calibration_quality (float): Calibration quality metric
    """
    def __init__(self, image_dir, nx, ny, debug=False, cache_file=None):
        """ Init CameraCalibration with enhanced features.

        Parameters:
            image_dir (str): path to folder contains chessboard images
            nx (int): width of chessboard (number of squares)
            ny (int): height of chessboard (number of squares)
            debug (bool): Enable debug mode for visualization
            cache_file (str): Path to cache calibration data
        """
        self.image_dir = image_dir
        self.nx = nx
        self.ny = ny
        self.debug = debug
        self.cache_file = cache_file or f"{image_dir}_calibration.pkl"
        
        # Try to load cached calibration data
        if self._load_cached_calibration():
            logger.info("Loaded cached camera calibration data")
            return
        
        # Perform calibration if no cache found
        self._perform_calibration()
        
        # Cache the calibration data
        self._save_calibration_cache()
    
    def _load_cached_calibration(self):
        """Load calibration data from cache if available."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.mtx = data['mtx']
                    self.dist = data['dist']
                    self.rvecs = data.get('rvecs', [])
                    self.tvecs = data.get('tvecs', [])
                    self.calibration_quality = data.get('calibration_quality', 0.0)
                    return True
        except Exception as e:
            logger.warning(f"Failed to load cached calibration: {e}")
        return False
    
    def _save_calibration_cache(self):
        """Save calibration data to cache."""
        try:
            data = {
                'mtx': self.mtx,
                'dist': self.dist,
                'rvecs': self.rvecs,
                'tvecs': self.tvecs,
                'calibration_quality': self.calibration_quality
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Calibration data cached to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save calibration cache: {e}")
    
    def _perform_calibration(self):
        """Perform camera calibration from chessboard images."""
        fnames = glob.glob(f"{self.image_dir}/*.jpg") + glob.glob(f"{self.image_dir}/*.png")
        
        if not fnames:
            raise ValueError(f"No calibration images found in {self.image_dir}")
        
        logger.info(f"Found {len(fnames)} calibration images")
        
        objpoints = []
        imgpoints = []
        successful_images = 0
        
        # Coordinates of chessboard's corners in 3D
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
        
        # Go through all chessboard images
        for i, f in enumerate(fnames):
            try:
                img = mpimg.imread(f)
                if img is None:
                    logger.warning(f"Could not read image: {f}")
                    continue
                
                # Convert to grayscale image
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                # Find chessboard corners with improved parameters
                ret, corners = cv2.findChessboardCorners(
                    gray, (self.nx, self.ny),
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                          cv2.CALIB_CB_NORMALIZE_IMAGE +
                          cv2.CALIB_CB_FILTER_QUADS
                )
                
                if ret:
                    # Refine corner positions
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    imgpoints.append(corners)
                    objpoints.append(objp)
                    successful_images += 1
                    
                    if self.debug:
                        # Draw and display corners
                        img_with_corners = cv2.drawChessboardCorners(img.copy(), (self.nx, self.ny), corners, ret)
                        cv2.imshow(f'Calibration Image {i+1}', img_with_corners)
                        cv2.waitKey(500)
                else:
                    logger.warning(f"Could not find chessboard corners in: {f}")
                    
            except Exception as e:
                logger.error(f"Error processing image {f}: {e}")
                continue
        
        if self.debug:
            cv2.destroyAllWindows()
        
        if successful_images < 3:
            raise ValueError(f"Not enough successful calibration images. Found {successful_images}, need at least 3")
        
        logger.info(f"Successfully processed {successful_images} calibration images")
        
        # Perform camera calibration with improved parameters
        shape = (img.shape[1], img.shape[0])
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, shape, None, None,
            flags=cv2.CALIB_FIX_PRINCIPAL_POINT
        )
        
        if not ret:
            raise Exception("Unable to calibrate camera")
        
        # Calculate calibration quality
        self.calibration_quality = self._calculate_calibration_quality(objpoints, imgpoints)
        logger.info(f"Camera calibration completed. Quality: {self.calibration_quality:.3f}")
    
    def _calculate_calibration_quality(self, objpoints, imgpoints):
        """Calculate calibration quality metric."""
        try:
            total_error = 0
            total_points = 0
            
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(
                    objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist
                )
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
                total_points += 1
            
            mean_error = total_error / total_points
            return 1.0 / (1.0 + mean_error)  # Higher is better
        except:
            return 0.0
    
    def undistort(self, img):
        """ Return undistorted image with improved performance.

        Parameters:
            img (np.array): Input image

        Returns:
            Image (np.array): Undistorted image
        """
        if img is None:
            return None
        
        try:
            # Use optimized undistortion
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (w, h), 1, (w, h)
            )
            
            # Undistort the image
            dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
            
            # Crop the image to remove black borders
            x, y, w, h = roi
            if w > 0 and h > 0:
                dst = dst[y:y+h, x:x+w]
            
            return dst
            
        except Exception as e:
            logger.error(f"Error undistorting image: {e}")
            return img
    
    def get_calibration_info(self):
        """Get calibration information and quality metrics."""
        return {
            'calibration_quality': self.calibration_quality,
            'camera_matrix': self.mtx,
            'distortion_coefficients': self.dist,
            'num_images_used': len(self.rvecs),
            'image_size': (self.mtx[0, 2] * 2, self.mtx[1, 2] * 2)
        }
    
    def validate_calibration(self, test_image):
        """Validate calibration quality on a test image."""
        try:
            undistorted = self.undistort(test_image)
            if undistorted is None:
                return False
            
            # Check if undistorted image has reasonable dimensions
            h, w = undistorted.shape[:2]
            if h < 100 or w < 100:
                return False
            
            # Check for excessive distortion correction
            original_area = test_image.shape[0] * test_image.shape[1]
            undistorted_area = h * w
            area_ratio = undistorted_area / original_area
            
            return 0.5 < area_ratio < 1.5  # Reasonable area change
            
        except Exception as e:
            logger.error(f"Error validating calibration: {e}")
            return False
