"""
Configuration file for the Enhanced Detection System
"""

import os

class Config:
    """Base configuration class."""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    
    # Upload configuration
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.mpg', '.mpeg'}
    
    # Model configuration
    YOLO_MODEL_PATH = 'weights/yolov8n.pt'
    CLASS_NAMES_FILE = 'ftj/utils/coco.txt'
    
    # Detection parameters
    DEFAULT_CONFIDENCE_THRESHOLD = 0.45
    DEFAULT_NMS_THRESHOLD = 0.4
    
    # Lane detection parameters
    LANE_SMOOTHING_FACTOR = 10
    LANE_CURVE_THRESHOLD = 0.0003
    LANE_STRAIGHT_THRESHOLD = 0.0001
    LANE_MIN_PIXELS = 1500
    
    # Performance monitoring
    FPS_UPDATE_INTERVAL = 30  # Update FPS every 30 frames
    PERFORMANCE_HISTORY_SIZE = 100
    
    # Audio feedback
    AUDIO_FEEDBACK_INTERVAL = 10  # seconds between audio feedback
    AUDIO_ENABLED = True
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Video encoding
    VIDEO_QUALITY = 85  # JPEG quality (0-100)
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = 'WARNING'

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment variable."""
    config_name = os.environ.get('FLASK_ENV', 'default')
    return config.get(config_name, config['default'])

