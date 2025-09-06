"""
Memory optimization utilities for the Enhanced Detection System
"""

import gc
import psutil
import numpy as np
import cv2
import logging
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """Memory optimization class for managing system resources."""
    
    def __init__(self, max_memory_usage=0.8, cleanup_interval=30):
        """
        Initialize memory optimizer.
        
        Args:
            max_memory_usage (float): Maximum memory usage ratio (0.0-1.0)
            cleanup_interval (int): Cleanup interval in seconds
        """
        self.max_memory_usage = max_memory_usage
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        self.memory_history = deque(maxlen=100)
        self.frame_cache = deque(maxlen=5)  # Keep only last 5 frames
        self.cleanup_thread = None
        self.running = False
        
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.running = True
            self.cleanup_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            self.cleanup_thread.start()
            logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1)
        logger.info("Memory monitoring stopped")
    
    def _monitor_memory(self):
        """Monitor memory usage and perform cleanup when needed."""
        while self.running:
            try:
                current_memory = self.get_memory_usage()
                self.memory_history.append(current_memory)
                
                if current_memory > self.max_memory_usage:
                    logger.warning(f"High memory usage detected: {current_memory:.2%}")
                    self.cleanup_memory()
                
                # Periodic cleanup
                if time.time() - self.last_cleanup > self.cleanup_interval:
                    self.cleanup_memory()
                    self.last_cleanup = time.time()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(10)
    
    def get_memory_usage(self):
        """Get current memory usage ratio."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            # Return process memory usage relative to available system memory
            return memory_info.rss / system_memory.total
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0
    
    def cleanup_memory(self):
        """Perform memory cleanup operations."""
        try:
            # Clear frame cache
            self.frame_cache.clear()
            
            # Force garbage collection
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")
            
            # Clear OpenCV cache if possible
            try:
                cv2.setUseOptimized(True)
            except:
                pass
            
            logger.info("Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def optimize_image(self, img, max_size=(640, 480), quality=85):
        """
        Optimize image for memory usage.
        
        Args:
            img (np.array): Input image
            max_size (tuple): Maximum image size (width, height)
            quality (int): JPEG quality (0-100)
            
        Returns:
            np.array: Optimized image
        """
        if img is None:
            return None
        
        try:
            h, w = img.shape[:2]
            max_w, max_h = max_size
            
            # Resize if too large
            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert to appropriate data type
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            return img
            
        except Exception as e:
            logger.error(f"Error optimizing image: {e}")
            return img
    
    def cache_frame(self, frame):
        """Cache frame for potential reuse."""
        if frame is not None:
            self.frame_cache.append(frame.copy())
    
    def get_cached_frame(self):
        """Get the most recent cached frame."""
        if self.frame_cache:
            return self.frame_cache[-1]
        return None
    
    def get_memory_stats(self):
        """Get detailed memory statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                'process_memory_mb': memory_info.rss / 1024 / 1024,
                'process_memory_percent': memory_info.rss / system_memory.total * 100,
                'system_memory_total_mb': system_memory.total / 1024 / 1024,
                'system_memory_available_mb': system_memory.available / 1024 / 1024,
                'system_memory_percent': system_memory.percent,
                'cached_frames': len(self.frame_cache),
                'memory_history_avg': np.mean(self.memory_history) if self.memory_history else 0
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}

class FrameProcessor:
    """Optimized frame processor with memory management."""
    
    def __init__(self, memory_optimizer=None):
        """
        Initialize frame processor.
        
        Args:
            memory_optimizer (MemoryOptimizer): Memory optimizer instance
        """
        self.memory_optimizer = memory_optimizer or MemoryOptimizer()
        self.processing_stats = {
            'frames_processed': 0,
            'processing_time': deque(maxlen=100),
            'memory_usage': deque(maxlen=100)
        }
    
    def process_frame(self, frame, processor_func, *args, **kwargs):
        """
        Process frame with memory optimization.
        
        Args:
            frame (np.array): Input frame
            processor_func (callable): Processing function
            *args: Additional arguments for processor function
            **kwargs: Additional keyword arguments for processor function
            
        Returns:
            np.array: Processed frame
        """
        if frame is None:
            return None
        
        start_time = time.time()
        
        try:
            # Optimize input frame
            optimized_frame = self.memory_optimizer.optimize_image(frame)
            
            # Process frame
            result = processor_func(optimized_frame, *args, **kwargs)
            
            # Cache result if needed
            if result is not None:
                self.memory_optimizer.cache_frame(result)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats['frames_processed'] += 1
            self.processing_stats['processing_time'].append(processing_time)
            self.processing_stats['memory_usage'].append(self.memory_optimizer.get_memory_usage())
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame
    
    def get_processing_stats(self):
        """Get processing statistics."""
        stats = self.processing_stats.copy()
        if stats['processing_time']:
            stats['avg_processing_time'] = np.mean(stats['processing_time'])
            stats['max_processing_time'] = np.max(stats['processing_time'])
        else:
            stats['avg_processing_time'] = 0
            stats['max_processing_time'] = 0
        
        if stats['memory_usage']:
            stats['avg_memory_usage'] = np.mean(stats['memory_usage'])
            stats['max_memory_usage'] = np.max(stats['memory_usage'])
        else:
            stats['avg_memory_usage'] = 0
            stats['max_memory_usage'] = 0
        
        return stats

def optimize_numpy_arrays():
    """Optimize NumPy array operations."""
    try:
        # Enable NumPy optimizations
        np.seterr(all='ignore')  # Ignore floating point errors
        
        # Set optimal threading
        import os
        os.environ['OMP_NUM_THREADS'] = str(min(4, psutil.cpu_count()))
        os.environ['MKL_NUM_THREADS'] = str(min(4, psutil.cpu_count()))
        
        logger.info("NumPy optimizations applied")
        
    except Exception as e:
        logger.warning(f"Could not apply NumPy optimizations: {e}")

def optimize_opencv():
    """Optimize OpenCV settings."""
    try:
        # Enable OpenCV optimizations
        cv2.setUseOptimized(True)
        cv2.setNumThreads(min(4, psutil.cpu_count()))
        
        logger.info("OpenCV optimizations applied")
        
    except Exception as e:
        logger.warning(f"Could not apply OpenCV optimizations: {e}")

def get_system_info():
    """Get system information for optimization."""
    try:
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'disk_usage_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {}

# Global memory optimizer instance
memory_optimizer = MemoryOptimizer()
frame_processor = FrameProcessor(memory_optimizer)

# Apply optimizations on import
optimize_numpy_arrays()
optimize_opencv()

