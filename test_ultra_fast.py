#!/usr/bin/env python3
"""
Test script for ultra-fast detection system
"""

import cv2
import numpy as np
import time
import sys
import os

def test_webcam_speed():
    """Test webcam access speed"""
    print("Testing webcam access speed...")
    
    try:
        start_time = time.time()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not available")
            return False
        
        # Set optimized settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Test frame capture
        ret, frame = cap.read()
        end_time = time.time()
        
        cap.release()
        
        access_time = end_time - start_time
        
        if access_time < 1.0:  # Should be under 1 second
            print(f"✅ Webcam access: {access_time:.2f}s (FAST)")
            return True
        else:
            print(f"❌ Webcam access: {access_time:.2f}s (SLOW)")
            return False
            
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")
        return False

def test_object_detection_speed():
    """Test object detection speed"""
    print("Testing object detection speed...")
    
    try:
        from Object_dectection_yolov8 import ObjectDetector
        
        # Create test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detector = ObjectDetector()
        
        # Time multiple detections
        start_time = time.time()
        for i in range(10):
            result, detections = detector.detect_objects(test_img)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        if fps > 20:  # Should be at least 20 FPS
            print(f"✅ Object detection speed: {fps:.1f} FPS (ULTRA-FAST)")
            return True
        else:
            print(f"❌ Object detection speed: {fps:.1f} FPS (SLOW)")
            return False
            
    except Exception as e:
        print(f"❌ Object detection speed test failed: {e}")
        return False

def test_lane_detection():
    """Test lane detection functionality"""
    print("Testing lane detection...")
    
    try:
        from SimpleLaneDetection import SimpleLaneDetection
        
        # Create test image with lane lines
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw test lane lines
        cv2.line(test_img, (200, 480), (300, 300), (255, 255, 255), 8)
        cv2.line(test_img, (400, 480), (500, 300), (255, 255, 255), 8)
        
        detector = SimpleLaneDetection()
        result = detector.forward(test_img)
        
        if result is not None:
            # Save test images
            cv2.imwrite('test_ultra_lane_input.jpg', test_img)
            cv2.imwrite('test_ultra_lane_output.jpg', result)
            
            # Check if lanes were drawn
            green_pixels = np.sum(result[:, :, 1] > 200)  # Count green pixels
            
            if green_pixels > 1000:
                print("✅ Lane detection working - lanes detected!")
                return True
            else:
                print("❌ Lane detection failed - no lanes detected")
                return False
        else:
            print("❌ Lane detection failed - no output")
            return False
            
    except Exception as e:
        print(f"❌ Lane detection test failed: {e}")
        return False

def test_app_responsiveness():
    """Test overall app responsiveness"""
    print("Testing app responsiveness...")
    
    try:
        # Test imports
        from main import FindLaneLines
        from Object_dectection_yolov8 import ObjectDetector
        
        # Test initialization speed
        start_time = time.time()
        findLaneLines = FindLaneLines()
        object_detector = ObjectDetector()
        end_time = time.time()
        
        init_time = end_time - start_time
        
        if init_time < 5.0:  # Should initialize in under 5 seconds
            print(f"✅ App initialization: {init_time:.2f}s (FAST)")
            return True
        else:
            print(f"❌ App initialization: {init_time:.2f}s (SLOW)")
            return False
            
    except Exception as e:
        print(f"❌ App responsiveness test failed: {e}")
        return False

def main():
    """Run all ultra-fast tests"""
    print("🚀 Testing Ultra-Fast Detection System...")
    print("=" * 60)
    
    tests = [
        ("Webcam Access Speed", test_webcam_speed),
        ("Object Detection Speed", test_object_detection_speed),
        ("Lane Detection", test_lane_detection),
        ("App Responsiveness", test_app_responsiveness)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! System is ULTRA-FAST!")
        print("\n🚀 To start the ultra-fast application:")
        print("   python app.py")
        print("   Then open: http://localhost:5000")
        print("\n⚡ Expected performance:")
        print("   - Webcam access: < 1 second")
        print("   - Object detection: 20+ FPS")
        print("   - Lane detection: Working with green lines")
        print("   - Overall: Super responsive!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\n🔧 Quick fixes:")
        print("   - Ensure camera is not used by other apps")
        print("   - Check all dependencies are installed")
        print("   - Verify model files are in place")

if __name__ == "__main__":
    main()

