#!/usr/bin/env python3
"""
Simple test script to verify detection functionality
"""

import cv2
import numpy as np
import sys
import os

def test_camera():
    """Test camera functionality"""
    print("Testing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not available")
        return False
    
    ret, frame = cap.read()
    if ret:
        print("✅ Camera working - Frame shape:", frame.shape)
        cap.release()
        return True
    else:
        print("❌ Could not read from camera")
        cap.release()
        return False

def test_lane_detection():
    """Test lane detection"""
    print("Testing lane detection...")
    try:
        from main import FindLaneLines
        
        # Create a test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw some test lane lines
        cv2.line(test_img, (100, 400), (200, 200), (255, 255, 255), 5)
        cv2.line(test_img, (500, 400), (400, 200), (255, 255, 255), 5)
        
        # Test lane detection
        findLaneLines = FindLaneLines()
        result = findLaneLines.forward(test_img)
        
        if result is not None:
            print("✅ Lane detection working")
            return True
        else:
            print("❌ Lane detection failed")
            return False
            
    except Exception as e:
        print(f"❌ Lane detection error: {e}")
        return False

def test_object_detection():
    """Test object detection"""
    print("Testing object detection...")
    try:
        from Object_dectection_yolov8 import ObjectDetector
        
        # Create a test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test object detection
        detector = ObjectDetector()
        result, detections = detector.detect_objects(test_img)
        
        if result is not None:
            print("✅ Object detection working")
            return True
        else:
            print("❌ Object detection failed")
            return False
            
    except Exception as e:
        print(f"❌ Object detection error: {e}")
        return False

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    required_modules = [
        'cv2', 'numpy', 'flask', 'ultralytics', 
        'matplotlib', 'sklearn', 'gtts', 'pygame'
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"Missing modules: {failed_imports}")
        return False
    return True

def main():
    """Run all tests"""
    print("🔍 Running Detection System Tests...")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Camera", test_camera),
        ("Lane Detection", test_lane_detection),
        ("Object Detection", test_object_detection)
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
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is ready to run.")
        print("\n🚀 To start the application:")
        print("   python app.py")
        print("   Then open: http://localhost:5000")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Common fixes:")
        print("   pip install -r requirements.txt")
        print("   Check camera permissions")
        print("   Ensure all files are in correct locations")

if __name__ == "__main__":
    main()

