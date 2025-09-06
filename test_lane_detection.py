#!/usr/bin/env python3
"""
Test script specifically for lane detection
"""

import cv2
import numpy as np
import sys
import os

def create_test_lane_image():
    """Create a test image with clear lane lines"""
    # Create a test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw road surface (gray)
    cv2.rectangle(img, (0, 300), (640, 480), (50, 50, 50), -1)
    
    # Draw left lane line (white)
    cv2.line(img, (200, 480), (300, 300), (255, 255, 255), 8)
    
    # Draw right lane line (white)
    cv2.line(img, (400, 480), (500, 300), (255, 255, 255), 8)
    
    # Draw center line (yellow)
    cv2.line(img, (300, 480), (400, 300), (0, 255, 255), 6)
    
    return img

def test_lane_detection_pipeline():
    """Test the complete lane detection pipeline"""
    print("Testing lane detection pipeline...")
    
    try:
        from main import FindLaneLines
        
        # Create test image
        test_img = create_test_lane_image()
        
        # Save test image for debugging
        cv2.imwrite('test_lane_input.jpg', test_img)
        print("✅ Test image saved as 'test_lane_input.jpg'")
        
        # Initialize lane detection
        findLaneLines = FindLaneLines()
        
        # Process the image
        result = findLaneLines.forward(test_img)
        
        if result is not None:
            # Save result for debugging
            cv2.imwrite('test_lane_output.jpg', result)
            print("✅ Lane detection result saved as 'test_lane_output.jpg'")
            
            # Check if lanes were detected
            if findLaneLines.left_detected or findLaneLines.right_detected:
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
        import traceback
        traceback.print_exc()
        return False

def test_thresholding():
    """Test the thresholding step specifically"""
    print("Testing thresholding...")
    
    try:
        from Thresholding import Thresholding
        
        # Create test image
        test_img = create_test_lane_image()
        
        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        thresholding = Thresholding()
        binary = thresholding.forward(test_img)
        
        if binary is not None:
            # Save thresholded image
            cv2.imwrite('test_threshold_output.jpg', binary)
            print("✅ Thresholding result saved as 'test_threshold_output.jpg'")
            
            # Check if we have white pixels (lane lines)
            white_pixels = np.sum(binary > 0)
            print(f"White pixels detected: {white_pixels}")
            
            if white_pixels > 1000:
                print("✅ Thresholding working - lane pixels detected")
                return True
            else:
                print("❌ Thresholding failed - too few lane pixels")
                return False
        else:
            print("❌ Thresholding failed - no output")
            return False
            
    except Exception as e:
        print(f"❌ Thresholding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all lane detection tests"""
    print("🔍 Testing Lane Detection...")
    print("=" * 50)
    
    tests = [
        ("Thresholding", test_thresholding),
        ("Complete Pipeline", test_lane_detection_pipeline)
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
        print("🎉 Lane detection is working!")
    else:
        print("⚠️  Lane detection needs fixing.")
        print("\n🔧 Check the saved images:")
        print("   - test_lane_input.jpg (input)")
        print("   - test_threshold_output.jpg (thresholded)")
        print("   - test_lane_output.jpg (final result)")

if __name__ == "__main__":
    main()

