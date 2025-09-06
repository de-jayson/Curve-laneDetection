 
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify
import cv2 as cv
import numpy as np
import random
from ultralytics import YOLO
import os
import io
import time
import logging
import threading
from collections import deque
from main import FindLaneLines  # Import for lane and curve detection
from Object_dectection_yolov8 import ObjectDetector  # Import enhanced object detector
from memory_optimizer import memory_optimizer, frame_processor, get_system_info  # Memory optimization
from gtts import gTTS  # Google Text-to-Speech
import pygame  # Library to play the sound

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize Pygame for playing audio
try:
    pygame.mixer.init()
    audio_available = True
except Exception as e:
    logger.warning(f"Audio initialization failed: {e}")
    audio_available = False

# Initialize enhanced object detector
try:
    object_detector = ObjectDetector()
    logger.info("Object detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize object detector: {e}")
    object_detector = None

# Initialize the FindLaneLines class for lane detection
try:
    findLaneLines = FindLaneLines()
    logger.info("Lane detection initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize lane detection: {e}")
    findLaneLines = None

# Global variables to store the selected video source and mode
video_source = 0  # Default to live camera
detection_mode = 'lane'  # Default to lane detection

# Performance monitoring
fps_counter = 0
fps_start_time = time.time()
current_fps = 0
performance_history = deque(maxlen=100)

def update_fps():
    """Update FPS counter."""
    global fps_counter, fps_start_time, current_fps
    fps_counter += 1
    if fps_counter % 30 == 0:  # Update every 30 frames
        current_time = time.time()
        current_fps = 30 / (current_time - fps_start_time)
        fps_start_time = current_time
        performance_history.append(current_fps)

def generate_frames():
    """Generate video frames with enhanced performance and error handling."""
    global video_source, detection_mode
    
    try:
        cap = cv.VideoCapture(video_source)
        if not cap.isOpened():
            logger.error("Cannot open video source")
            return
        
        # Set camera properties for instant access and maximum speed
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv.CAP_PROP_FPS, 60)  # Higher FPS
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Single buffer for instant response
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # MJPEG for speed
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure for speed
        cap.set(cv.CAP_PROP_BRIGHTNESS, 128)  # Fixed brightness
        cap.set(cv.CAP_PROP_CONTRAST, 128)  # Fixed contrast
        cap.set(cv.CAP_PROP_SATURATION, 128)  # Fixed saturation
        
        # Variable to track the last time audio was played
        last_play_time = time.time()
        feedback_messages = [
            "Good lane keeping",
            "You're doing great, stay focused",
            "Excellent lane control, well done",
            "You almost drifted off, good you're back on track",
            "Nice driving! Stay alert and keep up the good work",
            "Your lane keeping is on point, keep going!",
            "Watch out, you are getting too close to the lane boundary",
            "You're swerving, try to maintain a straighter line",
            "Caution! you're not staying centered in the lane"
        ]
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from video source")
                break
            
            try:
                # Fast frame processing without heavy optimization
                processed_frame = frame.copy()
                
                if detection_mode == 'lane' and findLaneLines is not None:
                    # Lane detection - direct processing for speed
                    try:
                        processed_frame = findLaneLines.forward(frame)
                    except Exception as e:
                        logger.warning(f"Lane detection error: {e}")
                        # Fallback: just show original frame with text
                        cv.putText(processed_frame, "Lane Detection Error", (10, 50), 
                                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Audio feedback (reduced frequency)
                    if audio_available and time.time() - last_play_time >= 15:  # 15 seconds delay
                        try:
                            lane_feedback = random.choice(feedback_messages)
                            tts = gTTS(text=lane_feedback, lang='en')
                            audio_fp = io.BytesIO()
                            tts.write_to_fp(audio_fp)
                            audio_fp.seek(0)
                            
                            pygame.mixer.music.load(audio_fp, 'mp3')
                            pygame.mixer.music.play()
                            last_play_time = time.time()
                        except Exception as e:
                            logger.warning(f"Error playing audio: {e}")
                
                elif detection_mode == 'object' and object_detector is not None:
                    # Object detection - maximum speed processing
                    try:
                        # Process every 2nd frame for speed (skip frames)
                        if frame_count % 2 == 0:
                            # Resize frame for ultra-fast processing
                            small_frame = cv.resize(frame, (256, 192))
                            processed_small, detections = object_detector.detect_objects(small_frame)
                            
                            # Resize back to original size
                            processed_frame = cv.resize(processed_small, (frame.shape[1], frame.shape[0]))
                        else:
                            # Use previous frame result for skipped frames
                            processed_frame = frame.copy()
                            detections = []
                        
                        # Draw detection count
                        count_text = f"Objects: {len(detections)}"
                        cv.putText(processed_frame, count_text, (10, 70), 
                                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    except Exception as e:
                        logger.warning(f"Object detection error: {e}")
                        # Fallback: just show original frame with text
                        cv.putText(processed_frame, "Object Detection Error", (10, 50), 
                                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if processed_frame is None:
                    continue
                
                # Update FPS counter
                update_fps()
                
                # Draw FPS counter
                fps_text = f"FPS: {current_fps:.1f}"
                cv.putText(processed_frame, fps_text, (10, 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw mode indicator
                mode_text = f"Mode: {detection_mode.upper()}"
                cv.putText(processed_frame, mode_text, (10, 110), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw memory usage
                memory_usage = memory_optimizer.get_memory_usage()
                memory_text = f"Memory: {memory_usage:.1%}"
                cv.putText(processed_frame, memory_text, (10, 150), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Ultra-fast frame encoding
                ret, buffer = cv.imencode('.jpg', processed_frame, [cv.IMWRITE_JPEG_QUALITY, 60])
                if not ret:
                    logger.warning("Failed to encode frame")
                    continue
                
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                frame_count += 1
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error in generate_frames: {e}")
    finally:
        if 'cap' in locals():
            cap.release()


@app.route('/')
def index():
    """Home page route."""
    return render_template('home.html')

@app.route('/detect')
def detect():
    """Detection page route."""
    return render_template('detect.html')

@app.route('/about')
def about():
    """About page route."""
    return render_template('about.html')

@app.route('/object_detection')
def object_detection():
    """Object detection page route."""
    return render_template('object.html')

@app.route('/lane')
def lane_detection():
    """Lane detection page route."""
    return render_template('lane.html')

@app.route('/video_feed')
def video_feed():
    """Video feed route for streaming."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/performance')
def performance():
    """Performance monitoring route with enhanced metrics."""
    avg_fps = sum(performance_history) / len(performance_history) if performance_history else 0
    
    # Get memory statistics
    memory_stats = memory_optimizer.get_memory_stats()
    processing_stats = frame_processor.get_processing_stats()
    system_info = get_system_info()
    
    # Get detection status
    lane_status = "Ready" if findLaneLines is not None else "Not Available"
    object_status = "Ready" if object_detector is not None else "Not Available"
    
    # Get current video source info
    video_source_info = "Live Camera" if video_source == 0 else f"File: {os.path.basename(video_source)}"
    
    return jsonify({
        'current_fps': current_fps,
        'average_fps': avg_fps,
        'total_frames': fps_counter,
        'detection_mode': detection_mode,
        'video_source': video_source_info,
        'lane_detection_available': findLaneLines is not None,
        'object_detection_available': object_detector is not None,
        'lane_detection_status': lane_status,
        'object_detection_status': object_status,
        'audio_available': audio_available,
        'memory_stats': memory_stats,
        'processing_stats': processing_stats,
        'system_info': system_info,
        'detection_active': True if (findLaneLines is not None or object_detector is not None) else False
    })

@app.route('/set_source', methods=['POST'])
def set_source():
    """Set video source and detection mode."""
    global video_source, detection_mode
    
    try:
        source = request.form.get('source')
        detection_mode = request.form.get('mode', 'lane')
        
        if source == 'live':
            video_source = 0
            detection_mode = 'lane'  # Live feed defaults to lane detection
            logger.info("Switched to live camera feed")
        elif source == 'upload':
            file = request.files.get('video')
            if file and file.filename:
                # Validate file type - support both video and image files
                allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.jpg', '.jpeg', '.png'}
                file_ext = os.path.splitext(file.filename)[1].lower()
                
                if file_ext not in allowed_extensions:
                    return jsonify({'error': 'Invalid file type. Please upload a video or image file.'}), 400
                
                # Save uploaded file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                video_source = file_path
                logger.info(f"Switched to uploaded file: {file.filename}")
            else:
                return jsonify({'error': 'No file uploaded'}), 400
        
        return jsonify({'success': True, 'mode': detection_mode})
        
    except Exception as e:
        logger.error(f"Error setting source: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Set detection mode."""
    global detection_mode
    
    try:
        data = request.get_json()
        new_mode = data.get('mode', 'lane')
        
        if new_mode in ['lane', 'object']:
            detection_mode = new_mode
            logger.info(f"Detection mode changed to: {detection_mode}")
            return jsonify({'success': True, 'mode': detection_mode})
        else:
            return jsonify({'error': 'Invalid mode. Use "lane" or "object".'}), 400
            
    except Exception as e:
        logger.error(f"Error setting mode: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process a single image for detection."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Read and process image
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Process based on detection mode
        mode = request.form.get('mode', 'lane')
        
        if mode == 'lane' and findLaneLines is not None:
            processed_img = findLaneLines.forward(img)
        elif mode == 'object' and object_detector is not None:
            processed_img, detections = object_detector.detect_objects(img)
        else:
            processed_img = img
        
        # Encode processed image
        ret, buffer = cv.imencode('.jpg', processed_img, [cv.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return jsonify({'error': 'Could not encode processed image'}), 500
        
        # Convert to base64 for JSON response
        import base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'processed_image': img_base64,
            'mode': mode,
            'detections': len(detections) if mode == 'object' and 'detections' in locals() else 0
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    try:
        # Create uploads directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            logger.info(f"Created uploads directory: {app.config['UPLOAD_FOLDER']}")
        
        # Check if required files exist
        required_files = ['weights/yolov8n.pt', 'ftj/utils/coco.txt']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            logger.warning(f"Missing required files: {missing_files}")
        
        # Start memory monitoring
        memory_optimizer.start_monitoring()
        
        # Start the Flask application
        logger.info("Starting Flask application...")
        logger.info("Available routes:")
        logger.info("  - / : Home page")
        logger.info("  - /detect : Detection page")
        logger.info("  - /lane : Lane detection")
        logger.info("  - /object_detection : Object detection")
        logger.info("  - /video_feed : Video stream")
        logger.info("  - /performance : Performance metrics")
        
        try:
            app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
        finally:
            # Stop memory monitoring on exit
            memory_optimizer.stop_monitoring()
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise


# Trial on  both lane and performance
# import random  # For randomly selecting feedback

# def generate_frames():
#     cap = cv.VideoCapture(video_source)
#     if not cap.isOpened():
#         print("Cannot open video source")
#         return

#     # Variable to track the last time audio was played
#     last_play_time = time.time()  # Initialize with the current time

#     # List of dynamic feedback messages
#     feedback_messages = [
#         "Good lane keeping, keep it up!",
#         "You're doing great, stay focused!",
#         "Excellent lane control, well done!",
#         "Keep driving safe and stay in your lane!",
#         "Your lane keeping is on point, keep going!",
#         "Nice driving! Stay alert and keep up the good work!"
#     ]

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         processed_frame = frame

#         if detection_mode == 'lane':
#             # Process the frame for lane detection
#             processed_frame = findLaneLines.forward(frame)

#             # Determine if the lane is straight or curved
#             if findLaneLines.is_straight_lane:
#                 lane_feedback = "Straight lane detected, keep driving steady!"
#             else:
#                 lane_feedback = "Curve ahead, adjust your steering carefully!"

#             # Randomly select a feedback message for performance
#             performance_feedback = random.choice(feedback_messages)

#             # Combine the lane detection feedback and performance feedback
#             full_feedback = f"{lane_feedback} {performance_feedback}"

#             # Check if 5 seconds have passed since the last audio play
#             if time.time() - last_play_time >= 5:  # 5 seconds delay
#                 try:
#                     # Generate speech audio in-memory
#                     tts = gTTS(text=full_feedback, lang='en')
#                     audio_fp = io.BytesIO()
#                     tts.write_to_fp(audio_fp)
#                     audio_fp.seek(0)

#                     # Play the audio directly from memory
#                     pygame.mixer.music.load(audio_fp, 'mp3')
#                     pygame.mixer.music.play()

#                     # Update the last play time
#                     last_play_time = time.time()

#                 except Exception as e:
#                     print(f"Error playing audio: {e}")

#         elif detection_mode == 'object':
#             detect_param = model.predict(source=[frame], conf=0.45, save=False)
#             DP = detect_param[0].numpy()

#             if len(DP) != 0:
#                 for i in range(len(detect_param[0])):
#                     boxes = detect_param[0].boxes
#                     box = boxes[i]
#                     clsID = box.cls.numpy()[0]
#                     conf = box.conf.numpy()[0]
#                     bb = box.xyxy.numpy()[0]

#                     cv.rectangle(
#                         frame,
#                         (int(bb[0]), int(bb[1])),
#                         (int(bb[2]), int(bb[3])),
#                         detection_colors[int(clsID)],
#                         3,
#                     )

#                     font = cv.FONT_HERSHEY_COMPLEX
#                     cv.putText(
#                         frame,
#                         class_list[int(clsID)],
#                         (int(bb[0]), int(bb[1]) - 10),
#                         font,
#                         1,
#                         (255, 255, 255),
#                         2,
#                     )

#                 processed_frame = frame

#         ret, buffer = cv.imencode('.jpg', processed_frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

