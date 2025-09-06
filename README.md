# Enhanced Lane Curve Detection & Object Detection System

A high-performance, real-time lane detection and object detection system built with Python, OpenCV, YOLOv8, and Flask. This system provides advanced lane detection with curve recognition, lane departure warnings, and real-time object detection capabilities.

## 🚀 Features

### Lane Detection
- **Advanced Lane Detection**: Sophisticated algorithm with polynomial fitting and smoothing
- **Curve Recognition**: Real-time detection of left/right curves and straight lanes
- **Lane Departure Warning**: Alerts when vehicle drifts from lane center
- **Smoothing & Stability**: Temporal smoothing for stable lane detection
- **Real-time Feedback**: Audio and visual feedback for lane keeping

### Object Detection
- **YOLOv8 Integration**: State-of-the-art object detection using YOLOv8
- **Real-time Performance**: Optimized for high FPS processing
- **Multiple Object Classes**: Detects vehicles, pedestrians, traffic signs, and more
- **Confidence Thresholding**: Adjustable detection sensitivity
- **Bounding Box Visualization**: Clear visual indicators for detected objects

### Web Interface
- **Modern UI/UX**: Responsive design with real-time performance monitoring
- **Live Camera Feed**: Real-time video streaming with overlays
- **Video Upload**: Process pre-recorded videos
- **Performance Metrics**: FPS counter, detection status, and system health
- **Mobile Responsive**: Works on desktop, tablet, and mobile devices

## 🛠️ Technical Improvements

### Performance Optimizations
- **Multi-threading**: Parallel processing for better performance
- **Memory Management**: Efficient memory usage with garbage collection
- **Frame Rate Optimization**: Optimized video processing pipeline
- **GPU Acceleration**: CUDA support for faster inference (when available)

### Algorithm Enhancements
- **Improved Thresholding**: Advanced Sobel edge detection and color space analysis
- **Better Curve Detection**: More robust curve detection with configurable thresholds
- **Smoothing Algorithms**: Temporal smoothing for stable lane detection
- **Error Handling**: Comprehensive error handling and recovery

### Code Quality
- **Modular Design**: Clean, maintainable code structure
- **Configuration Management**: Centralized configuration system
- **Logging**: Comprehensive logging for debugging and monitoring
- **Type Hints**: Better code documentation and IDE support

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- OpenCV 4.9.0+
- CUDA-compatible GPU (optional, for faster processing)
- 4GB+ RAM recommended
- Webcam or video file input

### Python Dependencies
```
Flask==2.3.3
opencv-python-headless==4.9.0.80
ultralytics==8.1.34
numpy==1.26.4
matplotlib==3.8.4
moviepy==1.0.3
scikit-learn==1.5.0
gTTS==2.5.1
pygame==2.1.0.dev6
Pillow==10.0.0
torch>=1.9.0
torchvision>=0.10.0
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Ad_LaneCurve_detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO weights**
   ```bash
   # The weights should be in weights/yolov8n.pt
   # If not present, ultralytics will download them automatically
   ```

5. **Create required directories**
   ```bash
   mkdir uploads
   mkdir -p ftj/utils
   ```

## 🎯 Usage

### Starting the Application

1. **Run the Flask application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Choose detection mode**
   - **Lane Detection**: For lane and curve detection
   - **Object Detection**: For real-time object recognition

### Using the Web Interface

1. **Home Page**: Overview of available features
2. **Detection Page**: Choose between lane or object detection
3. **Live Detection**: Use your webcam for real-time processing
4. **Video Upload**: Upload and process video files
5. **Performance Monitor**: View real-time system performance

### API Endpoints

- `GET /`: Home page
- `GET /detect`: Detection selection page
- `GET /lane`: Lane detection interface
- `GET /object_detection`: Object detection interface
- `GET /video_feed`: Video stream endpoint
- `GET /performance`: Performance metrics API
- `POST /set_source`: Set video source (live/upload)
- `POST /set_mode`: Switch detection mode

## ⚙️ Configuration

The system can be configured through `config.py`:

```python
# Detection parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.45
LANE_CURVE_THRESHOLD = 0.0003
LANE_STRAIGHT_THRESHOLD = 0.0001

# Performance settings
FPS_UPDATE_INTERVAL = 30
PERFORMANCE_HISTORY_SIZE = 100

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
```

## 🔧 Advanced Features

### Lane Detection Algorithm
- **Perspective Transformation**: Bird's-eye view conversion
- **Color Space Analysis**: HLS and HSV thresholding
- **Sobel Edge Detection**: Gradient-based edge detection
- **Polynomial Fitting**: Second-order polynomial lane fitting
- **Temporal Smoothing**: Frame-to-frame stability

### Object Detection Pipeline
- **YOLOv8 Model**: Pre-trained on COCO dataset
- **Non-Maximum Suppression**: Eliminates duplicate detections
- **Confidence Filtering**: Adjustable detection sensitivity
- **Real-time Processing**: Optimized for live video streams

### Performance Monitoring
- **FPS Tracking**: Real-time frame rate monitoring
- **Memory Usage**: System resource monitoring
- **Detection Statistics**: Object count and detection rates
- **Error Logging**: Comprehensive error tracking

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check camera permissions
   - Verify camera is not being used by another application
   - Try different camera indices (0, 1, 2, etc.)

2. **Low FPS performance**
   - Reduce video resolution in config
   - Enable GPU acceleration if available
   - Close other resource-intensive applications

3. **Model loading errors**
   - Ensure YOLO weights are present in `weights/` directory
   - Check internet connection for automatic download
   - Verify file permissions

4. **Audio feedback not working**
   - Check system audio settings
   - Install required audio codecs
   - Verify pygame installation

### Performance Tips

1. **For better lane detection**:
   - Ensure good lighting conditions
   - Use high-contrast lane markings
   - Avoid extreme camera angles

2. **For better object detection**:
   - Adjust confidence threshold based on needs
   - Use higher resolution videos for better accuracy
   - Ensure objects are well-lit and visible

## 📊 Performance Benchmarks

### System Requirements
- **Minimum**: 4GB RAM, CPU-only processing
- **Recommended**: 8GB RAM, GPU acceleration
- **Optimal**: 16GB RAM, CUDA-compatible GPU

### Expected Performance
- **Lane Detection**: 15-30 FPS (CPU), 30-60 FPS (GPU)
- **Object Detection**: 10-20 FPS (CPU), 20-40 FPS (GPU)
- **Memory Usage**: 2-4GB typical, 6-8GB with GPU

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Ultralytics for YOLOv8 implementation
- Flask community for web framework
- Contributors and testers

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information
4. Include system specifications and error logs

---

**Note**: This system is designed for educational and research purposes. For production use in autonomous vehicles, additional safety measures and validation are required.