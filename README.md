Face Authentication-Based Screen Lock
This project uses real-time face recognition to secure a Windows system by locking the screen if the authorized user (admin) is not detected for 1.5 seconds. It leverages OpenCV, PyTorch, and facenet-pytorch (MTCNN + InceptionResnetV1) to identify faces from webcam feed and compare them against a reference image.

🔐 Features
- Real-time face detection and recognition
- Automatically locks screen when unauthorized or no face is detected
- Tolerant to brief occlusions or head movements (1.5s grace period)
- GPU acceleration supported (if available)
  
🧠 Tech Stack
- Python
- OpenCV
- PyTorch
- facenet-pytorch


