 # Fatigue Detection System
 A real-time fatigue monitoring program designed for truck drivers to detect signs of drowsiness and inattention using facial feature analysis.
 ![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
 ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
 ![dlib](https://img.shields.io/badge/dlib-19.x-orange.svg)
 ## Features
 - **Eye Aspect Ratio (EAR) Calculation** - Detects eye blinks and prolonged eye closure
 - **Mouth Opening Detection** - Identifies yawning or open mouth states
 - **Head Tilt Analysis** - Monitors head position for signs of fatigue
 - **Real-time Processing** - Optimized for performance with lower resolution settings
 - **Visual Feedback** - Displays facial landmarks and detection results on video feed
 ## Requirements
 - Python 3.x
 - OpenCV (`cv2`)
 - dlib
 - NumPy
 ## Installation
 1. Clone the repository:
    ```
    git clone https://github.com/yourusername/fatigue-detection.git
    cd fatigue-detection
    ```
 2. Install required packages:
    ```
    pip install opencv-python dlib numpy
    ```
 3. Download the facial landmark predictor file:
    - Obtain `shape_predictor_68_face_landmarks.dat` from [dlib's website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
    - Extract and place it in the project directory
 ## Usage
 Run the program:
    ```
    python main.py
    ```
 Press 'q' to quit the application.
 ## Detection Parameters
 | Parameter | Value | Description |
 |-----------|-------|-------------|
 | EAR_THRESHOLD | 0.2 | Eye closure threshold |
 | EYE_AR_CONSEC_FRAMES | 48 | Consecutive frames for blink detection |
 | EYE_CLOSE_THRESHOLD | 0.3 | Threshold for closed eyes |
 | MOUTH_OPEN_THRESHOLD | 0.5 | Mouth openness ratio |
 | HEAD_TILT_THRESHOLD | 10.0° | Head tilt threshold |
 ## How It Works
 1. Captures video feed from webcam (default camera)
 2. Detects faces using dlib's frontal face detector
 3. Identifies 68 facial landmarks
 4. Calculates:
    - Eye Aspect Ratio for blink detection
    - Mouth opening ratio for yawn detection
    - Head tilt angle for posture monitoring
 5. Issues alerts when signs of fatigue are detected
 ## Alerts
 The system triggers warnings when detecting:
 - 👁️ Prolonged eye closure
 - ⚡ Absence of blinking
 - 😮 Yawning (mouth opening)
 - 📏 Head tilting
 ## Project Structure
 ```
 fatigue-detection/
 ├── main.py
 ├── shape_predictor_68_face_landmarks.dat
 ├── README.md
 └── requirements.txt
 ```
 ## Notes
 - For best results, ensure adequate lighting on the driver's face
 - Camera should be positioned to clearly capture facial features
 - Parameters may need adjustment for individual users
 ## License
 This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 ## Contributing
 1. Fork the project
 2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
 3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
 4. Push to the branch (`git push origin feature/AmazingFeature`)
 5. Open a Pull Request
 ## Acknowledgments
 - Uses dlib's facial landmark detection
 - Inspired by research on driver fatigue detection systems
