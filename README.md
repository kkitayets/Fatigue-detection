# Fatigue Detection System

A real-time fatigue monitoring program designed for truck drivers to detect signs of drowsiness and inattention using facial feature analysis.

Features

Eye Aspect Ratio (EAR) Calculation: Detects eye blinks and prolonged eye closure
Mouth Opening Detection: Identifies yawning or open mouth states
Head Tilt Analysis: Monitors head position for signs of fatigue
Real-time Processing: Optimized for performance with lower resolution settings
Visual Feedback: Displays facial landmarks and detection results on video feed
Requirements

Python 3.x
OpenCV (cv2)
dlib
NumPy
Installation

Install required packages:
bash
pip install opencv-python dlib numpy
Download the facial landmark predictor file:
Obtain shape_predictor_68_face_landmarks.dat from dlib's website
Place it in the same directory as the script
Usage

Run the program:

bash
python main.py
Press 'q' to quit the application.

Detection Parameters

EAR_THRESHOLD: 0.2 (Eye closure threshold)
EYE_AR_CONSEC_FRAMES: 48 (Consecutive frames for blink detection)
EYE_CLOSE_THRESHOLD: 0.3 (Threshold for closed eyes)
MOUTH_OPEN_THRESHOLD: 0.5 (Mouth openness ratio)
HEAD_TILT_THRESHOLD: 10.0 degrees (Head tilt threshold)
How It Works

Captures video feed from webcam (default camera)
Detects faces using dlib's frontal face detector
Identifies 68 facial landmarks
Calculates:
Eye Aspect Ratio for blink detection
Mouth opening ratio for yawn detection
Head tilt angle for posture monitoring
Issues alerts when signs of fatigue are detected
Alerts

The system triggers warnings when detecting:

Prolonged eye closure
Absence of blinking
Yawning (mouth opening)
Head tilting
Notes

For best results, ensure adequate lighting on the driver's face
Camera should be positioned to clearly capture facial features
Parameters may need adjustment for individual users
