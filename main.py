from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import time
import base64
import numpy as np

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Define the paths and options for the gesture recognizer model
model_path = 'gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize the hand drawing utility from MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Callback function to handle results
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # Prepare the results to send back to the client
    results = []
    for i, gesture in enumerate(result.gestures):
        handedness = result.handedness[i][0].category_name
        gestures = [{"gesture": g.category_name, "score": g.score} for g in gesture]

        for gesture in gestures:
            if gesture["gesture"] == "ILoveYou":
                gesture["gesture"] = "Rock"


        results.append({"handedness": handedness, "gestures": gestures})

    # Encode the frame as JPEG
    _, buffer = cv2.imencode('.jpg', output_image.numpy_view())
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # Emit the results to the client
    socketio.emit('gesture_result', {'image': jpg_as_text, 'results': results})

# Set up the gesture recognizer options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=2,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Background thread to process video frames
def process_video():
    # Open the webcam using OpenCV
    cap = cv2.VideoCapture(0)
    
    with GestureRecognizer.create_from_options(options) as recognizer:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue

                # Convert the frame to a MediaPipe Image object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                # Get the current time in milliseconds
                timestamp_ms = int(time.time() * 1000)

                # Perform gesture recognition asynchronously
                recognizer.recognize_async(mp_image, timestamp_ms)

                # Wait briefly to simulate real-time processing
                time.sleep(0.03)

        finally:
            # Release resources
            cap.release()

# Start the background thread when the server starts
@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(process_video)

if __name__ == '__main__':
    socketio.run(app, debug=True)
