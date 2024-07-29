import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Define the paths and options for the gesture recognizer model
model_path = 'gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize a global variable to store the result for the interval display
latest_result = None

# Callback function to handle results
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result  # Update the latest result

# Set up the gesture recognizer options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=2,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Open the webcam using OpenCV
cap = cv2.VideoCapture(0)

# Create a gesture recognizer instance
with GestureRecognizer.create_from_options(options) as recognizer:
    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    try:
        last_display_time = time.time()  # Track the last time results were displayed
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the frame to a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Get the current time in milliseconds
            timestamp_ms = int(time.time() * 1000)

            # Perform gesture recognition asynchronously
            recognizer.recognize_async(mp_image, timestamp_ms)

            # Get the current time
            current_time = time.time()

            # Check if 1 second has passed since the last display
            if current_time - last_display_time >= 1.0:
                if latest_result is not None:
                    # Print the latest results
                    for i, gesture in enumerate(latest_result.gestures):
                        handedness = latest_result.handedness[i]
                        print(f'Hand {i + 1}:')
                        print(f'  Handedness: {handedness[0].category_name}')
                        for g in gesture:
                            print(f'  Gesture: {g.category_name} ({g.score:.2f})')
                        print()

                    # Update the last display time
                    last_display_time = current_time

            # Convert the frame to RGB for MediaPipe Hands
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hand landmarks
            results = hands.process(rgb_frame)
            
            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Display the frame with landmarks
            cv2.imshow('Gesture Recognition', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()