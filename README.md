## SignIX - Gesture recognition using Mediapipe

This project implements a real-time gesture recognition system using [MediaPipe](https://mediapipe.dev) and OpenCV. It detects hand gestures from a live webcam feed and displays hand landmarks on the video stream.

## Overview

Gesture recognition is a technology that can interpret human gestures via mathematical algorithms. With the help of computer vision and machine learning, we can build systems that recognize hand gestures and use them to interact with digital systems.

This project demonstrates how to use MediaPipe's Gesture Recognizer to recognize hand gestures in real-time and visualize the hand landmarks on the video feed. This can be applied in various real-life scenarios, such as human-computer interaction, gaming, assistive technology, smart home automation, and more.

## Features

- **Real-Time Gesture Recognition**: Recognizes gestures from a live webcam feed.
- **Hand Landmark Visualization**: Displays hand landmarks on the video stream for visual feedback.
- **Potential Applications**: Can be integrated into various applications such as touchless interfaces, assistive devices, and more.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or later
- OpenCV
- MediaPipe

## Installation

Follow these steps to set up the project:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/fazalkadivar21/SignIX.git
   cd SignIX
   ```

2. **Install Dependencies**

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Gesture Recognition Model**

   Download the [gesture recognition model](https://mediapipe.dev/model/gesture_recognizer.task) and place it in the project directory. Update the `model_path` variable in the code with the absolute path to the model file.

## Usage

Run the following command to start the gesture recognition application:

```bash
python main.py
```

- **Webcam Feed**: The application will use your webcam to capture live video. Ensure your webcam is connected and accessible.
- **Gesture Recognition**: The system will recognize hand gestures and display the recognized gestures and hand landmarks on the video feed.
- **Exit**: Press the 'q' key to exit the application.

## Code Explanation

The following sections describe the key components of the code:

- **Imports and Setup**: Import necessary libraries and define the paths and options for the gesture recognizer model.
- **Callback Function**: The `print_result` function handles the output of the gesture recognition and draws landmarks on the video feed.
- **Video Capture**: Use OpenCV to capture video from the webcam.
- **Gesture Recognition**: Initialize the gesture recognizer with the provided options, and process each frame for gesture recognition.
- **Landmark Visualization**: Draw landmarks and connections on the frame using MediaPipe's drawing utilities.
- **Display Output**: Display the video feed with landmarks and recognized gestures in real-time.

## Potential Applications

Gesture recognition can be applied in various real-life scenarios:

- **Human-Computer Interaction**: Control devices and applications without physical contact.
- **Assistive Technology**: Assist people with disabilities by enabling gesture-based control.
- **Healthcare**: Monitor and analyze hand movements for rehabilitation exercises.
- **Gaming**: Create immersive experiences with gesture-based controls.
- **Smart Home Automation**: Control smart devices using gestures.
- **Education**: Develop interactive learning tools that use gesture recognition.

## Customization

To customize the gesture recognition system:

1. **Define New Gestures**: Update the model to recognize new gestures as needed for your application.
2. **Adjust Drawing Specifications**: Modify the colors and thickness used for drawing landmarks and connections.
3. **Integrate with Other Systems**: Incorporate gesture recognition into existing applications to enable new functionalities.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions, bug reports, or feature requests.
