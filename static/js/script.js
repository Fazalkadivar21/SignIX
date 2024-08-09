// Connect to the server using SocketIO
const socket = io();

// Function to display gesture results
function displayResults(data) {
    // Display the video frame
    const videoElement = document.getElementById('video');
    videoElement.src = 'data:image/jpeg;base64,' + data.image;

    // Display the gesture results
    const resultsElement = document.getElementById('results');
    let resultsHTML = '';
    data.results.forEach((result, index) => {
        resultsHTML += `<p>Hand ${index + 1}: ${result.handedness}</p>`;
        result.gestures.forEach(gesture => {
            resultsHTML += `<p>Gesture: ${gesture.gesture} (Score: ${gesture.score.toFixed(2)})</p>`;
        });
    });
    resultsElement.innerHTML = resultsHTML;
}

// Listen for gesture results from the server
socket.on('gesture_result', displayResults);
