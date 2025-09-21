const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snap = document.getElementById('snap');
const result = document.getElementById('result');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
})
.catch(err => console.error("Webcam error:", err));

// Capture frame and send to server
snap.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');

        fetch('/verify', { method: 'POST', body: formData })
        .then(response => response.text())
        .then(text => result.textContent = text)
        .catch(err => console.error(err));
    }, 'image/jpeg');
});
