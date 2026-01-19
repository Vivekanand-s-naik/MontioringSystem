document.addEventListener('DOMContentLoaded', () => {
    const startRecordingBtn = document.getElementById('startRecording');
    const stopRecordingBtn = document.getElementById('stopRecording');
    const transcriptionArea = document.getElementById('transcriptionArea');
    const summaryArea = document.getElementById('summaryArea');
    const errorMessage = document.getElementById('errorMessage');

    let mediaRecorder;
    let audioChunks = [];

    startRecordingBtn.addEventListener('click', startRecording);
    stopRecordingBtn.addEventListener('click', stopRecording);

    async function startRecording() {
        try {
            // Reset previous state
            audioChunks = [];
            transcriptionArea.value = '';
            summaryArea.value = '';
            errorMessage.textContent = '';

            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Create MediaRecorder
            mediaRecorder = new MediaRecorder(stream);

            // Event listeners for recording
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = sendAudioToServer;

            // Start recording
            mediaRecorder.start();

            // Update button states
            startRecordingBtn.disabled = true;
            stopRecordingBtn.disabled = false;

        } catch (error) {
            console.error('Error accessing microphone:', error);
            errorMessage.textContent = 'Could not access microphone. Please check permissions.';
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();

            // Update button states
            startRecordingBtn.disabled = false;
            stopRecordingBtn.disabled = true;
        }
    }

    async function sendAudioToServer() {
        // Create blob from recorded audio chunks
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });

        // Create form data
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        try {
            // Send to Flask backend
            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                // Handle server-side errors
                errorMessage.textContent = data.error;
            } else {
                // Update transcription and summary
                transcriptionArea.value = data.transcription;
                summaryArea.value = data.summary;
            }

        } catch (error) {
            console.error('Transcription error:', error);
            errorMessage.textContent = 'Failed to transcribe audio. Please try again.';
        }
    }
});