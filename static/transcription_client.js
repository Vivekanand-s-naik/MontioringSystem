class SpeechTranscriptionClient {
    constructor(serverUrl = 'ws://localhost:8765') {
        this.serverUrl = serverUrl;
        this.socket = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
    }

    connect() {
        try {
            this.socket = new WebSocket(this.serverUrl);

            this.socket.onopen = () => {
                console.log('WebSocket connection established');
                this.updateStatus('Connected', 'green');
            };

            this.socket.onmessage = this.handleServerMessage.bind(this);

            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateStatus('Connection Error', 'red');
            };

            this.socket.onclose = () => {
                console.log('WebSocket connection closed');
                this.updateStatus('Disconnected', 'red');
            };
        } catch (error) {
            console.error('WebSocket connection error:', error);
        }
    }

    async startRecording() {
        if (this.isRecording) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm',
                audioBitsPerSecond: 128000  // Higher quality audio
            });

            this.mediaRecorder.ondataavailable = this.handleDataAvailable.bind(this);
            this.mediaRecorder.start(1000);  // Collect data every second

            this.isRecording = true;
            this.updateStatus('Recording', 'red');
        } catch (error) {
            console.error('Microphone access error:', error);
            this.updateStatus('Microphone Error', 'red');
        }
    }

    stopRecording() {
        if (!this.isRecording) return;

        this.mediaRecorder.stop();
        this.isRecording = false;
        this.updateStatus('Stopped', 'orange');
    }

    handleDataAvailable(event) {
        if (event.data.size > 0) {
            this.audioChunks.push(event.data);
            this.sendAudioToServer(event.data);
        }
    }

    sendAudioToServer(audioBlob) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            const reader = new FileReader();
            reader.onloadend = () => {
                try {
                    this.socket.send(reader.result);
                } catch (error) {
                    console.error('Error sending audio:', error);
                }
            };
            reader.readAsArrayBuffer(audioBlob);
        }
    }

    handleServerMessage(event) {
        try {
            const data = JSON.parse(event.data);

            if (data.transcription) {
                this.updateTranscription(data.transcription);
                this.updateSummary(data.summary);
            } else if (data.error) {
                this.updateStatus(`Error: ${data.error}`, 'red');
            }
        } catch (error) {
            console.error('Message parsing error:', error);
        }
    }

    updateTranscription(text) {
        const transcriptionElement = document.getElementById('transcription');
        if (transcriptionElement) {
            transcriptionElement.textContent = text;
        }
    }

    updateSummary(text) {
        const summaryElement = document.getElementById('summary');
        if (summaryElement) {
            summaryElement.textContent = text;
        }
    }

    updateStatus(message, color = 'black') {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.style.color = color;
        }
    }
}

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    const client = new SpeechTranscriptionClient();
    client.connect();

    document.getElementById('startRecording').addEventListener('click', () => client.startRecording());
    document.getElementById('stopRecording').addEventListener('click', () => client.stopRecording());
});