import os
import re
import traceback
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

class TextSummarizer:
    def __init__(self):
        # Extended stop words list
        self.stop_words = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
            'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
            'an', 'the', 'and', 'but', 'if', 'or', 'a', 'so', 'very', 'just', 'now', 'yeah',
            'well', 'still', 'like', 'let', 'can', 'will', 'would', 'should', 'could'
        ])

    def clean_text(self, text):
        """
        Clean and normalize the text
        - Remove special characters
        - Convert to lowercase
        - Remove extra whitespace
        """
        # Remove non-alphanumeric characters except spaces
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Convert to lowercase
        cleaned = cleaned.lower()
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def extract_meaningful_content(self, text, min_word_length=3):
        """
        Extract meaningful content by:
        1. Cleaning the text
        2. Removing stop words
        3. Focusing on longer, potentially more significant words
        """
        # Clean the text
        cleaned_text = self.clean_text(text)

        # Split into words
        words = cleaned_text.split()

        # Filter out stop words and short words
        meaningful_words = [
            word for word in words
            if word not in self.stop_words and len(word) >= min_word_length
        ]

        # Calculate word frequencies
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Sort words by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # Return top words or original text
        if sorted_words:
            return ' '.join([word for word, _ in sorted_words[:10]])

        return text

    def summarize(self, text):
        """
        Summarization method that works with fragmented text
        """
        try:
            # If text is very short, return as-is
            if len(text.split()) <= 5:
                return text

            # Extract meaningful content
            summary = self.extract_meaningful_content(text)

            return summary

        except Exception as e:
            print(f"Summarization error: {e}")
            traceback.print_exc()
            return text

# Flask Application
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize summarizer
summarizer = TextSummarizer()

# Initialize speech recognizer
recognizer = sr.Recognizer()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('summr.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Check if an audio file is included in the request
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']

    # Check if file is allowed
    if not allowed_file(audio_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save the uploaded file
    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)

    try:
        # Use SpeechRecognition to transcribe audio
        with sr.AudioFile(filepath) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)

        # Remove the uploaded file
        os.remove(filepath)

        return jsonify({'transcription': transcription})

    except sr.UnknownValueError:
        # Speech was unintelligible
        os.remove(filepath)
        return jsonify({'error': 'Could not understand audio'}), 400
    except sr.RequestError as e:
        # Could not request results from service
        os.remove(filepath)
        return jsonify({'error': f'Could not request results: {e}'}), 500
    except Exception as e:
        # Catch any other unexpected errors
        os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize_text():
    # Get text from request
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Perform summarization
    try:
        summary = summarizer.summarize(text)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)