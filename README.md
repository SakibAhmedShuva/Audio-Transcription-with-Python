# Audio Transcription with Python 🎙️

A simple yet powerful audio transcription tool using Python and the Transformers library. This project utilizes the Whisper model from OpenAI to convert speech to text.

## 🚀 Features

- Easy-to-use audio transcription
- Support for various audio formats
- Built on OpenAI's Whisper model
- High accuracy transcription results
- Efficient processing using librosa

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.7+
- pip package manager

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/SakibAhmedShuva/Audio-Transcription-with-Python.git
```

2. Install the required dependencies:
```bash
pip install transformers
pip install librosa
pip install torch
```

## 💻 Usage

The main functionality is implemented in `transcription.ipynb`. Here's a basic example:

```python
from transformers import pipeline
import librosa

# Initialize the transcriber
transcriber = AudioTranscriber()

# Transcribe an audio file
text = transcriber.transcribe("your_audio_file.wav")
print(text)
```

## 📝 Example Code

```python
class AudioTranscriber:
    def __init__(self):
        self.transcriber = pipeline("automatic-speech-recognition", 
                                  model="openai/whisper-tiny")
    
    def transcribe(self, audio_path):
        # Load audio with standard sample rate
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Get transcription
        result = self.transcriber({"array": y, "sampling_rate": sr})
        
        return result["text"]
```

## ⚙️ Configuration

The current implementation uses the `whisper-tiny` model. For better accuracy, you can modify the model in the initialization:

```python
self.transcriber = pipeline("automatic-speech-recognition", 
                          model="openai/whisper-base")  # or other variants
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Sakib Ahmed Shuva**
- GitHub: [@SakibAhmedShuva](https://github.com/SakibAhmedShuva)

## 🙏 Acknowledgments

- OpenAI for the Whisper model
- Hugging Face Transformers team
- librosa developers

---
⭐️ If you found this project helpful, please star it on GitHub!
```
