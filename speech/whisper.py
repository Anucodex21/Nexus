import whisper
import torch
import os

class WhisperTranscriber:
    """Speech-to-text using OpenAI Whisper."""

    def __init__(self, model_size="base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_size).to(self.device)
        self.model_size = model_size

    def transcribe(self, audio_path, language=None, task="transcribe"):
        """Transcribe audio file to text."""
        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            fp16=torch.cuda.is_available()
        )
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": result.get("language", "unknown")
        }

    def transcribe_batch(self, audio_paths, language=None):
        """Transcribe multiple audio files."""
        results = []
        for path in audio_paths:
            result = self.transcribe(path, language)
            results.append(result)
        return results

    def detect_language(self, audio_path):
        """Detect language of audio."""
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)

        return {
            "language": detected_lang,
            "confidence": probs[detected_lang]
        }

    def translate(self, audio_path, target_language="en"):
        """Translate audio to target language."""
        return self.transcribe(audio_path, task="translate")
