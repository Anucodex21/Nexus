from TTS.api import TTS
import torch
import os

class TextToSpeech:
    """Text-to-speech synthesis."""

    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name).to(self.device)
        self.model_name = model_name

    def synthesize(self, text, output_path="output.wav", speaker=None):
        """Synthesize text to speech."""
        if speaker:
            self.tts.tts_to_file(text=text, speaker=speaker, file_path=output_path)
        else:
            self.tts.tts_to_file(text=text, file_path=output_path)

        return output_path

    def synthesize_batch(self, texts, output_dir="./outputs"):
        """Synthesize multiple texts."""
        os.makedirs(output_dir, exist_ok=True)
        paths = []

        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"output_{i}.wav")
            path = self.synthesize(text, output_path)
            paths.append(path)

        return paths

    def list_models(self):
        """List available TTS models."""
        return TTS.list_models()

    def change_model(self, model_name):
        """Change the TTS model."""
        self.tts = TTS(model_name).to(self.device)
        self.model_name = model_name
