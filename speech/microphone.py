import speech_recognition as sr
import pyaudio
import wave
import tempfile
import os

class MicrophoneInput:
    """Handle microphone input for speech recognition."""

    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.recognizer = sr.Recognizer()
        self.audio = None

    def record(self, duration=5, save_path=None):
        """Record audio from microphone."""
        print(f"Recording for {duration} seconds...")

        with sr.Microphone(sample_rate=self.sample_rate) as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.audio = self.recognizer.listen(source, timeout=duration)

        if save_path:
            with open(save_path, "wb") as f:
                f.write(self.audio.get_wav_data())
            return save_path

        # Save to temp file
        temp_path = tempfile.mktemp(suffix=".wav")
        with open(temp_path, "wb") as f:
            f.write(self.audio.get_wav_data())

        return temp_path

    def record_until_stop(self, save_path="recording.wav"):
        """Record until user stops (Ctrl+C)."""
        print("Recording... Press Ctrl+C to stop.")

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        frames = []
        try:
            while True:
                data = stream.read(self.chunk_size)
                frames.append(data)
        except KeyboardInterrupt:
            print("Recording stopped.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save recording
        with wave.open(save_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))

        return save_path

    def transcribe_live(self, duration=5):
        """Record and transcribe in one step."""
        audio_path = self.record(duration)

        with sr.AudioFile(audio_path) as source:
            audio = self.recognizer.record(source)

        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Speech recognition service error"

    def calibrate(self, duration=2):
        """Calibrate for ambient noise."""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
        print("Microphone calibrated for ambient noise.")
