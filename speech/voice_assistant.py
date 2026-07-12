from .whisper import WhisperTranscriber
from .tts import TextToSpeech
from .microphone import MicrophoneInput
import os

class VoiceAssistant:
    """Complete voice assistant with STT, LLM, and TTS."""

    def __init__(self, llm_client=None, whisper_model="base", tts_model="tts_models/en/ljspeech/tacotron2-DDC"):
        self.transcriber = WhisperTranscriber(model_size=whisper_model)
        self.tts = TextToSpeech(model_name=tts_model)
        self.microphone = MicrophoneInput()
        self.llm_client = llm_client
        self.conversation_history = []

    def listen(self, duration=5):
        """Listen to user input."""
        audio_path = self.microphone.record(duration)
        result = self.transcriber.transcribe(audio_path)
        return result["text"]

    def think(self, user_input, system_prompt="You are a helpful voice assistant."):
        """Process user input and generate response."""
        if self.llm_client:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            response = self.llm_client.chat(messages)
        else:
            response = f"Echo: {user_input}"

        self.conversation_history.append({
            "user": user_input,
            "assistant": response
        })

        return response

    def speak(self, text, output_path="response.wav"):
        """Convert response to speech."""
        return self.tts.synthesize(text, output_path)

    def run_interaction(self, duration=5):
        """Run a complete voice interaction."""
        print("Listening...")
        user_input = self.listen(duration)
        print(f"User said: {user_input}")

        print("Thinking...")
        response = self.think(user_input)
        print(f"Assistant: {response}")

        print("Speaking...")
        audio_path = self.speak(response)
        print(f"Response saved to: {audio_path}")

        return {
            "user_input": user_input,
            "response": response,
            "audio_path": audio_path
        }

    def continuous_mode(self):
        """Run in continuous listening mode."""
        print("Voice assistant started. Say 'exit' to stop.")

        while True:
            try:
                result = self.run_interaction()
                if "exit" in result["user_input"].lower():
                    print("Goodbye!")
                    break
            except Exception as e:
                print(f"Error: {e}")
                continue
