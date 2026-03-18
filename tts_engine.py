import pyttsx3


class TextToSpeechEngine:
    """Handles all text-to-speech operations"""

    def __init__(self):
        self.python_tts = pyttsx3.init()
        self.setup_voices()

    def setup_voices(self):
        """Setup text-to-speech voices"""
        voices = self.python_tts.getProperty('voices')
        self.python_tts.setProperty('voice', voices[1].id)  # 1 for female

    def python_tts_speak(self, text):
        """Perform Python Text-to-Speech"""
        self.python_tts.say(text)
        self.python_tts.runAndWait()

    def speak(self, text, tts_type="Python Text-to-Speech (OFFLINE)"):
        if "Google" in tts_type:
            self.google_tts(text)
        elif "Python" in tts_type:
            self.python_tts_speak(text)
        else:
            print("Defaulting to offline speech.")
            self.python_tts_speak(text)