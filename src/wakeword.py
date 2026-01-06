import os

import numpy as np
import pyaudio
from openwakeword.model import Model

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
WAKE_MODEL_NAME = "hey_jarvis_v0.1"
WAKE_MODEL = Model(
   wakeword_model_paths=[os.path.join(os.path.dirname(__file__), "..", "models", f"{WAKE_MODEL_NAME}.onnx")]
)
THRESHOLD=0.5


def listen_for_wakeword() -> bool:
    """
    Blocks until the wakeword is received.
    """
    audio = pyaudio.PyAudio()
    mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    while True:
        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        # Only using a single model here, it supports having multiple
        pred = WAKE_MODEL.predict(audio)[WAKE_MODEL_NAME]

        if pred >= THRESHOLD:
            break

    return True