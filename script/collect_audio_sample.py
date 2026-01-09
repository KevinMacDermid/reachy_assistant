"""
Small script to grab audio from the robot, for debugging
"""
from reachy_mini import ReachyMini
import time
from scipy.io import wavfile
import numpy as np
import logging


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    reachy = ReachyMini()
    try:
        sample_rate = reachy.media.get_input_audio_samplerate()
        print("Starting recording")
        reachy.media.start_recording()
        time.sleep(5)
        audio_chunk = reachy.media.get_audio_sample()
        print("Recording completed")

        if np.all(audio_chunk == 0):
            print("Recording samples are all 0")

        # Take single channel
        if audio_chunk.ndim == 2:
            audio_chunk = audio_chunk[:, 0]

        #audio_chunk = audio_chunk.astype(np.int16, copy=False)
        wavfile.write("sample.wav", sample_rate, audio_chunk)
    finally:
        del reachy  # stops client that's controlling motors