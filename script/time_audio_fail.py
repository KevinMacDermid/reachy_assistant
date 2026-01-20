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
    t0 = time.time()
    fail_count = 0
    try:
        while fail_count < 5:
            sample_rate = reachy.media.get_input_audio_samplerate()
            print("Starting recording")
            reachy.media.start_recording()
            time.sleep(5)
            audio_chunk = reachy.media.get_audio_sample()
            print("Recording completed")

            if np.all(audio_chunk == 0):
                print("Recording samples are all 0")
                fail_count += 1
                print(f"Failure after {time.time() - t0} seconds")

    finally:
        del reachy  # stops client that's controlling motors