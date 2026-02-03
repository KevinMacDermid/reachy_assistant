import time
import numpy as np
import os
from reachy_mini import ReachyMini
from openwakeword.model import Model
import logging

logger = logging.getLogger(__name__)

WAKE_MODEL_NAME = "hey_marvin"
WAKE_MODEL = Model(
        wakeword_model_paths=[os.path.join(os.path.dirname(__file__), "..", "models", f"{WAKE_MODEL_NAME}.onnx")]
    )
WAKE_THRESHOLD=0.3

# Exit code for supervisor to detect zero-audio failures.
ZERO_AUDIO_EXIT_CODE = 42
class ZeroAudioError(RuntimeError):
    pass


def listen_for_wakeword(reachy: ReachyMini) -> float:
    """
    Blocks until the wakeword is received.
    Returns:
        direction_of_arrival: The direction the sound came from
    """
    mic_rate = reachy.media.get_input_audio_samplerate()
    frames = 1280 * 8  # OpenWakeWord wants multiples of 1280 samples -> longer means more latency
    delay_s = frames / mic_rate
    time.sleep(delay_s)
    doa = 0

    logger.info(f"Starting wakeword listening using model {WAKE_MODEL_NAME}")
    while True:
        # Get audio
        chunk = reachy.media.get_audio_sample()
        if chunk is None:
            continue

        #doa = reachy.media.get_DoA()[0]

        # Take single channel
        if chunk.ndim == 2:
            chunk = chunk[:, 0]

        # Scale to int16
        if chunk.dtype != np.int16:
            # Reachy’s audio is float32 in [-1, 1]; scale and clip before casting
            chunk = np.clip(chunk, -1.0, 1.0)
            chunk = (chunk * np.iinfo(np.int16).max).astype(np.int16, copy=False)

        if np.all(chunk == 0):
            logger.error("All audio samples are zero going to wake word model!")
            raise ZeroAudioError("Zero audio samples detected")

        # OpenWakeWord model has buffer internally, so just send latest chunk
        pred = WAKE_MODEL.predict(chunk)[WAKE_MODEL_NAME]

        logger.debug(f"Wake prediction = {pred}")
        if pred >= WAKE_THRESHOLD:
            logger.info(f"Wake word detected with DoA {doa}")
            WAKE_MODEL.reset()  # reset the wake model's internal buffer
            break

        time.sleep(delay_s)

    logger.info("Wakeword Received!")
    #reachy.set_target_body_yaw(-1 * doa)
    return doa
