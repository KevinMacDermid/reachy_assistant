import asyncio
import base64
import logging
import os
import time

import numpy as np
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openwakeword.model import Model
from reachy_mini import ReachyMini
from scipy.signal import resample

logger = logging.getLogger(__name__)

WAKE_MODEL_NAME = "hey_marvin"
WAKE_MODEL = Model(
        wakeword_model_paths=[os.path.join(os.path.dirname(__file__), "..", "models", f"{WAKE_MODEL_NAME}.onnx")]
    )
WAKE_THRESHOLD=0.3

OPENAI_SAMPLE_RATE = 24000  # OpenAI realtime conversations only support this sample rate
OPENAI_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe-2025-12-15"
OPENAI_VOICE = "marin"

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

        # Take a single channel
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


async def send_audio_openai(
        reachy: ReachyMini,
        stop_event: asyncio.Event,
        conn: AsyncRealtimeConnection):
    """Continuously poll mic and send to OpenAI."""
    mic_rate = reachy.media.get_input_audio_samplerate()
    try:
        while not stop_event.is_set():
            chunk = reachy.media.get_audio_sample()

            if chunk is None:
                await asyncio.sleep(0.1)
                continue

            # Take a single channel
            if chunk.ndim == 2:
                chunk = chunk[:, 0]

            # Resample to 24kHz if necessary
            if mic_rate != OPENAI_SAMPLE_RATE:
                num_samples = int(len(chunk) * OPENAI_SAMPLE_RATE / mic_rate)
                chunk = resample(chunk, num_samples)

            # Scale to int16
            if chunk.dtype != np.int16:
                # Reachy’s audio is float32 in [-1, 1]; scale and clip before casting
                chunk = np.clip(chunk, -1.0, 1.0)
                chunk = (chunk * np.iinfo(np.int16).max).astype(np.int16, copy=False)

            # Encode and put in buffer
            if np.all(chunk == 0):
                print("Transcribe chunk is all zeroes")

            base64_audio = base64.b64encode(chunk.tobytes()).decode("utf-8")
            await conn.input_audio_buffer.append(audio=base64_audio)

    except Exception as e:
        logger.error(f"Error sending audio: {e}")


