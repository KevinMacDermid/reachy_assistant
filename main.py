from reachy_mini.motion.recorded_move import RecordedMoves
from dotenv import load_dotenv
import asyncio
import base64
import os
import numpy as np
from openai import AsyncOpenAI
from openai.types.realtime import (
    RealtimeSessionCreateRequestParam,
     RealtimeAudioConfigParam,
    RealtimeAudioConfigInputParam,
    AudioTranscriptionParam
)
from openai.types.realtime.realtime_audio_formats_param import AudioPCM
from openai.types.realtime.realtime_audio_input_turn_detection_param import SemanticVad, ServerVad
from reachy_mini import ReachyMini
from scipy.signal import resample
from openwakeword.model import Model
import logging
import time


logger = logging.getLogger(__name__)
EMOTION_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
DANCE_MOVES = RecordedMoves("pollen-robotics/reachy-mini-dances-library")
TARGET_SAMPLE_RATE = 24000
_ = load_dotenv()

#WAKE_MODEL_NAME = "hey_jarvis_v0.1"
WAKE_MODEL_NAME = "hey_marvin"
WAKE_MODEL = Model(
   wakeword_model_paths=[os.path.join(os.path.dirname(__file__), "models", f"{WAKE_MODEL_NAME}.onnx")]
)
WAKE_THRESHOLD=0.5


def listen_for_wakeword(reachy) -> bool:
    """
    Blocks until the wakeword is received.
    """
    mic_rate = reachy.media.get_input_audio_samplerate()
    reachy.media.start_recording()
    _ = reachy.media.get_audio_sample()   # sometimes it seems to still have the previous

    frames = 1280 * 5  # OpenWakeWord wants multiples of 1280 samples
    delay_s = frames / mic_rate
    time.sleep(delay_s)

    while True:
        # Get audio
        chunk = reachy.media.get_audio_sample()
        if chunk is None:
            continue

        # Take single channel
        if chunk.ndim == 2:
            chunk = chunk[:, 0]

        # Scale to int16
        if chunk.dtype != np.int16:
            # Reachy’s audio is float32 in [-1, 1]; scale and clip before casting
            chunk = np.clip(chunk, -1.0, 1.0)
            chunk = (chunk * np.iinfo(np.int16).max).astype(np.int16, copy=False)

        if np.all(chunk == 0):
            logger.warning("All audio samples are zero going to wake word model!")

        # OpenWakeWord model has buffer internally, so just send latest chunk
        pred = WAKE_MODEL.predict(chunk)[WAKE_MODEL_NAME]

        print(pred)
        if pred >= WAKE_THRESHOLD:
            WAKE_MODEL.reset()  # reset the wake model's internal buffer
            break

        time.sleep(delay_s)

    reachy.media.stop_recording()
    return True


def main():
    """
    Main conversation App
    """
    while True:
        # Tried starting the daemon here, but it wouldn't work, start it separately
        with ReachyMini() as reachy:
            reachy.goto_sleep()
            # ToDo: should force this to use the robot's audio channel
            listen_for_wakeword(reachy)
            print("Wakeword Received!")
            reachy.wake_up()
            time.sleep(0.5)


async def transcribe_audio():
    # 1. Initialize Robot
    robot = ReachyMini()
    robot.media.start_recording()
    mic_rate = robot.media.get_input_audio_samplerate()

    # 2. Initialize OpenAI Client
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    logger.info("Connecting to OpenAI...")
    async with client.realtime.connect(model="gpt-4o-realtime-preview") as conn:
        # Configure session for transcription

        session_config = RealtimeSessionCreateRequestParam(
            type="realtime",
            output_modalities=["text"],
            model="gpt-4o-transcribe",
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    format=AudioPCM(
                        type = "audio/pcm",
                        rate = 24000
                    ),
                    transcription=AudioTranscriptionParam(model="gpt-4o-transcribe"),
                    turn_detection=ServerVad(type="server_vad")

                ),
            ),
        )
        await conn.session.update(session=session_config)

        async def send_audio():
            """Continuously poll mic and send to OpenAI."""
            try:
                samples_to_commit = 0
                while True:
                    audio_chunk = robot.media.get_audio_sample()
                    if audio_chunk is None:
                        await asyncio.sleep(0.01)
                        continue

                    # Take single channel
                    if audio_chunk.ndim == 2:
                        audio_chunk = audio_chunk[:, 0]

                    # Resample to 24kHz if necessary
                    if mic_rate != TARGET_SAMPLE_RATE:
                        num_samples = int(len(audio_chunk) * TARGET_SAMPLE_RATE / mic_rate)
                        audio_chunk = resample(audio_chunk, num_samples).astype(np.int16)
                    else:
                        audio_chunk = audio_chunk.astype(np.int16, copy=False)

                    # Encode and put in buffer
                    if not np.all(audio_chunk == 0):
                        print("got sample with some data in it")
                    base64_audio = base64.b64encode(audio_chunk.tobytes()).decode("utf-8")
                    event = await conn.input_audio_buffer.append(audio=base64_audio)

                    # Commit (shouldn't do given that we have ServerVad
                    #samples_to_commit += audio_chunk.size
                    #if samples_to_commit >= 0.5 * TARGET_SAMPLE_RATE:
                    #    await conn.input_audio_buffer.commit()
                    #    samples_to_commit = 0

            except Exception as e:
                logger.error(f"Error sending audio: {e}")

        async def receive_events():
            """Listen for transcription events."""
            async for event in conn:
                print(event)
                # Partial transcription (real-time)
                if event.type == "conversation.item.input_audio_transcription.partial":
                    print(f"Partial: {event.transcript}", end="\r")

                # Final transcription
                elif event.type == "conversation.item.input_audio_transcription.completed":
                    print(f"\nFinal: {event.transcript}")

                elif event.type == "error":
                    print(f"\nError: {str(event.error)}")

        logger.info("Listening... (Ctrl+C to stop)")
        try:
            # Run both loops concurrently
            await asyncio.gather(send_audio(), receive_events())
        except asyncio.CancelledError:
            pass
        finally:
            robot.media.stop_recording()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
    #asyncio.run(transcribe_audio())
    #while True:
    #    robot = ReachyMini()
    #    robot.start_recording()
    #    chunk = robot.media.get_audio_sample()
    #    if chunk is not None and not np.all(chunk == 0.0):
    #        print("Got a sample")

    #    time.sleep(0.5)

