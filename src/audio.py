import asyncio
import base64
import logging
import os
import time

from websockets import ConnectionClosedOK

from tools.lights import LIGHTS_MCP
from tools.emotions import EMOTIONS_MCP

from enum import Enum

import numpy as np
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime import (RealtimeSessionCreateRequestParam,
                                   RealtimeAudioConfigParam,
                                   RealtimeAudioConfigInputParam,
                                   AudioTranscriptionParam,
                                   RealtimeAudioConfigOutputParam, RealtimeFunctionToolParam)
from openai.types.realtime.realtime_audio_formats_param import AudioPCM
from openai.types.realtime.realtime_audio_input_turn_detection_param import SemanticVad
from openwakeword.model import Model
from pedalboard import Pedalboard
from pedalboard_native import PitchShift
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


class ConversationMode(Enum):
    BEBOOP = "beboop"
    VOICE = "voice"


# Tools
# ToDo: move to separate file in tools
TOOLS = []
for tool in asyncio.run(LIGHTS_MCP.list_tools()):
    TOOLS.append(
        RealtimeFunctionToolParam(
            name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema,
            type="function",
        )
    )
for tool in asyncio.run(EMOTIONS_MCP.list_tools()):
    TOOLS.append(
        RealtimeFunctionToolParam(
            name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema,
            type="function",
        )
    )

TOOLS.append(
    RealtimeFunctionToolParam(
        name="end_conversation",
        description="Ends the current conversation session when the user dismisses the assistant, or puts it to sleep",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        type="function"
    )
)

TOOLS.append(
    RealtimeFunctionToolParam(
        name="change_conversation_mode",
        description="Switch between a robot mode (also called Beboop mode) and a voice mode.",
        parameters = {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": [m.value for m in ConversationMode],
                    "description": "Which conversation mode to switch to.",
                }
            },
            "required": ["mode"],
            "additionalProperties": False,
        },
        type = "function"
    )
)

def get_openai_session_config(mode: ConversationMode) -> RealtimeSessionCreateRequestParam:
    if mode == ConversationMode.BEBOOP:
        beboop_prompt = f"""
        You are a helpful little robot with just a head, no arms and legs. You're actually a 
        reachy-mini robot from HuggingFace with the name Marvin.

        You can't talk, so you will need to pick tools for displaying emotions or performing actions.
        You can provide a response providing a very brief explanation for your tool choice. Generally pick an emotion
        instead of a response every time.

        If you're dismissed, or asked to go to sleep, call the "end_conversation" tool.

        If there's a request about the conversation (other than ending it), then you should trigger change conversation
        mode to voice.
        """
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            output_modalities=["text"],
            model=OPENAI_TRANSCRIPTION_MODEL,
            instructions=beboop_prompt,
            tool_choice="auto",
            tools=TOOLS,
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    format=AudioPCM(
                        type="audio/pcm",
                        rate=24000
                    ),
                    transcription=AudioTranscriptionParam(
                        model=OPENAI_TRANSCRIPTION_MODEL,
                        language="en"),
                    turn_detection=SemanticVad(type="semantic_vad", eagerness="low")

                ),
            ),
        )
    elif mode == ConversationMode.VOICE:
        voice_prompt = f"""
        You are a helpful little robot with just a head, no arms and legs. You're actually a 
        reachy-mini robot from HuggingFace with the name Marvin. You can talk, as well 
        as express yourself using the emotion too. Try to use emotions pretty often, alongside your
        responses. 

        Generally act friendly, maybe in an almost naive way, as you can talk but you're still a cute
        little robot (don't explicitly point this out though). 

        You speak english unless specifically asked to do otherwise.

        If you're dismissed, or asked to go to sleep, call the "end_conversation" tool.
        If you are asked something about changing conversation modes, it's probably to switch to BEBOOP mode,
        unless you are clearly instructed otherwise.
        """
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            output_modalities=["audio"],
            model=OPENAI_TRANSCRIPTION_MODEL,
            instructions=voice_prompt,
            tool_choice="auto",
            tools=TOOLS,
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    format=AudioPCM(
                        type="audio/pcm",
                        rate=24000
                    ),
                    transcription=AudioTranscriptionParam(
                        model=OPENAI_TRANSCRIPTION_MODEL,
                        language="en"),
                    turn_detection=SemanticVad(type="semantic_vad", eagerness="low")
                ),
                output=RealtimeAudioConfigOutputParam(
                    format=AudioPCM(
                        type="audio/pcm",
                        rate=24000),
                    voice=OPENAI_VOICE
                )
            ),
        )
    else:
        raise ValueError(f"Invalid voice mode {str(mode)} received")


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

    except ConnectionClosedOK:
        logger.info("Closing input audio connection successful")   # suppresses an error log
    except Exception as e:
        logger.error(f"Error sending audio: {e}")


async def send_audio_speaker(
        reachy: ReachyMini,
        stop_event: asyncio.Event,
        speaker_queue: "asyncio.Queue[NDArray[np.int16]]",
):
    """Outputs the audio to the reachy speaker in chunks"""
    # Create a pitch shifter for cute voice effect
    speaker_rate = reachy.media.get_output_audio_samplerate()
    pitch_shifter = Pedalboard([PitchShift(semitones=5)])
    while True:
        if stop_event.is_set():
            return None
        output =  await speaker_queue.get()
        # Use None as an indication to stop listening (sentinel value)
        if output is None:
            return None

        last_activity_time = asyncio.get_event_loop().time()
        # Scale to float32 in range -1 to 1
        output = (output / np.iinfo(np.int16).max).astype(np.float32, copy=False)
        output = np.clip(output, -1.0, 1.0)

        # Apply pitch shift for cute effect (maintains state across chunks to avoid discontinuities)
        output = pitch_shifter(output.T, OPENAI_SAMPLE_RATE).T

        # Resample to 24kHz if necessary
        if speaker_rate != OPENAI_SAMPLE_RATE:
            num_samples = int(len(output) * speaker_rate / OPENAI_SAMPLE_RATE)
            output = resample(output, num_samples)
        reachy.media.push_audio_sample(output)


