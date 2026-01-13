import json

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
    AudioTranscriptionParam,
    RealtimeToolsConfigParam,
    RealtimeFunctionToolParam
)
from openai.types.realtime.realtime_audio_formats_param import AudioPCM
from openai.types.realtime.realtime_audio_input_turn_detection_param import SemanticVad, ServerVad
from reachy_mini import ReachyMini
from scipy.signal import resample
from openwakeword.model import Model
import logging
import time
from src.mcp_light_server import LIGHTS_MCP, LightState, set_light_state

_ = load_dotenv()
logger = logging.getLogger(__name__)

# Wake model
TARGET_SAMPLE_RATE = 24000
WAKE_MODEL_NAME = "hey_marvin"
WAKE_MODEL = Model(
        wakeword_model_paths=[os.path.join(os.path.dirname(__file__), "models", f"{WAKE_MODEL_NAME}.onnx")]
    )
WAKE_THRESHOLD=0.5

# Silent Robot
#DANCE_MOVES = RecordedMoves("pollen-robotics/reachy-mini-dances-library")
EMOTION_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
BEBOOP_PROMPT = f"""
You are a helpful little robot with just a head, no arms and legs. You're actually a 
reachy-mini robot from HuggingFace with the name Marvin. You can only respond with 
specific emotion move from the following list:
 {str({name: EMOTION_MOVES.get(name).description for name in EMOTION_MOVES.list_moves()})}

Answer with just the name of the move that outlines the type of response you'd give, like
if you were confused you might reply with "uncertain1".

If you're dismissed, or asked to go to sleep, call the "end_conversation" tool.
"""

# Lighting Tools

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


def listen_for_wakeword(reachy: ReachyMini) -> float:
    """
    Blocks until the wakeword is received.
    Returns:
        direction_of_arrival: The direction the sound came from
    """
    mic_rate = reachy.media.get_input_audio_samplerate()
    frames = 1280 * 5  # OpenWakeWord wants multiples of 1280 samples
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
            logger.warning("All audio samples are zero going to wake word model!")

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


def main():
    """
    Main conversation App
    """
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # Tried starting the daemon here, but it wouldn't work, start it separately
    with ReachyMini(automatic_body_yaw=True) as reachy:
        while True:
            reachy.media.start_recording()
            try:
                reachy.goto_sleep()
                _ = listen_for_wakeword(reachy)
                reachy.wake_up()
                # Main conversation
                asyncio.run(main_conversation(reachy, client))
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down")
                break
            finally:
                reachy.stop_recording()


async def main_conversation(
        reachy: ReachyMini,
        client: AsyncOpenAI):
    """
    Main conversation coroutine, receives voice in real time and generates
    either robot responses or voice output.

    Args:
        reachy: The robot instance
        client: Asynchronous OpenAI client for realtime conversations
    """
    mic_rate = reachy.media.get_input_audio_samplerate()
    stop_event = asyncio.Event()

    logger.info("Connecting to OpenAI...")
    async with client.realtime.connect(model="gpt-4o-realtime-preview") as conn:
        # Configure session for realtime conversation
        session_config = RealtimeSessionCreateRequestParam(
            type="realtime",
            output_modalities=["text"],
            model="gpt-4o-transcribe",
            instructions=BEBOOP_PROMPT,
            tool_choice="auto",
            tools=TOOLS,
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
                while not stop_event.set():
                    chunk = reachy.media.get_audio_sample()
                    if chunk is None:
                        await asyncio.sleep(0.01)
                        continue

                    # Take single channel
                    if chunk.ndim == 2:
                        chunk = chunk[:, 0]

                    # Resample to 24kHz if necessary
                    if mic_rate != TARGET_SAMPLE_RATE:
                        num_samples = int(len(chunk) * TARGET_SAMPLE_RATE / mic_rate)
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

        async def receive_events():
            """Listen for transcription events."""
            async for event in conn:
                if event.type == "conversation.item.input_audio_transcription.partial":
                    logger.debug(f"Partial: {event.transcript}")

                elif event.type == "conversation.item.input_audio_transcription.completed":
                    logger.info(f"\nUser: {event.transcript}")

                elif event.type == "error":
                    logger.error(f"\nError: {str(event.error)}")

                elif event.type == "response.output_text.done":
                    logger.info(f"Model: {event.text}")
                    try:
                        await reachy.async_play_move(EMOTION_MOVES.get(event.text))
                    except ValueError:
                        logger.error("Move {event.text} was not in the emotion moves list")
                        await reachy.async_play_move(EMOTION_MOVES.get("confused1"))

                elif event.type == "response.done":
                    if len(event.response.output) > 0:
                        output = event.response.output[0]
                        if output.type == "function_call":
                            tool_name = output.name
                            if tool_name == "end_conversation":
                                logger.info("Tool call: end_conversation → stopping conversation loop.")
                                stop_event.set()  # or cancel tasks / break out as appropriate
                                raise asyncio.CancelledError()
                            elif tool_name == "set_light_state":
                                # Currently just calling the light directly, may remove MCP
                                logger.info("Light set state received")
                                args = json.loads(output.arguments or "{}")
                                light_state = LightState(**args["light_state"])
                                # set_light_state uses subprocess (blocking), so run it in a thread
                                result_text = await asyncio.to_thread(set_light_state, light_state)
                                if not "Error setting light state" in result_text:
                                    logger.info(f"Tool call: set_light_state → {result_text}")
                                    await reachy.async_play_move(EMOTION_MOVES.get("success1"))
                                    # Generally stop after first light command
                                    #stop_event.set()
                                    #raise asyncio.CancelledError()
                            else:
                                logger.error(f"Unrecognized tool received. Name = {tool_name}")

        logger.info("Conversation Started... (Ctrl+C to stop)")
        try:
            # Run both loops concurrently
            await asyncio.gather(send_audio(), receive_events())
        except asyncio.CancelledError:
            pass
        finally:
            reachy.media.stop_recording()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

