import json
from reachy_mini.motion.recorded_move import RecordedMoves
from dotenv import load_dotenv
import asyncio
import base64
import os
import numpy as np
import sys
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
from tools.lights import LIGHTS_MCP, LightState, set_light_state
from tools.emotions import EMOTIONS_MCP
from enum import Enum

_ = load_dotenv()
logger = logging.getLogger(__name__)

# Exit code for supervisor to detect zero-audio failures.
ZERO_AUDIO_EXIT_CODE = 42
class ZeroAudioError(RuntimeError):
    pass


# Wake model
TARGET_SAMPLE_RATE = 24000
WAKE_MODEL_NAME = "hey_marvin"
WAKE_MODEL = Model(
        wakeword_model_paths=[os.path.join(os.path.dirname(__file__), "models", f"{WAKE_MODEL_NAME}.onnx")]
    )
WAKE_THRESHOLD=0.3

class ConversationMode(Enum):
    BEBOOP = "beboop"
    VOICE = "voice"

# Silent Robot
EMOTION_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
BEBOOP_PROMPT = f"""
You are a helpful little robot with just a head, no arms and legs. You're actually a 
reachy-mini robot from HuggingFace with the name Marvin.

You can't talk, so you will need to pick tools for displaying emotions or performing actions.
You can provide a response providing a very brief explanation for your tool choice. Generally pick an emotion
instead of a response every time.

If you're dismissed, or asked to go to sleep, call the "end_conversation" tool.
"""
MOVE_GOTO_DURATION = 0.5
USER_IDLE_TIMEOUT = 12.0

# Tools
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


def _listen_for_wakeword(reachy: ReachyMini) -> float:
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


def _log_task_result(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning("Move task failed: %s", e)


def _get_session_config(mode: ConversationMode) -> RealtimeSessionCreateRequestParam:
    """
    Gets the OpenAI session configuration
    """
    if mode == ConversationMode.BEBOOP:
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            output_modalities=["text"],
            model="gpt-4o-transcribe",
            instructions=BEBOOP_PROMPT,
            tool_choice="auto",
            tools=TOOLS,
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    format=AudioPCM(
                        type="audio/pcm",
                        rate=24000
                    ),
                    transcription=AudioTranscriptionParam(
                        model="gpt-4o-transcribe",
                        language="en"),
                    turn_detection=ServerVad(type="server_vad")

                ),
            ),
        )
    elif mode == ConversationMode.VOICE:
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            output_modalities=["text"],
            model="gpt-4o-transcribe",
            instructions=BEBOOP_PROMPT,
            tool_choice="auto",
            tools=TOOLS,
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    format=AudioPCM(
                        type="audio/pcm",
                        rate=24000
                    ),
                    transcription=AudioTranscriptionParam(
                        model="gpt-4o-transcribe",
                        language="en"),
                    turn_detection=ServerVad(type="server_vad")

                ),
            ),
        )
    else:
        raise ValueError(f"Invalid voice mode {str(mode)} received")


async def run_conversation(
        reachy: ReachyMini,
        client: AsyncOpenAI,
        mode: ConversationMode) -> ConversationMode | None:
    """
    Main conversation coroutine, receives voice in real time and generates
    either robot responses or voice output.

    Args:
        reachy: The robot instance
        client: Asynchronous OpenAI client for realtime conversations
        mode: The conversation mode to use

    Returns:
        mode_to_set: If provided, then conversation should be restarted in
            the associated mode.
    """
    mic_rate = reachy.media.get_input_audio_samplerate()
    stop_event = asyncio.Event()

    last_user_activity_time = asyncio.get_event_loop().time()

    logger.info(f"Starting new conversation in {str(mode)}")
    async with client.realtime.connect(model="gpt-4o-realtime-preview") as conn:
        # Configure session for realtime conversation
        session_config = _get_session_config(mode)
        await conn.session.update(session=session_config)

        async def idle_watchdog():
            """Stop the conversation if no user activity occurs for a while."""
            nonlocal last_user_activity_time
            try:
                while not stop_event.is_set():
                    await asyncio.sleep(0.25)
                    now = asyncio.get_event_loop().time()
                    if (now - last_user_activity_time) >= USER_IDLE_TIMEOUT:
                        logger.info(
                            "User idle for %.1fs (>= %.1fs). Timing out conversation.",
                            (now - last_user_activity_time),
                            USER_IDLE_TIMEOUT,
                        )
                        stop_event.set()
                        try:
                            await conn.close()
                        except Exception:
                            pass
                        return
            except asyncio.CancelledError:
                raise

        async def send_audio():
            """Continuously poll mic and send to OpenAI."""
            try:
                while not stop_event.is_set():
                    chunk = reachy.media.get_audio_sample()
                    if chunk is None:
                        await asyncio.sleep(0.1)
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
            nonlocal last_user_activity_time
            global CONV_MODE
            async for event in conn:
                if event.type == "conversation.item.input_audio_transcription.partial":
                    last_user_activity_time = asyncio.get_event_loop().time()
                    logger.debug(f"Partial: {event.transcript}")

                elif event.type == "conversation.item.input_audio_transcription.completed":
                    last_user_activity_time = asyncio.get_event_loop().time()
                    logger.info(f"\nUser: {event.transcript}")

                elif event.type == "error":
                    logger.error(f"\nError: {str(event.error)}")

                elif event.type == "response.output_text.done":
                    logger.info(f"Model: {event.text}")

                elif event.type == "response.done":
                    last_user_activity_time = asyncio.get_event_loop().time()
                    if len(event.response.output) > 0:
                        output = event.response.output[0]
                        if output.type == "function_call":
                            tool_name = output.name
                            if tool_name == "end_conversation":
                                logger.info("Tool call: end_conversation → stopping conversation loop.")
                                stop_event.set()
                                await conn.close()
                                return
                            elif tool_name == "change_conversation_mode":
                                new_mode = ConversationMode(json.loads(output.arguments)["mode"])
                                logger.info(f"Tool call: change_conversation_mode to {str(new_mode)}")
                                if new_mode != mode:
                                    logger.info(f"New Mode {str(new_mode)} differs from {str(mode)}, switching")
                                    # Finish conversation to avoid issues with partial responses
                                    stop_event.set()
                                    await conn.close()
                                    return new_mode
                            elif tool_name == "set_light_state":
                                logger.info("Tool call: set_light_state state received")
                                args = json.loads(output.arguments or "{}")
                                light_state = LightState(**args["light_state"])
                                # set_light_state uses subprocess (blocking), so run it in a thread
                                result_text = await asyncio.to_thread(set_light_state, light_state)
                                if not "Error setting light state" in result_text:
                                    logger.info(f"Tool call: set_light_state → {result_text}")
                                    await reachy.async_play_move(EMOTION_MOVES.get("success1"),
                                                                 initial_goto_duration=MOVE_GOTO_DURATION)
                                    # Generally stop after first light command
                                    stop_event.set()
                                    await conn.close()
                                    return None
                            elif tool_name == "show_emotion":
                                logger.info(f"Tool call: show_emotion for {output.arguments}")
                                try:
                                    move_name = json.loads(output.arguments)["move_name"]
                                    # Create task here to avoid blocking
                                    move_task = asyncio.create_task(
                                        reachy.async_play_move(EMOTION_MOVES.get(move_name),
                                                               initial_goto_duration=MOVE_GOTO_DURATION)
                                    )
                                    move_task.add_done_callback(_log_task_result)
                                except Exception:
                                    logger.error("Move {output.arguments} failed to play")
                                    await reachy.async_play_move(EMOTION_MOVES.get("confused1"),
                                                                 initial_goto_duration=MOVE_GOTO_DURATION)
                            else:
                                logger.error(f"Unrecognized tool received. Name = {tool_name}")

        logger.info("Conversation Started... (Ctrl+C to stop)")
        watchdog_task = asyncio.create_task(idle_watchdog())
        new_mode = None
        try:
            # Run both loops concurrently
            _, new_mode, _ = await asyncio.gather(send_audio(), receive_events(), watchdog_task)
        finally:
            stop_event.set()
            watchdog_task.cancel()
            if new_mode is None:
                reachy.goto_sleep()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass
        return new_mode


def main():
    """
    Main conversation App
    """
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    conv_mode = ConversationMode.BEBOOP
    skip_wakeword = False
    # Tried starting the daemon here, but it wouldn't work, start it separately
    with ReachyMini(automatic_body_yaw=True) as reachy:
        reachy.goto_sleep()
        while True:
            reachy.media.start_recording()
            try:
                if not skip_wakeword:
                    _ = _listen_for_wakeword(reachy)
                    reachy.wake_up()
                else:
                    skip_wakeword = False

                new_mode = asyncio.run(run_conversation(reachy, client, conv_mode))
                if isinstance(new_mode, ConversationMode):
                    conv_mode = new_mode
                    skip_wakeword = True


            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down")
                break
            except ZeroAudioError:
                logger.error("Exiting due to zero audio samples.")
                sys.exit(ZERO_AUDIO_EXIT_CODE)
            finally:
                reachy.stop_recording()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
