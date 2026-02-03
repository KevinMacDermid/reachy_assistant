import asyncio
import base64
import json
import logging
import os
import sys
from enum import Enum

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.realtime import (
    RealtimeFunctionToolParam,
    RealtimeResponseCreateParamsParam
)
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves

from src.audio import (
    listen_for_wakeword,
    ZeroAudioError,
    ZERO_AUDIO_EXIT_CODE,
    send_audio_openai, send_audio_speaker, get_openai_session_config
)
from tools.emotions import EMOTIONS_MCP
from tools.lights import LIGHTS_MCP, LightState, set_light_state

_ = load_dotenv()
logger = logging.getLogger(__name__)


class ConversationMode(Enum):
    BEBOOP = "beboop"
    VOICE = "voice"

# Silent Robot
EMOTION_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
MOVE_GOTO_DURATION = 0.5
USER_IDLE_TIMEOUT = 30.0

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


def _log_task_result(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning("Move task failed: %s", e)


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
    # Can move queue to send_audio_speaker once asyncio pattern cleaned up
    speaker_queue: "asyncio.Queue[NDArray[np.int16]]" = asyncio.Queue()
    stop_event = asyncio.Event()
    last_activity_time = asyncio.get_event_loop().time()

    logger.info(f"Starting new conversation in {str(mode)}")
    async with client.realtime.connect(model="gpt-4o-realtime-preview") as conn:
        # Configure session for realtime conversation
        session_config = get_openai_session_config(mode)
        await conn.session.update(session=session_config)
        
        # In voice mode want to do a quick greeting
        if mode == ConversationMode.VOICE:
            await conn.response.create(
                response=RealtimeResponseCreateParamsParam(
                    instructions="Start the conversation with a short friendly greeting in english"
                )
            )

        async def idle_watchdog():
            """Stop the conversation if no user activity occurs for a while."""
            nonlocal last_activity_time
            try:
                while not stop_event.is_set():
                    await asyncio.sleep(0.25)
                    now = asyncio.get_event_loop().time()
                    if (now - last_activity_time) >= USER_IDLE_TIMEOUT:
                        logger.info(
                            "User idle for %.1fs (>= %.1fs). Timing out conversation.",
                            (now - last_activity_time),
                            USER_IDLE_TIMEOUT,
                        )
                        speaker_queue.put_nowait(None)
                        stop_event.set()
                        try:
                            await conn.close()
                        except Exception:
                            pass
                        return
            except asyncio.CancelledError:
                raise

        async def receive_events():
            """Listen for transcription events."""
            nonlocal last_activity_time
            async for event in conn:
                # Any event stops the watchdog from triggering
                last_activity_time = asyncio.get_event_loop().time()

                if event.type == "error":
                    logger.error(f"\nError: {str(event.error)}")
                # Input audio transcripts
                elif event.type == "conversation.item.input_audio_transcription.partial":
                    logger.debug(f"Partial: {event.transcript}")

                elif event.type == "conversation.item.input_audio_transcription.completed":
                    logger.info(f"\nUser: {event.transcript}")

                # Text mode output
                elif event.type == "response.output_text.done":
                    logger.info(f"Model: {event.text}")

                # Voice Audio events
                elif event.type in ("response.audio.delta", "response.output_audio.delta"):
                    output = np.frombuffer(base64.b64decode(event.delta), dtype=np.int16).reshape(-1, 1)
                    await speaker_queue.put(output)

                # Voice text transcripts
                elif event.type == "response.output_audio_transcript.done":
                    logger.info(f"Model: {event.transcript}")

                # Function Calls
                elif event.type == "response.done":
                    if len(event.response.output) > 0:
                        output = event.response.output[0]
                        if output.type == "function_call":
                            tool_name = output.name
                            if tool_name == "end_conversation":
                                logger.info("Tool call: end_conversation → stopping conversation loop.")
                                stop_event.set()
                                speaker_queue.put_nowait(None)
                                await conn.close()
                                return None
                            elif tool_name == "change_conversation_mode":
                                new_mode_req = ConversationMode(json.loads(output.arguments)["mode"])
                                logger.info(f"Tool call: change_conversation_mode to {str(new_mode_req)}")
                                if new_mode_req != mode:
                                    logger.info(f"New Mode {str(new_mode_req)} differs from {str(mode)}, switching")
                                    # Finish conversation to avoid issues with partial responses
                                    stop_event.set()
                                    speaker_queue.put_nowait(None)
                                    await conn.close()
                                    return new_mode_req
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
                                    speaker_queue.put_nowait(None)
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
                else:
                    logger.debug(f"Got unhandled event {str(event.type)}", )

        logger.info("Conversation Started... (Ctrl+C to stop)")
        watchdog_task = asyncio.create_task(idle_watchdog())
        new_mode = None
        # ToDo: Change this to only wait on receive_events(), all others get cancelled when it completes
        try:
            _, _, new_mode, _ = await asyncio.gather(
                send_audio_openai(reachy, stop_event, conn),
                send_audio_speaker(reachy, stop_event, speaker_queue),
                receive_events(),
                watchdog_task)
        finally:
            stop_event.set()
            speaker_queue.put_nowait(None)
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
            reachy.media.start_playing()
            try:
                if not skip_wakeword:
                    _ = listen_for_wakeword(reachy)
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
                reachy.media.stop_playing()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
