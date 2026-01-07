# Reachy Mini
This is a small project to explore the Reachy Mini robot.

To Do:
 - Bug: Not finding audio device
 - Wake Word (OpenWakeWord)
 - Emotions (Should be able to access the SDK emotions library)
 - Light Control (MCP Lighting Server)
 - Trigger Voice Conversation 

Other Ideas:
 - Face tracking + scan (look at you)
 - Link / embed some of the existing apps (20 questions, hand tracking)

# Installation
## Install PortAudio
This is required for OpenWakeWord, and is not installed automatically.

```bash
sudo apt install portaudio19-dev
```

## Install Python-SDK
https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/SDK/installation.md

That includes instructions for allowing access to the USB

## Install Dependencies
Run 
```
uv sync
```

# Running Daemon
HuggingFace provides a daemon that provides API for the robot.
```bash
uv run reachy-mini-daemon
```

# Wake Word
Working on this - need to find and run some existing voice model
https://huggingface.co/spaces/luisomoreau/hey_reachy_wake_word_detection

# Conversation
Looks like they have conversation with face tracking already
https://huggingface.co/spaces/pollen-robotics/reachy_mini_conversation_app/tree/main/src/reachy_mini_conversation_app